#!/usr/bin/env python3
"""Phase 1: V-JEPA's masked-prediction pretext on the structured game state. No action, no future, no reward.

    .venv/bin/python -u scripts/train_sjepa.py --data 'datasets/transitions/v3-*' --epochs 20 \
        --out runs/sjepa/p1.pt --prep configs/sjepa_prep_v3.json --fit-prep

WHAT IS BEING TESTED, and the bar it has to clear. Four latent world models in this repo optimise
`z_{t+1} = T(z_t, a_t)` and all four lose to the identity, because the latent moves 0.023-0.05 of a random
pair per step and copying is therefore near-optimal. The JEPA papers pose a different task -- mask part of the
INPUT, predict the target encoder's representation of the masked part -- and that task has no persistence
shortcut. It was never run here. This is it.

The result is NOT this script's loss (AGENTS 8). It is `scripts/probe_sjepa.py`: freeze the encoder, fit
read-only probes, and compare against an untrained encoder of the same architecture. `train_jepa_pure.py`
already measured that a predictive-only objective can end up WORSE than random initialisation on 7 of 10
reward fields, so that control is the falsification, and it decides whether phase 2 happens at all.

Nothing here trains through a probe. The only gradient is the masked-prediction L1.

WHAT IS LOGGED EVERY EPOCH, and why each column is mandatory:

    loss                  the objective. On its own it is not evidence of anything (AGENTS 8).
    eff_rank / min_std    AGENTS 2: a constant encoder is the global optimum of any pure prediction loss,
                          so predictability is never reported without informativeness. VQ scored skill
                          +0.000 by collapsing to the identity and the loss looked fine.
    mask_ratio_achieved   the requested ratio and the achieved ratio are different numbers (AGENTS 36).
    tok_target_frac       what share of tokens the loss actually scored. A mixture that stops producing
                          targets would otherwise show up only as a suspiciously good loss.
    oov                   discrete values unseen at prep time, counted on VALID cells only.

RESUME IS NOT OPTIONAL (AGENTS 39). Three arms were restarted to change a flag in one session and each began
from a random init because `train_rvs.py` had no resume and only checkpointed on improvement; one lost 1.54
online boss damage. So: every epoch writes `<out>.last.pt` with the optimiser state unconditionally, starting
on top of existing progress EXITS unless `--resume` or `--fresh` is passed, and `tests/test_sjepa_resume.py`
asserts the loss CURVE continues and the weights are closer to the checkpoint than to a fresh init -- not
that the word RESUMED was printed, which is what shipped the first time and was rightly rejected.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zelda_rl import ckpt_s3                                                    # noqa: E402
from zelda_rl.eval_log import EvalSink                                          # noqa: E402
from zelda_rl.sjepa import (MaskSampler, Prep, SJepa, Schema, collapse_stats,   # noqa: E402
                            tok_targets)
from zelda_rl.sjepa import _ln as _lnorm                                        # noqa: E402

# This script opens no Dolphin, so AGENTS 28's one-thread pin does not apply; it wants every core it can get.
# EXPERIMENT-CONTRACT: exempt -- no make_env, no DOLPHIN_BIN. Checkpoints go through ckpt_s3 and metrics
# through EvalSink, both below; the exemption is only for the thread pin, which would halve throughput here.


def episode_files(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        for root in sorted(glob.glob(pat)):
            out += sorted(Path(root).rglob("episodes/*.npz"))
    if not out:
        raise SystemExit(f"no episodes under {patterns} (expected <dir>/<stamp>/episodes/*.npz)")
    return out


def load_clips(files: list[Path], T: int, *, stride: int = 1, limit: int = 0):
    """-> (clips (N,T,width) float32, per-clip source episode index).

    Clips never cross an episode boundary: a window spanning a reset would ask the model to predict a state
    from an unrelated start, and `LATENT_WORLD_MODEL.md` 6 records the opposite error (exit frames that
    trained on nothing) costing a whole line of work. Short episodes are SKIPPED and counted -- an episode
    with fewer than T frames would otherwise hit the `W[-0:]` class of bug that cost 17 sweep arms
    (AGENTS 36 row 1).
    """
    clips, src, short, frames = [], [], 0, 0
    for i, f in enumerate(files):
        with np.load(f) as z:
            st = z["state"].astype(np.float32)
        frames += st.shape[0]
        if st.shape[0] < T:
            short += 1
            continue
        for t0 in range(0, st.shape[0] - T + 1, stride):
            clips.append(st[t0:t0 + T])
            src.append(i)
        if limit and len(clips) >= limit:
            break
    if not clips:
        raise SystemExit(f"no clips of length {T} from {len(files)} episodes ({short} too short)")
    print(f"clips: {len(clips)} of T={T} from {len(files)} episodes / {frames} frames "
          f"({short} episodes shorter than T, skipped)", flush=True)
    return np.stack(clips), np.array(src)


def save_ckpt(path: Path, model, opt, epoch: int, cfg: dict, prep: Prep, *, run_name: str) -> None:
    """The ONE writer, for both `.last.pt` and the best-of file -- so they cannot drift in metadata or in
    optimiser state. `train_rvs.py` saved the optimiser as None, which made even a hand-rolled resume restart
    Adam cold (AGENTS 39)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                "config": cfg, "prep": prep.to_json(), "phase": 1}, path)
    ckpt_s3.upload(path, run_name, epoch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="repeatable glob of collection dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", required=True, help="fitted normalisation/vocab JSON; --fit-prep writes it")
    ap.add_argument("--fit-prep", action="store_true")
    ap.add_argument("--clip-len", type=int, default=8, help="T: frames per clip (the 'video' axis)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--tok-target-frac", type=float, default=0.5)
    ap.add_argument("--mask-weights", default="entity_tube=0.3,group_tube=0.3,time_block=0.2,cell=0.2")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--pred-depth", type=int, default=2)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--clip-grad", type=float, default=0.0, help="0 = OFF. The JEPA's grad ran 105-120 "
                    "against --clip 20 for 28 epochs, scaling every step 5-6x; see WM_EXPERIMENT_REGIME")
    ap.add_argument("--val-frac", type=float, default=0.15, help="held out by EPISODE, not by frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0, help="0 = leave torch's default")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb-project", default="zelda-sjepa")
    ap.add_argument("--wandb-group", default=None)
    a, unknown = ap.parse_known_args()
    if unknown:
        print(f"ignoring unrecognised args (fleet launch injects its own): {unknown}", flush=True)
    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)

    out = Path(a.out)
    last = out.with_suffix(".last.pt")
    if last.exists() and not (a.resume or a.fresh):
        raise SystemExit(f"{last} already holds training progress (epoch "
                         f"{torch.load(last, weights_only=False, map_location='cpu').get('epoch')}). "
                         f"Pass --resume to continue from it, or --fresh to deliberately start over.")

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    files = episode_files(a.data)
    clips, src = load_clips(files, a.clip_len, stride=a.stride, limit=a.limit_clips)
    width = clips.shape[-1]
    schema = Schema(width)

    # Split by EPISODE. A frame-level split leaks: consecutive frames of a 60 Hz game are nearly identical, so
    # a random frame split puts a target's own neighbours in the training set and every held-out number is
    # optimistic. LATENT_WORLD_MODEL uses slot splits for the same reason.
    n_ep = len(files)
    perm = rng.permutation(n_ep)
    n_val = max(1, int(a.val_frac * n_ep))
    val_ep = set(perm[:n_val].tolist())
    is_val = np.array([s in val_ep for s in src])
    tr, va = clips[~is_val], clips[is_val]
    print(f"split: {len(tr)} train clips / {len(va)} val clips over {n_ep - n_val}/{n_val} episodes",
          flush=True)

    if a.fit_prep:
        prep = Prep.fit(schema, tr.reshape(-1, width))
        Path(a.prep).parent.mkdir(parents=True, exist_ok=True)
        Path(a.prep).write_text(json.dumps(prep.to_json()))
        print(f"wrote {a.prep}", flush=True)
    else:
        if not Path(a.prep).is_file():
            raise SystemExit(f"--prep {a.prep} not found. Fit it once with --fit-prep; fitting it per run "
                             f"(or per pod) makes two runs' numbers incomparable -- see MAP.md on --prep.")
        prep = Prep.from_json(json.loads(Path(a.prep).read_text()))
        if prep.s.width != width:
            raise SystemExit(f"--prep was fitted on width {prep.s.width}, data is {width}")

    mw = {k: float(v) for k, v in (kv.split("=") for kv in a.mask_weights.split(","))}
    ms = MaskSampler(schema, ratio=a.mask_ratio, weights=mw, rng=rng)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = SJepa(schema, prep, d=a.dim, depth=a.depth, heads=a.heads,
                  pred_depth=a.pred_depth, ema=a.ema).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cfg = {**vars(a), "width": width, "n_params": sum(p.numel() for p in model.parameters()),
           "n_train_clips": int(len(tr)), "n_val_clips": int(len(va)), "n_episodes": n_ep}
    start = 1
    if a.resume and last.exists():
        ck = torch.load(last, weights_only=False, map_location=dev)
        # The EMA target encoder is built lazily on first use, so a freshly constructed model has no
        # `tok_t`/`enc_t` and a strict load fails on ~90 unexpected keys -- i.e. --resume was BROKEN and
        # would have died rather than continued. Found by tests/test_sjepa_resume.py, which is the whole
        # reason that test exists (AGENTS 39: a fix is not shipped until a test fails without it).
        model._ensure_target()
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = int(ck["epoch"]) + 1
        print(f"RESUMED from {last} at epoch {ck['epoch']} -> continuing at {start}", flush=True)
    print(f"device {dev} | {cfg['n_params'] / 1e6:.2f}M params | mask {a.mask_ratio} {mw}", flush=True)

    run_name = a.run_name or out.stem
    sink = EvalSink(run_label=run_name, project=a.wandb_project, group=a.wandb_group, config=cfg)

    def batches(data, shuffle):
        order = rng.permutation(len(data)) if shuffle else np.arange(len(data))
        for i in range(0, len(order), a.batch):
            sel = order[i:i + a.batch]
            if len(sel) == 0:
                continue
            raw = data[sel]
            x, ix = prep.encode(raw)
            cm = np.stack([ms.sample(a.clip_len)[0] for _ in sel])
            tt = tok_targets(schema, cm, a.tok_target_frac)
            if not tt.any():                       # cannot happen at sane ratios; assert rather than nan out
                continue
            yield (torch.from_numpy(x).to(dev), torch.from_numpy(ix).to(dev),
                   torch.from_numpy(cm).to(dev), torch.from_numpy(tt).to(dev),
                   float(cm.sum()) / float(np.broadcast_to(schema.valid, cm.shape).sum()),
                   float(tt.mean()), prep.last_oov)

    for ep in range(start, a.epochs + 1):
        t0 = time.time()
        model.train()
        tl, n, mr, tf, oov = 0.0, 0, [], [], 0
        for x, ix, cm, tt, r, f, o in batches(tr, True):
            loss, _ = model(x, ix, cm, tt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               a.clip_grad if a.clip_grad > 0 else 1e9)
            opt.step()
            model.update_target()
            tl += float(loss.detach()); n += 1
            mr.append(r); tf.append(f); oov += o
        model.eval()
        vl, vn = 0.0, 0
        shuf, ctrl_n = 0.0, 0
        with torch.no_grad():
            for x, ix, cm, tt, *_ in batches(va, False):
                vl += float(model(x, ix, cm, tt)[0]); vn += 1
                # THE CONTROL THAT CAUGHT THE LAST FATAL BUG, now running every epoch instead of living in a
                # scratchpad script I have to remember (AGENTS 21/33/36: a rule that stays prose gets broken).
                # `shuffle` predicts a target token's representation using a DIFFERENT CLIP's representation
                # at the same token position. If it is as good as the model, the target reps do not depend on
                # the input and the loss is measuring a positional constant. At epoch 2 of the first run the
                # model scored 0.0302 and this scored 0.0029 -- ten times better -- while every other column
                # on this line looked healthy.
                t = model.encode_target(x, ix)
                shuf += float(torch.nn.functional.l1_loss(
                    _lnorm(torch.roll(t, 1, dims=0)[tt]), _lnorm(t[tt])))
                ctrl_n += 1
            xs, ixs = prep.encode(va[:min(256, len(va))])
            z = model.encode(torch.from_numpy(xs).to(dev), torch.from_numpy(ixs).to(dev))
            cs = collapse_stats(z)
            # Variance split: does the representation move with the STATE, or only with token position?
            # collapse_stats pools both and therefore cannot see positional collapse -- eff_rank 17.9/96 and
            # min_std 0.657 were both "healthy" on a representation that was 98.7% positional.
            per_pos = z.mean(0)
            acl = float((z - per_pos[None]).std(0).mean())
            apos = float(per_pos.reshape(-1, z.shape[-1]).std(0).mean())
            cs["clip_over_pos"] = acl / max(apos, 1e-9)
        m = {"epoch": ep, "train_loss": tl / max(1, n), "val_loss": vl / max(1, vn),
             "shuffle_floor": shuf / max(1, ctrl_n),
             "mask_ratio_achieved": float(np.mean(mr)), "tok_target_frac": float(np.mean(tf)),
             "grad_norm": float(gn), "oov_cells": oov, "secs": time.time() - t0, **cs}
        # The two controls LEAD the line, before the loss, because the loss is not the result (AGENTS 8)
        # and because a headline that appears alone gets read alone (AGENTS 34).
        flag = "" if m["shuffle_floor"] > m["val_loss"] else "  <<< SHUFFLE BEATS THE MODEL: the target reps "\
                                                             "do not depend on the input"
        if cs["clip_over_pos"] < 0.1:
            flag += "  <<< POSITIONAL COLLAPSE: state moves the rep <10% as much as token position does"
        print(f"ep {ep:3d} shuffle {m['shuffle_floor']:.4f} vs val {m['val_loss']:.4f} | "
              f"clip/pos {cs['clip_over_pos']:.3f} eff_rank {cs['eff_rank']:.1f}/{a.dim} "
              f"min_std {cs['min_std']:.3f} | train {m['train_loss']:.4f} | "
              f"mask {m['mask_ratio_achieved']:.3f} tgt_tok {m['tok_target_frac']:.3f} "
              f"grad {m['grad_norm']:.1f} oov {oov} | {m['secs']:.0f}s{flag}", flush=True)
        # `rows=[m]` is not decoration. EvalSink.log writes the jsonl ONLY `if rows` and otherwise sends
        # `metrics` to W&B alone -- so a trainer reporting per-epoch aggregates with no per-episode rows has
        # its numbers durable NOWHERE, which is how the WM fleet's ~40 epochs of key-event numbers were lost
        # (AGENTS 43). Passing the metrics AS a row puts them in the jsonl and therefore in S3.
        sink.log(step=ep, metrics=m, rows=[m])
        save_ckpt(last, model, opt, ep, cfg, prep, run_name=run_name)
    sink.close()
    print(f"done. encoder for probing: {last}\n"
          f"NEXT, and the result is NOT the loss above:\n"
          f"  .venv/bin/python -u scripts/probe_sjepa.py --ckpt {last} --data {a.data[0]!r}", flush=True)


if __name__ == "__main__":
    main()
