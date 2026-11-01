#!/usr/bin/env python3
"""Phase 1's RESULT: read-only probes on a frozen sjepa encoder, against three floors.

    .venv/bin/python -u scripts/probe_sjepa.py --ckpt runs/sjepa/p1.last.pt --data 'datasets/transitions/v3-*'

NO GRADIENT REACHES THE ENCODER. Every head here is fitted closed-form on frozen features (DINO-WM 2411.04983
reports decoder gradients hurt, and keeping the readout post-hoc is what makes "the objective discarded this"
a statement about the objective rather than about the head). `AGENTS.md` 2 requires informativeness beside
predictability; this script is the informativeness half, and the trainer's loss is not a result without it.

WHY THE PROBES READ A *MASKED* ENCODING -- the design trap, stated because it is easy to get backwards.
V-JEPA probes a COMPLETE clip for a label the pixels do not contain (an ImageNet class). Here every candidate
label IS a field of the input: probing an unmasked state for actor `id` or `pos_x` is the identity map, would
score ~1.0 on everything, and would measure nothing at all. So the probe encodes a MASKED state and predicts
the HIDDEN content. That is the capability the pretext is supposed to buy -- infer what you cannot see from
what you can -- and it is the only version of the question with a non-trivial answer.

THE THREE FLOORS, all mandatory (AGENTS 1). A number without them is not a result:

  untrained   same architecture, random weights, same mask, same head. `train_jepa_pure.py` measured a
              trained predictive encoder WORSE than this on 7 of 10 reward fields (link_pos 35.1 vs 29.0),
              so this is the falsification bar, not a formality.
  trivial     mean (continuous) or majority class (discrete), ignoring the input entirely.
  visible     ridge on the VISIBLE RAW CELLS directly, no encoder. This is the free-signal control, and it
              is the one most likely to kill a result here: `speed` = |vel|, and pos_t is pos_{t-1} + vel_t,
              so a masked cell can be exact arithmetic on a visible one. AGENTS 1's +0.742 collapsed to
              -0.012 under exactly this class of ablation. If the encoder does not beat `visible`, it learned
              arithmetic that a linear map already had.

PROBE FAMILIES, mapped from V-JEPA's own evaluation axes:

  semantic     discrete field recovery: actor `id`/`cat`/`state_idx`/`fn_rel`, Link `action_fn_rel`,
               `melee_state`, `bg_flags`. Accuracy vs the majority floor. (V-JEPA: ImageNet/CIFAR/iNat.)
  spatial      counting (how many actors present / alive) and "depth" (distance Link->nearest actor, per-actor
               distance, dist_interact_wall, floor_height). MAE in world units vs the mean floor.
               (V-JEPA: object counting and depth prediction.)
  lowshot      every probe above refitted on 1% and 10% of the training frames. (V-JEPA: low-shot.)

Reported with `n` on every row. A probe fitted on 1% of a small split is a sample, not a measurement
(AGENTS 38), and the low-shot columns are the ones most able to mislead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_sjepa import episode_files, load_clips                       # noqa: E402
from zelda_rl.sjepa import MaskSampler, Prep, SJepa, Schema, tok_targets        # noqa: E402

# EXPERIMENT-CONTRACT: exempt -- opens no Dolphin (no make_env, no DOLPHIN_BIN), so AGENTS 28's thread pin
# does not apply. Results are written to --out as JSON and printed; nothing is evaluated into stdout alone.


def ridge(X, Y, lam=1.0):
    """Closed-form ridge. X (n,d), Y (n,k) -> W (d+1,k) with a bias column appended to X.

    Solved in the DUAL when d > n, which is mathematically identical and is the difference between seconds
    and minutes here: the `visible` control is T*E*F = 7,344 features against ~1,000 training rows, so the
    primal normal equations are a 7345^3 solve (~4e11 flops) repeated for every probe and every low-shot
    fraction. The dual is n^3. Without this the free-signal control -- the floor most likely to kill the
    result -- is the reason the script never finishes, which is the worst possible thing to drop for speed.
    """
    X = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
    n, d = X.shape
    if d > n:
        K = X @ X.T + lam * np.eye(n, dtype=X.dtype)
        return X.T @ np.linalg.solve(K, Y)
    A = X.T @ X + lam * np.eye(d, dtype=X.dtype)
    return np.linalg.solve(A, X.T @ Y)


def apply_ridge(W, X):
    return np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1) @ W


def fit_regress(Xtr, ytr, Xte, yte, lam=1.0):
    """-> MAE of a ridge, in the target's own units."""
    W = ridge(Xtr.astype(np.float64), ytr.astype(np.float64)[:, None], lam)
    p = apply_ridge(W, Xte.astype(np.float64))[:, 0]
    return float(np.abs(p - yte).mean())


def fit_classify(Xtr, ytr, Xte, yte, lam=1.0):
    """-> accuracy of a ridge onto one-hot targets (argmax readout). AGENTS 2: 'ridge is fine'."""
    cls = np.unique(ytr)
    if cls.size < 2:
        return float("nan"), cls.size
    oh = (ytr[:, None] == cls[None, :]).astype(np.float64)
    W = ridge(Xtr.astype(np.float64), oh, lam)
    p = apply_ridge(W, Xte.astype(np.float64))
    return float((cls[p.argmax(1)] == yte).mean()), int(cls.size)


# ---- targets -----------------------------------------------------------------------------------------
SEMANTIC_LINK = ("action_fn_rel", "melee_state", "bg_flags", "targeted_row", "melee_anim")
SEMANTIC_ROW = ("id", "cat", "state_idx", "fn_rel", "present")
DEPTH_LINK = ("dist_interact_wall", "floor_height")


def build_targets(schema: Schema, raw, cell_masked, t_sel: int):
    """Per-clip probe targets at frame `t_sel`, each with the mask status of the cell(s) it comes from.

    A target is only scored where it was actually HIDDEN -- that is the whole question. Scoring a target whose
    cell was visible measures the identity map, which is the trap this script's docstring is about.
    """
    g = schema.unpack(raw[:, t_sel])                       # (N, E, F) raw values at the probed frame
    cm = cell_masked[:, t_sel]                             # (N, E, F) bool
    out = {}
    fi = schema.fidx

    def link(name):
        f = fi[name]
        return g[:, 0, f], cm[:, 0, f]

    for n in SEMANTIC_LINK:
        if schema.valid[0, fi[n]]:
            out[f"semantic/link_{n}"] = (*link(n), "class")
    for n in DEPTH_LINK:
        if schema.valid[0, fi[n]]:
            out[f"spatial/link_{n}"] = (*link(n), "reg")

    # actor row 1 (the first actor slot) is probed individually, and the counting/depth targets pool rows
    r = 1
    for n in SEMANTIC_ROW:
        out[f"semantic/row1_{n}"] = (g[:, r, fi[n]], cm[:, r, fi[n]], "class")

    pres = g[:, 1:, fi["present"]]
    hp = g[:, 1:, fi["hp"]]
    # COUNTING: how many actors are present / alive. Scored where EVERY contributing cell was hidden, so the
    # count cannot be read off a visible row -- a partially visible scene makes counting a subtraction.
    pres_m = cm[:, 1:, fi["present"]].all(1)
    out["spatial/count_present"] = (pres.sum(1), pres_m, "reg")
    out["spatial/count_alive"] = (((pres > 0) & (hp > 0)).sum(1), pres_m & cm[:, 1:, fi["hp"]].all(1), "reg")

    # DEPTH: Link -> nearest present actor, in world units. Needs Link's pos and the actors' pos hidden.
    lp = np.stack([g[:, 0, fi[f"pos_{c}"]] for c in "xyz"], 1)
    ap = np.stack([np.stack([g[:, e, fi[f"pos_{c}"]] for c in "xyz"], 1) for e in range(1, schema.n_ent)], 1)
    d = np.linalg.norm(ap - lp[:, None], axis=-1)
    d = np.where(pres > 0, d, np.inf)
    near = np.where(np.isfinite(d).any(1), d.min(1), np.nan)
    pos_cols = [fi[f"pos_{c}"] for c in "xyz"]
    hid = cm[:, 0, pos_cols].all(1) & cm[:, 1:, :][:, :, pos_cols].all((1, 2))
    ok = np.isfinite(near)
    out["spatial/nearest_dist"] = (np.where(ok, near, 0.0), hid & ok, "reg")
    return out


def features(model, prep, raw, cell_masked, t_sel, dev, schema, kind="grid", batch=512):
    """Frozen encoder features. `kind`:

      grid    every token, all frames, flattened: T*E*d. THE DEFAULT, and the only fair one.
      frame   every token at the probed frame: E*d.
      pooled  Link's token plus the mean over actor tokens at the probed frame: 2*d.

    `pooled` was the first and only implementation and it quietly rigged the comparison. It gives the encoder
    arm 2*96 = 192 features while the `visible` free-signal control gets T*E*F = 7,344 raw cells -- a 38x
    capacity advantage to the control. "SJEPA loses to visible" would then be unreadable: worse representation,
    or 38x less to regress on? AGENTS 12 is the same error in another guise (a ratio read without its
    denominator), and AGENTS 35 is why a comparison has to be matched before it decides anything.

    `grid` gives the encoder T*E*d = 13,056 against the control's 7,344, so if anything the capacity now
    favours the encoder slightly and a LOSS is the conservative, trustworthy direction.
    """
    outs = []
    for i in range(0, len(raw), batch):                      # batched: the full grid is a large activation
        x, ix = prep.encode(raw[i:i + batch])
        with torch.no_grad():
            z = model.encode(torch.from_numpy(x).to(dev), torch.from_numpy(ix).to(dev),
                             torch.from_numpy(cell_masked[i:i + batch]).to(dev))
        if kind == "grid":
            f = z.reshape(len(z), -1)
        elif kind == "frame":
            f = z[:, t_sel].reshape(len(z), -1)
        elif kind == "pooled":
            zt = z[:, t_sel]
            f = torch.cat([zt[:, 0], zt[:, 1:].mean(1)], -1)
        else:
            raise ValueError(f"unknown feature kind {kind!r}")
        outs.append(f.float().cpu().numpy())
    return np.concatenate(outs)


def visible_features(schema: Schema, prep: Prep, raw, cell_masked, t_sel):
    """The free-signal control: the raw VISIBLE cells, z-scored, masked cells zeroed, flattened.

    This is what a linear map can get without any representation learning at all, and it is the floor most
    likely to beat the encoder: `speed` = |vel|, pos_t = pos_{t-1} + vel_t. AGENTS 1 exists because a +0.742
    became -0.012 once exactly this class of signal was removed.
    """
    x, _ = prep.encode(raw)
    v = np.broadcast_to(schema.valid, cell_masked.shape)
    x = x * (v & ~cell_masked)
    return x.reshape(len(x), -1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--clip-len", type=int, default=0, help="0 = take it from the checkpoint's config")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--limit-clips", type=int, default=6000)
    ap.add_argument("--mask-ratio", type=float, default=0.0, help="0 = the checkpoint's training ratio")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--lowshot", default="0.01,0.1")
    ap.add_argument("--min-n", type=int, default=200, help="a probe with fewer hidden examples is reported "
                    "as n/a rather than as a number (AGENTS 38: state the n or do not quote it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--feat", default="grid", choices=("grid", "frame", "pooled"),
                    help="encoder feature form; grid = all T*E tokens, the only one capacity-matched to the "
                         "visible control")
    ap.add_argument("--out", default="")
    a, unknown = ap.parse_known_args()
    if unknown:
        print(f"ignoring unrecognised args: {unknown}", flush=True)
    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)

    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    prep = Prep.from_json(ck["prep"])
    schema = prep.s
    T = a.clip_len or int(cfg["clip_len"])
    ratio = a.mask_ratio or float(cfg["mask_ratio"])
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")

    rng = np.random.default_rng(a.seed)
    clips, _src = load_clips(episode_files(a.data), T, stride=a.stride, limit=a.limit_clips)
    mw = {k: float(v) for k, v in (kv.split("=") for kv in cfg["mask_weights"].split(","))}
    ms = MaskSampler(schema, ratio=ratio, weights=mw, rng=np.random.default_rng(a.seed))
    cm = np.stack([ms.sample(T)[0] for _ in range(len(clips))])
    t_sel = T // 2
    print(f"probing {len(clips)} clips, T={T}, frame {t_sel}, mask ratio {ratio} {mw}", flush=True)

    def build(d: int, depth: int, trained: bool):
        m = SJepa(schema, prep, d=d, depth=depth, heads=int(cfg["heads"]),
                  pred_depth=int(cfg["pred_depth"]), ema=float(cfg["ema"]))
        # The EMA target encoder is built lazily on first use, so a freshly constructed model has no
        # `tok_t`/`enc_t` keys and a strict load of a trained checkpoint fails on 90 unexpected ones.
        # Building it first also means the untrained arm has the SAME parameter count as the trained one.
        m._ensure_target()
        if trained:
            m.load_state_dict(ck["model"])
        return m.to(dev).eval()

    torch.manual_seed(a.seed + 1)         # the untrained control is a DIFFERENT draw from the training init
    arms = {
        "sjepa": features(build(int(cfg["dim"]), int(cfg["depth"]), True), prep, clips, cm, t_sel, dev,
                          schema, a.feat),
        "untrained": features(build(int(cfg["dim"]), int(cfg["depth"]), False), prep, clips, cm, t_sel, dev,
                              schema, a.feat),
        "visible": visible_features(schema, prep, clips, cm, t_sel),
    }
    print("feature widths: " + " ".join(f"{k}={v.shape[1]}" for k, v in arms.items()) +
          "  (matched by construction: a 38x capacity gap to the control made the verdict unreadable)",
          flush=True)
    tgts = build_targets(schema, clips, cm, t_sel)

    n = len(clips)
    perm = rng.permutation(n)
    ntr = int(a.train_frac * n)
    tr_i, te_i = perm[:ntr], perm[ntr:]
    shots = [1.0] + [float(s) for s in a.lowshot.split(",") if s.strip()]

    results: list[dict] = []
    for name, (y, hid, kind) in sorted(tgts.items()):
        tr = tr_i[hid[tr_i]]
        te = te_i[hid[te_i]]
        if len(te) < a.min_n or len(tr) < a.min_n:
            results.append({"probe": name, "kind": kind, "n_train": int(len(tr)), "n_test": int(len(te)),
                            "status": f"n/a: {len(tr)} train / {len(te)} test hidden examples "
                                      f"< --min-n {a.min_n}"})
            continue
        # A target with one value is unmeasurable, not badly predicted. The first version let it through:
        # fit_classify returned nan for <2 classes, every nan comparison is False, and the verdict logic
        # therefore printed "loses to untrained" for five row1_* probes that were CONSTANT in this dataset.
        # A check that reports failure on correct work is ignored within a day (AGENTS 36 rows 2 and 6), and
        # one that reports it on an unmeasurable quantity is worse -- it invents a negative result.
        uniq = int(np.unique(y[np.concatenate([tr, te])]).size)
        if uniq < 2:
            results.append({"probe": name, "kind": kind, "n_train": int(len(tr)), "n_test": int(len(te)),
                            "status": f"n/a: target is CONSTANT ({uniq} distinct value) in this dataset "
                                      f"-- nothing to predict"})
            continue
        row = {"probe": name, "kind": kind, "n_train": int(len(tr)), "n_test": int(len(te)),
               "n_distinct": uniq}
        if kind == "reg":
            row["trivial"] = float(np.abs(y[te] - y[tr].mean()).mean())
        else:
            vals, cnt = np.unique(y[tr], return_counts=True)
            row["trivial"] = float((y[te] == vals[cnt.argmax()]).mean())
            row["n_classes"] = int(np.unique(y[tr]).size)
        for shot in shots:
            k = max(a.min_n, int(shot * len(tr)))
            sub = tr[:k]
            for arm, X in arms.items():
                key = arm if shot == 1.0 else f"{arm}@{shot:g}"
                row[key] = (fit_regress(X[sub], y[sub], X[te], y[te]) if kind == "reg"
                            else fit_classify(X[sub], y[sub], X[te], y[te])[0])
            row[f"n_train@{shot:g}" if shot != 1.0 else "n_train_full"] = int(k)
        results.append(row)

    # ---- report ----
    print(f"\n{'probe':34s}{'kind':6s}{'n_te':>7s}{'trivial':>10s}{'visible':>10s}"
          f"{'untrained':>11s}{'SJEPA':>10s}{'  verdict'}", flush=True)
    better = lambda k, x, y_: (x < y_) if k == "reg" else (x > y_)                              # noqa: E731
    wins = losses = 0
    for r in results:
        if "status" in r:
            print(f"{r['probe']:34s}{r['kind']:6s}{r['n_test']:>7d}   {r['status']}", flush=True)
            continue
        k = r["kind"]
        if not all(np.isfinite(r.get(x, np.nan)) for x in ("sjepa", "untrained", "visible")):
            print(f"{r['probe']:34s}{k:6s}{r['n_test']:>7d}   n/a: an arm returned nan "
                  f"(sjepa {r.get('sjepa')}, untrained {r.get('untrained')}, visible {r.get('visible')})",
                  flush=True)
            continue
        v = ("beats both" if better(k, r["sjepa"], r["untrained"]) and better(k, r["sjepa"], r["visible"])
             else "loses to untrained" if not better(k, r["sjepa"], r["untrained"])
             else "loses to visible")
        wins += v == "beats both"
        losses += v != "beats both"
        print(f"{r['probe']:34s}{k:6s}{r['n_test']:>7d}{r['trivial']:>10.4f}{r['visible']:>10.4f}"
              f"{r['untrained']:>11.4f}{r['sjepa']:>10.4f}  {v}", flush=True)
    # LOW-SHOT: the same frozen features, head refitted on 1% / 10% of the hidden training examples.
    # V-JEPA's third axis. It is reported with its n on every cell because a head fitted on a few hundred
    # rows is a sample, not a measurement (AGENTS 38), and the low-shot columns are the ones most able to
    # mislead -- a representation can look better here purely by having fewer effective degrees of freedom.
    extra = [f"{x:g}" for x in shots if x != 1.0]
    if extra:
        print(f"\nLOW-SHOT (head refitted on a fraction of the hidden train set; --min-n floors it at "
              f"{a.min_n})", flush=True)
        hdr = "".join(f"{'sj@' + e:>10s}{'un@' + e:>10s}{'vi@' + e:>10s}" for e in extra)
        print(f"{'probe':34s}{'n_full':>8s}" + "".join(f"{'n@' + e:>8s}" for e in extra) + hdr, flush=True)
        for r in results:
            if "status" in r or not all(np.isfinite(r.get(x, np.nan))
                                        for x in ("sjepa", "untrained", "visible")):
                continue
            ns = "".join(f"{r.get('n_train@' + e, 0):>8d}" for e in extra)
            cells = "".join(f"{r.get('sjepa@' + e, float('nan')):>10.4f}"
                            f"{r.get('untrained@' + e, float('nan')):>10.4f}"
                            f"{r.get('visible@' + e, float('nan')):>10.4f}" for e in extra)
            print(f"{r['probe']:34s}{r.get('n_train_full', 0):>8d}{ns}{cells}", flush=True)

    scored = wins + losses
    print(f"\nSJEPA beats BOTH floors on {wins}/{scored} scored probes.", flush=True)
    print("AGENTS 1/2: `trivial` is the do-nothing floor, `untrained` is the same architecture with random\n"
          "weights, `visible` is a ridge on the visible raw cells (the free-signal control). A win needs to\n"
          "clear untrained AND visible -- clearing only `trivial` means the task has structure, not that\n"
          "this encoder found any.", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"ckpt": str(a.ckpt), "config": cfg, "ratio": ratio,
                                           "n_clips": int(n), "results": results}, indent=2))
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
