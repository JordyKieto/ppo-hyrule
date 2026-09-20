"""Stage 1: train an object-oriented causal dynamics model at the full graph G1, score it per field.

    .venv/bin/python scripts/train_oocdm.py --epochs 4
    .venv/bin/python scripts/train_oocdm.py --data datasets/transitions/v3-*/20260919-084052 --epochs 8

docs/CAUSAL_WORLD_MODEL.md section 9. One hypothesis is under test and causal discovery is NOT part of it:
does predicting each field in its own units, with class-shared parameters and inter-object attention, beat
the do-nothing floor that every latent model in LATENT_WORLD_MODEL.md failed to beat? The causal graph is
held at G1 (every edge present), so any win here is attributable to the object-oriented factorisation and
not to causality. Stage 2 adds masking and CMI discovery and changes nothing else.

WHAT IS REPORTED, AND WHY BOTH COLUMNS
--------------------------------------
Per field, in the field's OWN units, never pooled into one scalar:

  identity / model / skill          over every scored transition
  identity|chg / model|chg / skill  restricted to the transitions where the field actually CHANGED

The second block is the one that cannot be won by copying the input. Section 30 of the latent trail found a
predictor whose global optimum was "emit z_t" scoring skill exactly +0.000; with a residual-from-identity
parameterisation that failure mode is always available, so it is measured directly rather than argued about.
A model that has learned nothing scores skill 0.000 on the left and 0.000 on the right. A model that has
learned to copy scores ~0 on the left and NEGATIVE on the right.

CONTROLS RUN BEFORE TRAINING (AGENTS.md section 1)
--------------------------------------------------
  identity init     the untrained model is asserted to be EXACTLY do-nothing, numerically
  permutation       two instances of a class are swapped; every prediction must swap with them and nothing
                    else may move. Causation symmetry is architectural here, so this is a unit test
  slot split        train {1,2,3,5,6,10} / tune {4,8} / test {7,9}, matching the latent trail so numbers
                    are comparable. Codecs are fitted on TRAIN ONLY
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import (OOCDM, assert_identity_init, field_losses, fit_codecs,  # noqa: E402
                            make_batch, pack, predict_raw)
from zelda_rl.oomdp import (SCHEMA_V3, assert_partition, assert_roundtrip, decompose,  # noqa: E402
                            err, prep, unit)
from zelda_rl.state_layout import STATE_DIM_V3  # noqa: E402

DEFAULT_DATA = ["datasets/transitions/v3-main/20260919-084052"]
TRAIN_SLOTS, TUNE_SLOTS, TEST_SLOTS = {1, 2, 3, 5, 6, 10}, {4, 8}, {7, 9}


def load(dirs, limit=0):
    """-> {'train'|'tune'|'test': [(state, action)]}, split by START SLOT (the project's convention)."""
    out = {"train": [], "tune": [], "test": []}
    for d in dirs:
        d = Path(d)
        rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
        for r in (rows[:limit] if limit else rows):
            z = np.load(d / r["file"])
            st, a = z["state"], z["action"]
            assert st.shape[-1] == STATE_DIM_V3, f"{r['file']} is {st.shape[-1]}-d, not v3"
            s = r["slot"]
            key = "test" if s in TEST_SLOTS else ("tune" if s in TUNE_SLOTS else "train")
            out[key].append((st, np.r_[a, a[-1]][: len(st)]))
    return out


def check_permutation(model, batch, cname="En_Dekubaba", tol=2e-4):
    """Swap two instances of a class; every prediction must swap with them and nothing else may move."""
    if model.capacity.get(cname, 0) < 2:
        return "skipped (fewer than 2 instances)"
    with torch.no_grad():
        base = model(batch)
        perm = list(range(model.capacity[cname]))
        perm[0], perm[1] = perm[1], perm[0]
        b2 = {k: ({c: ({f: (t[:, perm] if c == cname else t) for f, t in v.items()})
                   for c, v in batch[k].items()} if k in ("x", "raw", "y", "valid")
                  else ({c: (t[:, perm] if c == cname else t) for c, t in batch[k].items()}
                        if k == "mask" else batch[k]))
              for k in batch}
        sw = model(b2)
    worst = 0.0
    for key in base:
        a, b = base[key], sw[key]
        if key.startswith(cname + "|"):
            b = b[:, perm]
        worst = max(worst, float((a - b).abs().max()))
    return f"max deviation {worst:.2e} {'OK' if worst < tol else 'FAILED'}"


class Stat:
    __slots__ = ("n", "ident", "model", "n_chg", "ident_chg", "model_chg")

    def __init__(self):
        self.n = self.n_chg = 0
        self.ident = self.model = self.ident_chg = self.model_chg = 0.0

    def add(self, f, cur, nxt, pred):
        c, n_, p = prep(f, cur), prep(f, nxt), prep(f, pred)
        ei, em = err(f, n_, c), err(f, n_, p)
        chg = ei > 0
        self.n += len(ei); self.ident += float(ei.sum()); self.model += float(em.sum())
        self.n_chg += int(chg.sum())
        self.ident_chg += float(ei[chg].sum()); self.model_chg += float(em[chg].sum())

    def row(self):
        if not self.n:
            return None
        i, m = self.ident / self.n, self.model / self.n
        ic = self.ident_chg / self.n_chg if self.n_chg else 0.0
        mc = self.model_chg / self.n_chg if self.n_chg else 0.0
        return {"n": self.n, "chg": self.n_chg / self.n, "identity": i, "model": m,
                "skill": 1 - m / i if i else float("nan"), "identity_chg": ic, "model_chg": mc,
                "skill_chg": 1 - mc / ic if ic else float("nan")}


@torch.no_grad()
def evaluate(model, p, codecs, device, batch_size=512, cap=0):
    model.eval()
    idx = p.pairs if not cap else p.pairs[:: max(1, len(p.pairs) // cap)]
    stats = {}
    for s in range(0, len(idx), batch_size):
        b = make_batch(p, codecs, model, idx[s: s + batch_size], device)
        pred = predict_raw(model, model(b), b)
        for c in model.order:
            for t in model.target_names[c.name]:
                v = b["valid"][c.name][t].cpu().numpy()
                if not v.any():
                    continue
                tt = b["t"]
                cur, nxt = p.attrs[c.name][t][tt], p.attrs[c.name][t][tt + 1]
                stats.setdefault(f"{c.name}|{t}", Stat()).add(c.field(t), cur[v], nxt[v], pred[f"{c.name}|{t}"][v])
    model.train()
    return {k: r for k, s in stats.items() if (r := s.row())}


def report(rows, model, title):
    print(f"\n=== {title} " + "=" * max(0, 96 - len(title)))
    print(f"  {'class.field':<28} {'unit':>6} {'chg%':>7} | {'identity':>9} {'model':>9} {'skill':>7} "
          f"| {'ident|chg':>9} {'model|chg':>9} {'skill|chg':>9}")
    order = sorted(rows.items(), key=lambda kv: -kv[1]["chg"])
    live = [(k, r) for k, r in order if r["chg"] > 0]
    for k, r in live:
        c, t = k.split("|")
        u = unit(next(x for x in model.order if x.name == c).field(t))
        print(f"  {k.replace('|', '.'):<28} {u:>6} {100 * r['chg']:7.3f} | {r['identity']:9.4f} "
              f"{r['model']:9.4f} {r['skill']:7.3f} | {r['identity_chg']:9.4f} {r['model_chg']:9.4f} "
              f"{r['skill_chg']:9.3f}")
    n_dead = len(order) - len(live)
    won = sum(1 for _, r in live if r["skill_chg"] > 0)
    print(f"  ({n_dead} fields never change on these episodes and are omitted)")
    print(f"  fields beating do-nothing ON CHANGED STEPS: {won} / {len(live)}")
    return live


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    ap.add_argument("--episodes", type=int, default=0, help="per collection; 0 = all")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/oocdm/stage1.pt")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print(f"schema partition: {assert_partition()}")
    t0 = time.time()
    data = load(args.data, args.episodes)
    print(f"episodes: train {len(data['train'])}, tune {len(data['tune'])}, test {len(data['test'])} "
          f"({time.time() - t0:.0f}s)")
    assert_roundtrip(*data["train"][0])

    gs = [decompose(st, a) for st, a in data["train"][:64]]
    codecs = fit_codecs(gs)                                   # TRAIN ONLY -- no test leakage into a scale
    packs = {k: pack(v) for k, v in data.items() if v}
    cap = {c: max(p.capacity[c] for p in packs.values()) for c in packs["train"].capacity}
    packs = {k: p.pad_to(cap) for k, p in packs.items()}      # one capacity for every split
    print(f"transitions: " + ", ".join(f"{k} {len(p)}" for k, p in packs.items()) + f" | capacity {cap}")

    dev = torch.device(args.device)
    model = OOCDM(codecs, cap).to(dev)
    print(f"params: {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M, "
          f"{sum(len(v) for v in model.target_names.values())} field heads")

    # Asserted on EVERY split, not just train: the first version of this passed on train and was wrong on
    # test, because codecs are fitted on train and an unseen categorical value had no identity slot.
    for k, pk in packs.items():
        assert_identity_init(model, make_batch(pk, codecs, model, pk.pairs[:256], dev))
    b0 = make_batch(packs["train"], codecs, model, packs["train"].pairs[:128], dev)
    print(f"control -- identity init: untrained model is EXACTLY do-nothing on {', '.join(packs)}  OK")
    print(f"control -- permutation:  {check_permutation(model, b0)}")

    base = evaluate(model, packs["test"], codecs, dev, cap=20000)
    worst = max(((abs(r["skill"]), k) for k, r in base.items()), default=(0.0, "-"))
    print(f"control -- untrained test skill: worst |skill| {worst[0]:.2e} on {worst[1]} "
          f"(nonzero only through the float round-trip of the angle codec)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    tr = packs["train"]
    for ep in range(args.epochs):
        perm = rng.permutation(tr.pairs)
        tot, nb, t1 = 0.0, 0, time.time()
        for s in range(0, len(perm) - args.batch, args.batch):
            b = make_batch(tr, codecs, model, perm[s: s + args.batch], dev)
            L = field_losses(model, model(b), b)
            loss = sum(L.values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        rows = evaluate(model, packs["tune"], codecs, dev, cap=20000)
        won = sum(1 for r in rows.values() if r["chg"] > 0 and r["skill_chg"] > 0)
        live = sum(1 for r in rows.values() if r["chg"] > 0)
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / max(nb, 1):8.3f}  "
              f"tune fields beating do-nothing on changed steps: {won}/{live}  "
              f"({time.time() - t1:.0f}s)")

    rows = evaluate(model, packs["test"], codecs, dev)
    live = report(rows, model, f"TEST slots {sorted(TEST_SLOTS)}, {args.epochs} epochs")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "capacity": cap, "args": vars(args)}, args.out)
    print(f"\nwrote {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps({k: v for k, v in rows.items()}, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
