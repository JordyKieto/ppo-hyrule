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

LOGGING
-------
Uses the project's RunLogger, so a run writes runs/<name>/train.log + updates.jsonl and streams to wandb
(project `zelda-rl-oocdm`, override with --wandb-project or $WANDB_PROJECT). Pass --no-wandb for a local-only
run. Per-field skill is logged per epoch as skill/<Class.field> and skill_chg/<Class.field> -- 77 series
rather than one headline, because a single pooled number over these fields is 36% composition (Stage 0).

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
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import (OOCDM, OOCGraph, assert_identity_init, field_losses,  # noqa: E402
                            fit_codecs, make_batch, pack, predict_raw, velocity_baseline)
from zelda_rl.oomdp import (SCHEMA_V3, assert_partition, assert_roundtrip, decompose,  # noqa: E402
                            err, prep, unit)
from zelda_rl.runlog import LOGGER, RunLogger  # noqa: E402
from zelda_rl.state_layout import STATE_DIM_V3  # noqa: E402

DEFAULT_DATA = ["datasets/transitions/v3-main/20260919-084052"]
TRAIN_SLOTS, TUNE_SLOTS, TEST_SLOTS = {1, 2, 3, 5, 6, 10}, {4, 8}, {7, 9}
# P0's hand-specified parent set for Link.pos', from the ridge probe in docs/CAUSAL_WORLD_MODEL.md 16.
HAND_PARENTS = ["pos", "vel", "speed", "stick", "bg_flags"]


def load(dirs, limit=0, sample_seed=0):
    """-> {'train'|'tune'|'test': [(state, action)]}, split by START SLOT (the project's convention).

    `limit` is a TOTAL across collections, sampled at random. Taking the first N of one collection is not a
    subset, it is a different problem: v3-main is combat where Link often stands still, and Link.pos changes
    on 0.6% of its first 60 episodes against 8.3% on the real test split.
    """
    catalogue = []
    for d in dirs:
        d = Path(d)
        catalogue += [(d, r) for r in (json.loads(l) for l in
                                       (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    if limit:
        keep = sorted(np.random.default_rng(sample_seed).permutation(len(catalogue))[:limit])
        catalogue = [catalogue[i] for i in keep]
    out = {"train": [], "tune": [], "test": []}
    for d, r in catalogue:
        if True:
            z = np.load(d / r["file"])
            st, a = z["state"], z["action"]
            assert st.shape[-1] == STATE_DIM_V3, f"{r['file']} is {st.shape[-1]}-d, not v3"
            slot = r["slot"]
            key = "test" if slot in TEST_SLOTS else ("tune" if slot in TUNE_SLOTS else "train")
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
    assert base, "forward() returned nothing -- the permutation control would pass vacuously"
    worst = 0.0
    for key in base:
        a, b = base[key], sw[key]
        if key.startswith(cname + "|"):
            b = b[:, perm]
        worst = max(worst, float((a - b).abs().max()))
    return f"max deviation {worst:.2e} over {len(base)} heads {'OK' if worst < tol else 'FAILED'}"


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
    LOGGER.info("=== %s %s", title, "=" * max(0, 90 - len(title)))
    LOGGER.info("  %-28s %6s %7s | %9s %9s %7s | %9s %9s %9s", "class.field", "unit", "chg%",
                "identity", "model", "skill", "ident|chg", "model|chg", "skill|chg")
    order = sorted(rows.items(), key=lambda kv: -kv[1]["chg"])
    live = [(k, r) for k, r in order if r["chg"] > 0]
    for k, r in live:
        c, t = k.split("|")
        u = unit(next(x for x in model.order if x.name == c).field(t))
        LOGGER.info("  %-28s %6s %7.3f | %9.4f %9.4f %7.3f | %9.4f %9.4f %9.3f",
                    k.replace("|", "."), u, 100 * r["chg"], r["identity"], r["model"], r["skill"],
                    r["identity_chg"], r["model_chg"], r["skill_chg"])
    n_dead = len(order) - len(live)
    won = sum(1 for _, r in live if r["skill_chg"] > 0)
    LOGGER.info("  (%d fields never change on these episodes and are omitted)", n_dead)
    LOGGER.info("  fields beating do-nothing ON CHANGED STEPS: %d / %d", won, len(live))
    return live


def wandb_metrics(rows, split):
    """Per-field series, not one pooled headline: Stage 0 measured a pooled average over these fields as
    36% composition, so a single number would mostly report which fields are constant."""
    live = {k: r for k, r in rows.items() if r["chg"] > 0}
    m = {f"{split}/fields_live": len(live),
         f"{split}/fields_beating_identity": sum(1 for r in live.values() if r["skill_chg"] > 0)}
    if live:
        m[f"{split}/won_frac"] = m[f"{split}/fields_beating_identity"] / len(live)
        m[f"{split}/median_skill_chg"] = float(np.median([r["skill_chg"] for r in live.values()]))
        m[f"{split}/median_skill"] = float(np.median([r["skill"] for r in live.values()]))
    for k, r in live.items():
        n = k.replace("|", ".")
        m[f"skill_{split}/{n}"], m[f"skill_chg_{split}/{n}"] = r["skill"], r["skill_chg"]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    ap.add_argument("--episodes", type=int, default=0,
                    help="TOTAL episodes sampled at random across all collections; 0 = all")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="training seed (init + batch order)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="", help="checkpoint path; default <run-dir>/model.pt. Never shared: "
                                              "a smoke run overwrote the 6-epoch paper-shape checkpoint once")
    ap.add_argument("--json", default="")
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--name", default="", help="run name; default oocdm-stage1-<timestamp>")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "zelda-rl-oocdm"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-group", default="oocdm")
    ap.add_argument("--hand-graph", action="store_true",
                    help="P0 gate (docs section 16): mask Link.pos' to the parents the ridge probe says it "
                         "needs, leave every other field at the full graph. One variable.")
    args = ap.parse_args()

    name = args.name or f"oocdm-stage1-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(args.run_dir or f"runs/{name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir, vars(args), use_wandb=args.wandb, project=args.wandb_project,
                    entity=args.wandb_entity, run_name=name, resume=False,
                    tags=["oocdm", "stage1", "graph-G1"], group=args.wandb_group)
    if log.run is not None:
        import wandb
        wandb.define_metric("global_step")
        for pre in ("train", "tune", "test", "loss", "skill_tune", "skill_chg_tune", "control", "graph"):
            wandb.define_metric(f"{pre}/*", step_metric="global_step")

    LOGGER.info("run dir: %s", run_dir)
    LOGGER.info("schema partition: %s", assert_partition())
    t0 = time.time()
    data = load(args.data, args.episodes, args.sample_seed)
    LOGGER.info("episodes: train %d, tune %d, test %d (%.0fs)", len(data["train"]), len(data["tune"]),
                len(data["test"]), time.time() - t0)
    assert_roundtrip(*data["train"][0])

    # 160 rather than 64: codec scales for rare fields need episodes that actually contain a change.
    gs = [decompose(st, a) for st, a in data["train"][:160]]
    # TRAIN ONLY (no test leakage into a scale), and the delta scale is measured over the same steps the
    # value head is trained on -- changed steps when the gate is on, all steps when it is off.
    # The delta scale must match where the VALUE head trains, which since section 13 is all steps by
    # default -- not where the gate is. Tying this to --gate was wrong.
    codecs = fit_codecs(gs)
    packs = {k: pack(v) for k, v in data.items() if v}
    cap = {c: max(p.capacity[c] for p in packs.values()) for c in packs["train"].capacity}
    packs = {k: p.pad_to(cap) for k, p in packs.items()}      # one capacity for every split
    LOGGER.info("transitions: %s | capacity %s", ", ".join(f"{k} {len(p)}" for k, p in packs.items()), cap)

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = OOCDM(codecs, cap).to(dev)
    if args.hand_graph:
        # Ridge on the model's own input: vel alone +0.219, all 38 Link fields -0.066 (section 16). If
        # masking is the mechanism the paper says it is, restricting THIS ONE target to those parents should
        # recover most of that. Everything else stays at G1, so exactly one thing differs from the baseline.
        g = OOCGraph(model.order, model.field_names, model.target_names, full=True)
        g.set_local("Link", "pos", HAND_PARENTS)
        g.set_global("Link", "pos", None)
        model.set_graph(g)
        LOGGER.info("P0 hand graph: Link.pos' <- local %s, global {} | %d of %d edges active",
                    HAND_PARENTS, g.n_edges(), g.n_out * g.n_in + sum(m.size for m in g.local.values()))
    n_par, n_head = sum(x.numel() for x in model.parameters()), sum(len(v) for v in model.target_names.values())
    LOGGER.info("params: %.2fM, %d field heads", n_par / 1e6, n_head)

    # Asserted on EVERY split, not just train: the first version of this passed on train and was wrong on
    # test, because codecs are fitted on train and an unseen categorical value had no identity slot.
    for k, pk in packs.items():
        assert_identity_init(model, make_batch(pk, codecs, model, pk.pairs[:256], dev))
    b0 = make_batch(packs["train"], codecs, model, packs["train"].pairs[:128], dev)
    LOGGER.info("control -- identity init: untrained model is EXACTLY do-nothing on %s  OK", ", ".join(packs))
    perm_msg = check_permutation(model, b0)
    LOGGER.info("control -- permutation:  %s", perm_msg)

    vb = velocity_baseline(packs["test"])
    if vb:
        LOGGER.info("control -- trivial baseline: Link.pos = pos + %.2f*vel scores val|chg %+.3f (XZ, %d "
                    "changed steps). AGENTS.md section 1 free-signal control; a value head that does not "
                    "beat this has earned nothing.", vb["k"], vb["val_chg"], vb["n"])
    base = evaluate(model, packs["test"], codecs, dev, cap=20000)
    worst = max(((abs(r["skill"]), k) for k, r in base.items()), default=(0.0, "-"))
    LOGGER.info("control -- untrained test skill: worst |skill| %.2e on %s%s", worst[0], worst[1],
                "" if worst[0] == 0 else "  (nonzero only through the float round-trip of the angle codec)")
    gr = model.graph
    n_cand = gr.n_out * gr.n_in + sum(m.size for m in gr.local.values())
    log.update({"control/untrained_worst_abs_skill": worst[0], "control/permutation_ok": float("OK" in perm_msg),
                "control/params_m": n_par / 1e6, "control/field_heads": n_head,
                "graph/edges_active": gr.n_edges(), "graph/edges_candidate": n_cand,
                "graph/edges_global": int(gr.global_matrix.sum()),
                "graph/edges_local": int(sum(m.sum() for m in gr.local.values())),
                "graph/frac_active": gr.n_edges() / n_cand,
                "control/velocity_baseline_val_chg": vb.get("val_chg", float("nan")) if vb else float("nan")},
               0)
    if log.run is not None:
        # the graph IS the experiment: two runs differing only in it are otherwise indistinguishable
        log.run.summary.update({"graph/kind": "hand" if args.hand_graph else "G1",
                                "graph/hand_parents": ",".join(HAND_PARENTS) if args.hand_graph else ""})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
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
        step = (ep + 1) * len(tr.pairs)
        m = wandb_metrics(rows, "tune")
        m["train/loss"] = tot / max(nb, 1)
        m["train/epoch"] = ep + 1
        m["train/sec_per_epoch"] = time.time() - t1
        m.update({f"loss/{k.replace('|', '.')}": float(v.detach()) for k, v in L.items()})
        log.update(m, step)
        LOGGER.info("epoch %d/%d  loss %8.3f  tune beating do-nothing on changed steps: %d/%d  (%.0fs)",
                    ep + 1, args.epochs, m["train/loss"], m["tune/fields_beating_identity"],
                    m["tune/fields_live"], m["train/sec_per_epoch"])

    rows = evaluate(model, packs["test"], codecs, dev)
    live = report(rows, model, f"TEST slots {sorted(TEST_SLOTS)}, {args.epochs} epochs")
    final = wandb_metrics(rows, "test")
    log.update(final, args.epochs * len(tr.pairs))
    log.summary({k: v for k, v in final.items() if v is not None})
    out = Path(args.out) if args.out else (run_dir / "model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Codecs travel WITH the weights. They are fitted statistics -- categorical cardinalities are measured
    # from data -- so refitting them on a different episode sample silently changes head widths and the
    # checkpoint will not load (action_fn_rel came back 27 wide instead of 28).
    torch.save({"model": model.state_dict(), "capacity": cap, "args": vars(args), "codecs": codecs}, out)
    LOGGER.info("wrote %s (%d fields live on test)", out, len(live))
    log.artifact(out)
    (run_dir / "fields.json").write_text(json.dumps(rows, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    LOGGER.info("wrote %s", run_dir / "fields.json")
    log.close()


if __name__ == "__main__":
    main()
