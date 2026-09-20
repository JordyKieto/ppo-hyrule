"""P2 component 9: the paper's discover <-> train loop, run to graph stability.

    .venv/bin/python scripts/train_oocdm_loop.py --init runs/p1-paper-obj/model.pt --iters 3
    .venv/bin/python scripts/train_oocdm_loop.py --resume runs/p2-loop            # picks up where it stopped

WHAT THE PAPER DOES (../oocdm/scripts_/train.py:404, train_ooc.py)

    while not end_iter():
        collect samples
        update()        ->  RE-DISCOVER the graph from the current model, and REPLACE it
        fit(n_batch)    ->  train under -(ll_graph + ll_full + ll_random)/3 with the NEW graph
    end_iter():  n_iter reached, OR the graph unchanged for n_iter_stable iterations

Discovery runs BEFORE training each iteration and the graph is replaced, never accumulated. The loop
bootstraps: at iteration 0 the model is barely trained so every CMI is ~0 and the graph comes out nearly
empty, but `ll_full` and `ll_random` train it regardless; next iteration discovery finds real edges, so the
`ll_graph` term starts carrying information. Graph and model improve each other.

THE QUESTION THIS ANSWERS: does the graph CONVERGE? If it settles after a few iterations we have a causal
model. If it churns, the method does not identify a stable structure on this data -- and that is the result,
not something to tune away.

SETTINGS, and why
  epsilon 0.05        tuned: peak Jaccard 0.656 at 10,240 transitions. The paper tunes per environment over
                      0.03-0.3, so this is inside their range (docs/CAUSAL_WORLD_MODEL.md 17.4c)
  CMI 10,240          the paper uses its whole buffer, 10k-50k. At 1k the graph was sampling noise
  CMI batch 2048      no_grad, ~7x cheaper per transition than 128
  p_edge 0.9          the paper's default
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oocdm import (OOCDM, FullMask, GraphMask, OOCGraph, RandomMask, discover_graph,  # noqa: E402
                            field_losses, make_batch, pack, velocity_baseline)
from zelda_rl.oomdp import err, prep, unit  # noqa: E402
from zelda_rl.runlog import LOGGER, RunLogger  # noqa: E402

TRAIN_SLOTS, TUNE_SLOTS, TEST_SLOTS = {1, 2, 3, 5, 6, 10}, {4, 8}, {7, 9}
WATCH = ["Link|pos", "Link|vel", "Link|speed", "En_Dekubaba|pos", "En_Dekubaba|hp", "Camera|eye"]


def hms(sec: float) -> str:
    return str(timedelta(seconds=int(max(sec, 0))))


def load_split(dirs, limit, sample_seed):
    cat = []
    for d in dirs:
        d = Path(d)
        cat += [(d, r) for r in (json.loads(l) for l in
                                 (d / "manifest.jsonl").read_text().splitlines() if l.strip())]
    if limit:
        keep = sorted(np.random.default_rng(sample_seed).permutation(len(cat))[:limit])
        cat = [cat[i] for i in keep]
    out = {"train": [], "tune": [], "test": []}
    for d, r in cat:
        z = np.load(d / r["file"])
        st = z["state"]
        e = (st, np.r_[z["action"], z["action"][-1]][: len(st)])
        out["test" if r["slot"] in TEST_SLOTS else
            "tune" if r["slot"] in TUNE_SLOTS else "train"].append(e)
    return out


@torch.no_grad()
def quick_eval(model, p, codecs, device, cap=12000, batch_size=512):
    """Per-field skill on changed steps only -- the column copying cannot win."""
    model.eval()
    idx = p.pairs if len(p.pairs) <= cap else p.pairs[:: max(1, len(p.pairs) // cap)]
    acc = {}
    for s in range(0, len(idx), batch_size):
        b = make_batch(p, codecs, model, idx[s: s + batch_size], device)
        from zelda_rl.oocdm import predict_raw
        pred = predict_raw(model, model(b, GraphMask(model.graph)), b)
        for c in model.order:
            for t in model.target_names[c.name]:
                key, f = f"{c.name}|{t}", c.field(t)
                v = b["valid"][c.name][t].cpu().numpy()
                if not v.any():
                    continue
                cur = b["raw"][c.name][t].cpu().numpy()[v]
                nxt = p.attrs[c.name][t][b["t"] + 1][v]
                ei = err(f, prep(f, nxt), prep(f, cur))
                em = err(f, prep(f, nxt), prep(f, pred[key][v]))
                m = ei > 0
                d = acc.setdefault(key, [0.0, 0.0, 0])
                d[0] += float(ei[m].sum()); d[1] += float(em[m].sum()); d[2] += int(m.sum())
    model.train()
    return {k: (1 - v[1] / v[0]) for k, v in acc.items() if v[2] > 0 and v[0] > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="runs/p1-paper-obj/model.pt", help="checkpoint to start from")
    ap.add_argument("--resume", default="", help="run dir to resume; ignores --init")
    ap.add_argument("--run-dir", default="runs/p2-loop")
    ap.add_argument("--data", nargs="+", default=[f"datasets/transitions/v3-{k}/20260919-084052"
                                                  for k in ("main", "approach", "moves", "scenarios")])
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs-per-iter", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--cmi-batches", type=int, default=5)
    ap.add_argument("--cmi-batch-size", type=int, default=2048)
    ap.add_argument("--p-edge", type=float, default=0.9)
    ap.add_argument("--n-stable", type=int, default=2, help="stop after the graph is unchanged this often")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", dest="wandb", action="store_false")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "zelda-rl-oocdm"))
    a = ap.parse_args()

    run_dir = Path(a.resume or a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "loop_state.pt"
    resuming = bool(a.resume) and state_path.exists()
    name = run_dir.name
    log = RunLogger(run_dir, vars(a), use_wandb=a.wandb, project=a.wandb_project, entity=None,
                    run_name=name, resume=resuming, tags=["oocdm", "p2", "discover-train-loop"],
                    group="oocdm-p2")
    if log.run is not None:
        import wandb
        wandb.define_metric("iter")
        for pre in ("graph", "loss", "tune", "time", "edge"):
            wandb.define_metric(f"{pre}/*", step_metric="iter")

    LOGGER.info("run dir %s | %s", run_dir, "RESUMING" if resuming else "fresh")
    data = load_split(a.data, a.episodes, 0)
    packs = {k: pack(v) for k, v in data.items() if v}
    cap = {c: max(p.capacity[c] for p in packs.values()) for c in packs["train"].capacity}
    packs = {k: p.pad_to(cap) for k, p in packs.items()}
    LOGGER.info("episodes train/tune/test %d/%d/%d | transitions %s", len(data["train"]), len(data["tune"]),
                len(data["test"]), {k: len(p) for k, p in packs.items()})

    src = torch.load(state_path if resuming else a.init, map_location="cpu", weights_only=False)
    codecs = src["codecs"]
    dev = torch.device(a.device)
    model = OOCDM(codecs, cap).to(dev)
    model.load_state_dict(src["model"])
    start_iter, hist = 0, []
    if resuming:
        model.set_graph(OOCGraph(model.order, model.field_names, model.target_names,
                                 full=False).load_state_dict(src["graph"]))
        start_iter, hist = src["iter"] + 1, src.get("hist", [])
        LOGGER.info("resumed at iteration %d with %d edges", start_iter, model.graph.n_edges())
    else:
        model.set_graph(OOCGraph(model.order, model.field_names, model.target_names, full=False))
        LOGGER.info("starting from %s with an EMPTY graph (the paper's init)", a.init)

    vb = velocity_baseline(packs["test"])
    LOGGER.info("control -- trivial baseline Link.pos = pos + %.2f*vel scores val|chg %+.3f", vb["k"],
                vb["val_chg"])
    LOGGER.info("settings: eps %.3f | CMI %d transitions @ batch %d | p_edge %.2f | %d epochs/iter",
                a.eps, a.cmi_batches * a.cmi_batch_size, a.cmi_batch_size, a.p_edge, a.epochs_per_iter)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    if resuming and "optim" in src:
        opt.load_state_dict(src["optim"])
    rng = np.random.default_rng(0)
    tr = packs["train"]
    prev_edges = model.graph.edge_set()
    n_unchanged, times, t_start = 0, [], time.time()

    for it in range(start_iter, a.iters):
        t_it = time.time()
        LOGGER.info("=== iteration %d/%d ===============================================", it + 1, a.iters)

        # ---- DISCOVER (before training, as the paper does) ------------------------------------------
        t0 = time.time()
        g, cmis = discover_graph(model, packs["train"], codecs, dev, a.eps,
                                 n_batches=a.cmi_batches, batch_size=a.cmi_batch_size, seed=it)
        t_disc = time.time() - t0
        edges = g.edge_set()
        added, removed = edges - prev_edges, prev_edges - edges
        jac = len(edges & prev_edges) / max(len(edges | prev_edges), 1)
        model.set_graph(g)
        LOGGER.info("  discover %s | %d edges (+%d/-%d, jaccard vs prev %.3f)", hms(t_disc), len(edges),
                    len(added), len(removed), jac)
        for e in sorted(added, key=lambda x: -cmis.get(x, 0))[:5]:
            LOGGER.info("    + %+.4f  %s", cmis.get(e, 0), e)
        for e in sorted(removed)[:3]:
            LOGGER.info("    - %s", e)

        # ---- TRAIN under the new graph ---------------------------------------------------------------
        t0 = time.time()
        tot = {"graph": 0.0, "full": 0.0, "random": 0.0}
        nb = 0
        for _ in range(a.epochs_per_iter):
            perm = rng.permutation(tr.pairs)
            for s in range(0, len(perm) - a.batch, a.batch):
                b = make_batch(tr, codecs, model, perm[s: s + a.batch], dev)
                Lg = sum(field_losses(model, model(b, GraphMask(model.graph)), b).values())
                Lf = sum(field_losses(model, model(b, FullMask(model.graph)), b).values())
                Lr = sum(field_losses(model, model(b, RandomMask(model.graph, a.p_edge)), b).values())
                loss = (Lg + Lf + Lr) / 3.0
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                tot["graph"] += float(Lg.detach()); tot["full"] += float(Lf.detach())
                tot["random"] += float(Lr.detach()); nb += 1
        t_train = time.time() - t0
        ll = {k: v / max(nb, 1) for k, v in tot.items()}
        sk = quick_eval(model, packs["tune"], codecs, dev)
        won = sum(1 for v in sk.values() if v > 0)
        LOGGER.info("  train    %s | graph %.2f full %.2f random %.2f | tune beating do-nothing %d/%d",
                    hms(t_train), ll["graph"], ll["full"], ll["random"], won, len(sk))
        LOGGER.info("    watch: %s", "  ".join(f"{w.replace('|', '.')} {sk.get(w, float('nan')):+.3f}"
                                               for w in WATCH))

        times.append(time.time() - t_it)
        eta = np.mean(times) * (a.iters - it - 1)
        LOGGER.info("  iter %s | elapsed %s | ETA %s (finish ~%s)", hms(times[-1]), hms(time.time() - t_start),
                    hms(eta), (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M"))

        n_unchanged = n_unchanged + 1 if not added and not removed else 0
        hist.append({"iter": it, "edges": len(edges), "added": len(added), "removed": len(removed),
                     "jaccard_prev": jac, **{f"ll_{k}": v for k, v in ll.items()},
                     "tune_won": won, "tune_live": len(sk),
                     **{f"skill_{w.replace('|', '.')}": sk.get(w) for w in WATCH}})
        log.update({"iter": it, "graph/edges": len(edges), "graph/added": len(added),
                    "graph/removed": len(removed), "graph/jaccard_prev": jac,
                    "loss/ll_graph": ll["graph"], "loss/ll_full": ll["full"], "loss/ll_random": ll["random"],
                    "loss/ll_spread": max(ll.values()) - min(ll.values()),
                    "tune/fields_beating_identity": won, "tune/fields_live": len(sk),
                    "time/discover_s": t_disc, "time/train_s": t_train, "time/iter_s": times[-1],
                    **{f"tune/skill_{w.replace('|', '.')}": sk[w] for w in WATCH if w in sk}}, it)
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "codecs": codecs,
                    "capacity": cap, "graph": model.graph.state_dict(), "iter": it, "hist": hist,
                    "args": vars(a)}, state_path)
        (run_dir / "graph.json").write_text(json.dumps(sorted(edges), indent=1))
        (run_dir / "history.json").write_text(json.dumps(hist, indent=1))
        prev_edges = edges
        if n_unchanged >= a.n_stable:
            LOGGER.info("graph unchanged for %d iterations -- STABLE, stopping", n_unchanged)
            break

    LOGGER.info("done in %s | final graph %d edges -> %s", hms(time.time() - t_start),
                model.graph.n_edges(), run_dir / "graph.json")
    log.summary({"graph/final_edges": model.graph.n_edges(), "iters_run": len(hist)})
    log.close()


if __name__ == "__main__":
    main()
