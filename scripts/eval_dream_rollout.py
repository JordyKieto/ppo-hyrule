"""Does the dream reproduce a REAL trajectory? Same start, same actions, compared at every horizon.

    .venv/bin/python scripts/eval_dream_rollout.py checkpoints/dynamics/joint_v15_exitev.pt

THE barometer. Every other measurement in this project is either single-step (decode accuracy on real
latents) or self-referential (does the dream avoid spurious terminals, does a held action close distance),
and both can look healthy while the model reproduces nothing. This replays a real episode's actions from its
real start state and asks whether the dream goes where the game went.

The distinction that makes it necessary: `scripts/eval_rollout.py` rolls the raw latent and decodes absolute
fields, which on v15 diverges to a link_pos MAE of 10,030 world units by h=1800 against an identity baseline
of 26.3. The dream never reads those absolute fields -- it integrates displacement and carries HP through
`apply_transition` -- so the raw number understates the dream while the dream's own audits overstate it. This
evaluates the pipeline the policy actually inhabits.

What it reports, per horizon, always against identity (assume nothing changes):

  WALKING      distance travelled, dream vs real, plus a dense early curve -- because a break at step 1 and a
               break at step 200 call for completely different fixes
  GEOMETRY     absolute position error, and distance to the nearest Baba, which is what `approach` pays on
  EVENTS       damage, kills, Link HP lost, blocks, dodges -- cumulative, dream vs real
  STATE        live Baba count and Link HP, dream vs truth vs identity

Averaged over several action-noise seeds, since damage and despawn are sampled.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zelda_rl.bc as bc  # noqa: E402
from zelda_rl.bc import BCNet, Norm, StateDecoder, UnifiedDecoder, featurise, wants_v2  # noqa: E402
from zelda_rl.combat import CombatConfig, CombatTracker, DEKU_BABA  # noqa: E402
from zelda_rl.dynamics import TransitionHead, load_predictor  # noqa: E402
from zelda_rl.dream_state import (DreamTracker, apply_transition, decoded_states_from_raw,  # noqa: E402
                                  decoded_to_states)

# dense early, because "13x too slow by step 5" and "frozen after step 200" are different defects
HORIZONS = (1, 2, 3, 5, 10, 20, 40, 80, 150, 300, 600, 1200, 1800)


MAX_STEP_UNITS = 28.0     # Link's top speed: ~4.7 world units/game frame x 6 frames per 10 Hz env step


def live_babas(s):
    return sum(1 for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0)


def nearest_baba(s):
    d = [math.dist(e.pos, s.pos) for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0]
    return min(d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--min-len", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--move-gate", choices=["sample", "expect"], default="sample",
                    help="'sample': fire the move as a Bernoulli draw. 'expect': scale the step by P(moved), "
                         "the old behaviour, which understates a discrete event by construction.")
    ap.add_argument("--seeds", type=int, default=3, help="action-noise seeds: damage and despawn are sampled")
    ap.add_argument("--out", default="results/dream_rollout.json")
    a = ap.parse_args()
    torch.set_num_threads(6)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = (UnifiedDecoder(latent=LAT) if ck.get("unified_decoder") else StateDecoder(latent=LAT))
    dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = load_predictor(ck, LAT, NA)
    thead = None
    if ck.get("thead") is not None:
        thead = TransitionHead(LAT, NA); thead.load_state_dict(ck["thead"]); thead.eval()
    norm = Norm.load(ck["norm"]); scale = ck["decode_scale"]
    EC = ck.get("event_calib") or {}
    ET = ck.get("exit_thresh", 6.38)
    held = set(ck["split"]["val_slots"])

    dirs = a.data or [sorted(glob.glob("datasets/transitions/v3-main/*/"))[-1]]
    eps, dropped = [], []
    for d in dirs:
        for m in map(json.loads, open(Path(d) / "manifest.jsonl")):
            if m["slot"] not in held or m["steps"] < a.min_len:
                continue
            z = np.load(Path(d) / m["file"])
            S_ = z["state"]
            # Drop episodes whose FIRST step is physically impossible. Link's maximum is ~4.7 world units per
            # game frame x 6 frames per env step = ~28 units, and some savestate loads have not settled at
            # index 0: one slot-7 episode teleports 590 units from (-4,0,597) to (-42.7,-9,8.5). A single such
            # episode dragged the mean first-step travel from a true median of 0.0 to 46.2 and produced a
            # bogus "the dream walks 100x too slowly" reading.
            d0 = math.hypot(S_[1, bc.LI["pos_x"]] - S_[0, bc.LI["pos_x"]],
                            S_[1, bc.LI["pos_z"]] - S_[0, bc.LI["pos_z"]])
            if d0 > MAX_STEP_UNITS:
                dropped.append((m["slot"], round(float(d0))))
                continue
            eps.append((S_, z["action"]))
            if len(eps) >= a.episodes:
                break
    if not eps:
        raise SystemExit(f"no held-out episodes of >= {a.min_len} steps in {dirs}")
    H = [h for h in HORIZONS if h <= min(len(x[1]) for x in eps)]
    print(f"{len(eps)} held-out episodes (slots {sorted(held)}) of >= {a.min_len} steps, "
          f"{a.seeds} seeds, horizons to {H[-1]}")
    if dropped:
        print(f"  dropped {len(dropped)} episode(s) with a physically impossible first step "
              f"(savestate not settled): {dropped}")
    print()

    KEYS = ("trav", "trav_true", "pos", "pos_id", "near", "near_true", "near_id",
            "dmg", "dmg_true", "kill", "kill_true", "hpl", "hpl_true", "blk", "blk_true",
            "nb", "nb_true", "hp", "hp_true", "hp_id")
    acc = {h: {k: [] for k in KEYS} for h in H}
    cfg = CombatConfig()

    for S, A in eps:
        T = len(A)
        truth = decoded_states_from_raw(S[:T + 1])
        # ground truth events, from the REAL tracker on the REAL trajectory
        rt = CombatTracker(cfg); rt.reset(truth[0], T)
        rev, rtrav = [], 0.0
        for t in range(1, T + 1):
            rt.step(truth[t - 1], truth[t])
            rtrav += math.hypot(truth[t].pos[0] - truth[t - 1].pos[0], truth[t].pos[2] - truth[t - 1].pos[2])
            rev.append((rt.ep["damage_dealt"], rt.ep["defeated"], rt.ep["hp_lost"],
                        rt.ep["blocks"] + rt.ep["dodges"], rtrav))
        f = norm.apply(featurise(S[:1], link_v2=wants_v2(ck["link_dim"])))
        for seed in range(a.seeds):
            dt = DreamTracker(cfg); dt.reset(truth[0], T)
            timers, rng = {}, np.random.default_rng(seed)
            with torch.no_grad():
                z = enc.encode(*(torch.from_numpy(f[k]) for k in ("link", "rows", "ids", "mask")))
                cur = decoded_to_states(dec(z), scale)
                p0, n0, hp0 = cur[0].pos, nearest_baba(cur[0]), cur[0].hp
                trav = 0.0
                for t in range(1, T + 1):
                    oh = torch.nn.functional.one_hot(torch.tensor([int(A[t - 1])]), NA).float()
                    q = thead(z, oh) if thead is not None else None
                    z = pred(torch.cat([z, oh], -1))
                    nxt = decoded_to_states(dec(z), scale)
                    if q is not None:
                        nxt = apply_transition(cur, nxt, q, calib=EC, rng=rng, exit_thresh=ET,
                                               move_gate=a.move_gate, _timers=timers)
                    dt.step(cur[0], nxt[0])
                    trav += math.hypot(nxt[0].pos[0] - cur[0].pos[0], nxt[0].pos[2] - cur[0].pos[2])
                    cur = nxt
                    if t in acc:
                        r, ts = acc[t], truth[t]
                        rd, rk, rh, rb, rtv = rev[t - 1]
                        r["trav"].append(trav);            r["trav_true"].append(rtv)
                        r["pos"].append(math.hypot(cur[0].pos[0] - ts.pos[0], cur[0].pos[2] - ts.pos[2]))
                        r["pos_id"].append(math.hypot(p0[0] - ts.pos[0], p0[2] - ts.pos[2]))
                        r["near"].append(nearest_baba(cur[0])); r["near_true"].append(nearest_baba(ts))
                        r["near_id"].append(n0)
                        r["dmg"].append(dt.ep["damage_dealt"]);  r["dmg_true"].append(rd)
                        r["kill"].append(dt.ep["defeated"]);     r["kill_true"].append(rk)
                        r["hpl"].append(dt.ep["hp_lost"]);       r["hpl_true"].append(rh)
                        r["blk"].append(dt.ep["blocks"] + dt.ep["dodges"]); r["blk_true"].append(rb)
                        r["nb"].append(live_babas(cur[0]));      r["nb_true"].append(live_babas(ts))
                        r["hp"].append(cur[0].hp); r["hp_true"].append(ts.hp); r["hp_id"].append(hp0)

    M = {h: {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc[h].items()} for h in H}
    MD = {h: {k: float(np.nanmedian(v)) if v else float("nan") for k, v in acc[h].items()} for h in H}

    print("WALKING -- distance travelled (world units). The single most important row.")
    print("  median as well as mean: real per-step travel has median 0.0 with a long tail, so a mean-only")
    print("  comparison is dominated by the few fast frames.")
    print(f"{'h':>5} {'dream mean':>11} {'real mean':>10} {'ratio':>7} | "
          f"{'dream med':>10} {'real med':>9} {'ratio':>7}   verdict")
    for h in H:
        m, md = M[h], MD[h]
        r1 = m["trav"] / m["trav_true"] if m["trav_true"] > 1e-6 else float("nan")
        r2 = md["trav"] / md["trav_true"] if md["trav_true"] > 1e-6 else float("nan")
        v = ("too slow" if r1 < 0.7 else "too fast" if r1 > 1.4 else "ok")
        print(f"{h:>5} {m['trav']:>11.1f} {m['trav_true']:>10.1f} {r1:>7.2f} | "
              f"{md['trav']:>10.1f} {md['trav_true']:>9.1f} {r2:>7.2f}   {v}")

    print("\nGEOMETRY -- position error, and distance to the nearest Baba (what `approach` pays on)")
    print(f"{'h':>5} {'pos err':>9} {'identity':>9} | {'near d':>8} {'near real':>10} {'near id':>8}")
    for h in H:
        m = M[h]
        print(f"{h:>5} {m['pos']:>9.1f} {m['pos_id']:>9.1f} | {m['near']:>8.0f} {m['near_true']:>10.0f} "
              f"{m['near_id']:>8.0f}")

    print("\nEVENTS -- cumulative, dream vs real")
    print(f"{'h':>5} {'damage':>13} {'kills':>13} {'hearts lost':>15} {'blocks+dodges':>16}")
    for h in H:
        m = M[h]
        print(f"{h:>5} {m['dmg']:>5.2f}/{m['dmg_true']:<7.2f} {m['kill']:>5.2f}/{m['kill_true']:<7.2f} "
              f"{m['hpl']/16:>6.2f}/{m['hpl_true']/16:<8.2f} {m['blk']:>6.2f}/{m['blk_true']:<9.2f}")

    print("\nSTATE -- live Babas and Link HP: dream / truth / identity")
    print(f"{'h':>5} {'babas':>20} {'hearts':>22}")
    for h in H:
        m = M[h]
        print(f"{h:>5} {m['nb']:>6.2f}/{m['nb_true']:>5.2f}/{m['nb_true']*0+3:>5.2f} "
              f"{m['hp']/16:>9.2f}/{m['hp_true']/16:>5.2f}/{m['hp_id']/16:>5.2f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "episodes": len(eps), "seeds": a.seeds,
                                       "rows": {str(h): M[h] for h in H}}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
