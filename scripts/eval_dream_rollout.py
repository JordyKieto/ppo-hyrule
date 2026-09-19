"""Does the dream reproduce a REAL trajectory? Same start, same actions, compared at every horizon.

    .venv/bin/python scripts/eval_dream_rollout.py checkpoints/dynamics/joint_v15_exitev.pt

This is the success metric the project lacked. Everything measured before it was either single-step (decode
accuracy on real latents) or self-referential (does the dream avoid spurious terminals, does a held action
close distance). None of it asked the only question that matters for planning inside a model: **replay a real
episode's actions from its real start state and see whether the dream goes where the game went.**

The distinction that makes this necessary: `scripts/eval_rollout.py` rolls the raw latent and decodes absolute
fields, and on v15 that diverges to a link_pos MAE of 10,030 world units by h=1800 against an identity
baseline of 26.3. The dream does not read those absolute fields -- it integrates displacement and carries HP
through `apply_transition`. So the raw number understates the dream and the dream's own audits overstate it.
This evaluates the pipeline the policy actually inhabits.

Reported per horizon, always against the identity control (assume nothing changes):

  position       where Link is, and how far he has travelled, against the real trajectory
  key events     cumulative kills / damage dealt / Link HP lost / room exit, dream vs real
  decode         live Baba count, Link HP, Baba behaviour state

A model that cannot reproduce walking to a location cannot be planned in, whatever its single-step numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zelda_rl.bc as bc  # noqa: E402
from zelda_rl.bc import BCNet, Norm, StateDecoder, featurise, wants_v2  # noqa: E402
from zelda_rl.combat import CombatConfig, CombatTracker, DEKU_BABA  # noqa: E402
from zelda_rl.dynamics import TransitionHead, load_predictor  # noqa: E402
from zelda_rl.dream_state import (DreamTracker, apply_transition, decoded_states_from_raw,  # noqa: E402
                                  decoded_to_states)

HORIZONS = (1, 5, 10, 25, 50, 100, 200, 400, 800, 1200, 1800)


def live_babas(s):
    return sum(1 for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", default=None)
    ap.add_argument("--min-len", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", default="results/dream_rollout.json")
    a = ap.parse_args()
    torch.set_num_threads(6)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = StateDecoder(latent=LAT); dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = load_predictor(ck, LAT, NA)
    thead = None
    if ck.get("thead") is not None:
        thead = TransitionHead(LAT, NA); thead.load_state_dict(ck["thead"]); thead.eval()
    norm = Norm.load(ck["norm"]); scale = ck["decode_scale"]
    EC = ck.get("event_calib") or {}
    ET = ck.get("exit_thresh", 6.38)
    held = set(ck["split"]["val_slots"])

    d = a.data or sorted(glob.glob("datasets/transitions/v3-main/*/"))[-1]
    eps = []
    for m in map(json.loads, open(Path(d) / "manifest.jsonl")):
        if m["slot"] not in held or m["steps"] < a.min_len:
            continue
        z = np.load(Path(d) / m["file"])
        eps.append((z["state"], z["action"]))
        if len(eps) >= a.episodes:
            break
    if not eps:
        raise SystemExit(f"no held-out episodes of >= {a.min_len} steps in {d}")
    H = [h for h in HORIZONS if h <= min(len(a_) for _, a_ in eps)]
    print(f"{len(eps)} held-out episodes (slots {sorted(held)}) of >= {a.min_len} steps; horizons to {H[-1]}\n")

    rows = {h: {k: [] for k in ("pos", "pos_id", "trav", "trav_true", "nb", "nb_id", "hp", "hp_id",
                                "kill", "kill_true", "dmg", "dmg_true", "hpl", "hpl_true",
                                "exit", "exit_true")} for h in H}
    cfg = CombatConfig()
    for S, A in eps:
        T = len(A)
        true_states = decoded_states_from_raw(S[:T + 1])
        f = norm.apply(featurise(S[:1], link_v2=wants_v2(ck["link_dim"])))
        # real tracker on the real trajectory, for the ground-truth event counts
        rt = CombatTracker(cfg); rt.reset(true_states[0], T)
        real_ev = []
        for t in range(1, T + 1):
            rt.step(true_states[t - 1], true_states[t])
            real_ev.append((rt.ep["defeated"], rt.ep["damage_dealt"], rt.ep["hp_lost"],
                            int(rt.ep["end"] == "left_room")))
        # dream from the same start, fed the same actions
        dt = DreamTracker(cfg); dt.reset(true_states[0], T)
        timers = {}
        rng = np.random.default_rng(0)
        with torch.no_grad():
            z = enc.encode(*(torch.from_numpy(f[k]) for k in ("link", "rows", "ids", "mask")))
            cur = decoded_to_states(dec(z), scale)
            p0, tp0 = cur[0].pos, true_states[0].pos
            trav = trav_t = 0.0
            for t in range(1, T + 1):
                oh = torch.nn.functional.one_hot(torch.tensor([int(A[t - 1])]), NA).float()
                q = thead(z, oh) if thead is not None else None
                z = pred(torch.cat([z, oh], -1))
                nxt = decoded_to_states(dec(z), scale)
                if q is not None:
                    nxt = apply_transition(cur, nxt, q, calib=EC, rng=rng, exit_thresh=ET, _timers=timers)
                dt.step(cur[0], nxt[0])
                trav += float(np.hypot(nxt[0].pos[0] - cur[0].pos[0], nxt[0].pos[2] - cur[0].pos[2]))
                trav_t += float(np.hypot(true_states[t].pos[0] - true_states[t - 1].pos[0],
                                         true_states[t].pos[2] - true_states[t - 1].pos[2]))
                cur = nxt
                if t in rows:
                    r, tr_, ts = rows[t], true_states[t], true_states[t]
                    r["pos"].append(float(np.hypot(cur[0].pos[0] - ts.pos[0], cur[0].pos[2] - ts.pos[2])))
                    r["pos_id"].append(float(np.hypot(p0[0] - ts.pos[0], p0[2] - ts.pos[2])))
                    r["trav"].append(trav); r["trav_true"].append(trav_t)
                    r["nb"].append(live_babas(cur[0])); r["nb_id"].append(live_babas(true_states[0]))
                    r["hp"].append(cur[0].hp); r["hp_id"].append(true_states[0].hp)
                    dk, dd, dh, dx = (dt.ep["defeated"], dt.ep["damage_dealt"], dt.ep["hp_lost"],
                                      int(dt.ep["end"] == "left_room"))
                    rk, rd, rh, rx = real_ev[t - 1]
                    r["kill"].append(dk); r["kill_true"].append(rk)
                    r["dmg"].append(dd); r["dmg_true"].append(rd)
                    r["hpl"].append(dh); r["hpl_true"].append(rh)
                    r["exit"].append(dx); r["exit_true"].append(rx)
                    r["nb_id"][-1] = live_babas(true_states[0])

    def truth_nb(h):
        return np.mean([live_babas(decoded_states_from_raw(S[h:h + 1])[0]) for S, _ in eps])

    print("POSITION -- where Link actually is, against the real trajectory (world units)")
    print(f"{'h':>5} {'dream err':>11} {'identity err':>13} {'travelled dream':>16} {'travelled real':>15}")
    for h in H:
        r = rows[h]
        print(f"{h:>5} {np.mean(r['pos']):>11.1f} {np.mean(r['pos_id']):>13.1f} "
              f"{np.mean(r['trav']):>16.0f} {np.mean(r['trav_true']):>15.0f}")
    print("\nKEY EVENTS -- cumulative, dream vs real")
    print(f"{'h':>5} {'kills d/r':>12} {'damage d/r':>13} {'hearts lost d/r':>17} {'exited d/r':>12}")
    for h in H:
        r = rows[h]
        print(f"{h:>5} {np.mean(r['kill']):>5.2f}/{np.mean(r['kill_true']):<6.2f} "
              f"{np.mean(r['dmg']):>5.2f}/{np.mean(r['dmg_true']):<7.2f} "
              f"{np.mean(r['hpl'])/16:>7.2f}/{np.mean(r['hpl_true'])/16:<9.2f} "
              f"{np.mean(r['exit']):>4.2f}/{np.mean(r['exit_true']):<7.2f}")
    print("\nSTATE -- live Baba count and Link HP, dream vs truth vs identity")
    print(f"{'h':>5} {'babas d/true':>14} {'identity':>9} | {'hp d/true':>13} {'identity':>9}")
    for h in H:
        r = rows[h]
        tn = truth_nb(h)
        print(f"{h:>5} {np.mean(r['nb']):>6.2f}/{tn:<7.2f} {np.mean(r['nb_id']):>9.2f} | "
              f"{np.mean(r['hp'])/16:>6.2f}/{'?':<6} {np.mean(r['hp_id'])/16:>9.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"model": a.model, "episodes": len(eps),
         "rows": {str(h): {k: float(np.mean(v)) for k, v in rows[h].items()} for h in H}}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
