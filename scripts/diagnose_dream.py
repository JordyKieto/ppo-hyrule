"""Why does the dream hand out kills? Roll it forward with random actions and watch the decoded state.

    .venv/bin/python scripts/diagnose_dream.py --model checkpoints/dynamics/joint_v6_state2.pt

A random policy scored return 50.25 with 100% room-clear in 17 steps inside the dream, against a real-data
control of 25.28. A random policy cannot clear a Deku Baba fight in 1.7 seconds, so the kills are the world
model's, not the policy's.

`CombatTracker` books a kill when a tracked enemy's HP reaches 0 and it despawns. In the dream both of those
come from decoder heads (`row_hp`, `row_present`, `baba_count`) read off a PREDICTED latent. If the predictor
drifts toward the dataset mean -- where most Babas are dead, because episodes end after they die -- then
every dream episode ends in a clear no matter what the policy does.

This prints, per step of an autoregressive rollout from each real start state, the decoded quantities the
kill condition depends on, so the drift shows up as a step number.
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
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class, wants_v2)
from zelda_rl.combat import BABA_STATES, CombatConfig, DEKU_BABA  # noqa: E402
from zelda_rl.dream_state import (DreamTracker, apply_transition, decoded_states_from_raw,  # noqa: E402
                                  decoded_to_states)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/dynamics/joint_v6_state2.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--policy", default=None,
                    help="roll THIS trained dream policy and report how far Link's decoded position "
                         "actually travels. In the emulator the same policy walks ~500 world units and "
                         "leaves the room in 21-91 steps; if it barely moves in the dream, the dream cannot "
                         "show it the -20 it is about to earn.")
    ap.add_argument("--const-sweep", action="store_true",
                    help="hold each of the 63 actions constant and report kills. A freshly initialised PPO "
                         "policy is NOT uniform -- it is biased toward a few actions -- so this is the "
                         "distribution that actually probes the model, and it finds the exploit directly.")
    a = ap.parse_args()
    torch.set_num_threads(4)

    d = a.data or sorted(glob.glob("datasets/transitions/deku-baba-v2/*/"))[-1]
    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = StateDecoder(latent=LAT); dec.load_state_dict(ck["decoder"]); dec.eval()
    # same construction and call convention as scripts/train_dream_ppo.py
    from zelda_rl.dynamics import TransitionHead, load_predictor
    pred = load_predictor(ck, LAT, NA)
    thead = None
    if ck.get("thead") is not None:
        thead = TransitionHead(LAT, NA)
        thead.load_state_dict(ck["thead"]); thead.eval()
        print("using the action-conditioned transition head for motion and damage")
    norm = Norm.load(ck["norm"])
    scale = ck["decode_scale"]
    MT = ck.get("moved_thresh", 0.0)
    MC = tuple(ck.get("moved_calib", (1.0, 0.0)))
    HT, LHT = ck.get("hit_thresh", 0.5), ck.get("link_hit_thresh", 0.5)

    # first frame of each episode = a real savestate start state
    S, EP = [], []
    for i, m in enumerate(map(json.loads, open(Path(d) / "manifest.jsonl"))):
        S.append(np.load(Path(d) / m["file"])["state"][0])
        EP.append(m["slot"])
    S = np.stack(S); slots = np.array(EP)
    first = np.array([np.where(slots == s)[0][0] for s in sorted(set(slots.tolist()))])
    S0 = S[first]
    print(f"{len(S0)} start states (slots {sorted(set(slots.tolist()))}) from {d}\n")

    f = norm.apply(featurise(S0, link_v2=wants_v2(ck["link_dim"])))
    with torch.no_grad():
        z = enc.encode(*(torch.from_numpy(f[k]) for k in ("link", "rows", "ids", "mask")))

    # ground truth at t=0, via the real reader
    true0 = decoded_states_from_raw(S0)
    cfg = CombatConfig()
    trks = []
    for s in true0:
        t = DreamTracker(cfg); t.reset(s, a.steps); trks.append(t)
    print("t=0 TRUTH      : " + "  ".join(
        f"s{sorted(set(slots.tolist()))[i]}:{sum(1 for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0)}b"
        for i, s in enumerate(true0)))

    if a.policy:
        from zelda_rl.dream_state import MEM_DIM, DreamPolicy, memory_features
        pk = torch.load(a.policy, weights_only=False, map_location="cpu")
        pc = pk["config"]
        pol = DreamPolicy(LAT + MEM_DIM, NA, hidden=pc["hidden"], gru=pc.get("gru", 256))
        pol.load_state_dict(pk["model"]); pol.eval()
        cap = pc.get("max_steps", 1800)
        n = len(z)
        zz = z.clone()
        trs = []
        for s0 in true0:
            t = DreamTracker(cfg); t.reset(s0, cap); trs.append(t)
        h = pol.init_hidden(n)
        with torch.no_grad():
            cur = decoded_to_states(dec(zz), scale)
        p0 = np.stack([list(s.pos) for s in cur])
        print(f"\nRolling {a.policy} in the dream from {n} start states, {a.steps} steps:\n")
        print(f"{'t':>4} {'|disp| from start':>18} {'nearest baba':>13} {'scene!=0':>9} "
              f"{'live babas':>11} {'return':>8}")
        with torch.no_grad():
            for t in range(a.steps):
                m = np.stack([memory_features(tr, max(0.0, 1.0 - tr.t / cap)) for tr in trs])
                obs = torch.cat([zz, torch.from_numpy(m)], -1)
                act, _, _, _, h = pol.act(obs, h, torch.full((n,), 1.0 if t == 0 else 0.0))
                zz = pred(torch.cat([zz, torch.nn.functional.one_hot(act, NA).float()], -1))
                nxt = decoded_to_states(dec(zz), scale, prev=cur, moved_calib=MC)
                for i, tr in enumerate(trs):
                    tr.step(cur[i], nxt[i])
                cur = nxt
                pos = np.stack([list(s.pos) for s in nxt])
                disp = np.linalg.norm(pos[:, [0, 2]] - p0[:, [0, 2]], axis=1)
                near = [min((float(np.linalg.norm(np.subtract(e.pos, s.pos))) for e in s.enemies.values()
                             if e.hp > 0), default=float("nan")) for s in nxt]
                live = [sum(1 for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0) for s in nxt]
                if t < 12 or (t + 1) % 20 == 0:
                    print(f"{t+1:>4} {disp.mean():>18.1f} {np.nanmean(near):>13.0f} "
                          f"{sum(s.scene != 0 for s in nxt):>9} {np.mean(live):>11.2f} "
                          f"{np.mean([tr.ep['return'] for tr in trs]):>8.2f}")
        print(f"\nFor reference, in the EMULATOR this policy moved Link far enough to leave the room in "
              f"21-91 steps.\nIf |disp| above stays small, the dream never puts Link near the door, so the "
              f"-20 exit\nterminal can never fire during training -- which is exactly what left% 0.0% in the "
              f"training log means.")
        return

    if a.const_sweep:
        print(f"\nHolding each action constant for {a.steps} steps from all {len(S0)} start states:\n")
        from zelda_rl.env import action_name
        rows = []
        for act_id in range(NA):
            zz = z.clone()
            trs = []
            for s0 in true0:
                t = DreamTracker(cfg); t.reset(s0, a.steps); trs.append(t)
            nearest = lambda st: min((float(np.linalg.norm(np.subtract(e.pos, st.pos)))
                                      for e in st.enemies.values() if e.hp > 0), default=float("nan"))
            with torch.no_grad():
                cur = decoded_to_states(dec(zz), scale)
                d0 = [nearest(st) for st in cur]
                for _ in range(a.steps):
                    oh = torch.nn.functional.one_hot(torch.full((len(zz),), act_id), NA).float()
                    zz = pred(torch.cat([zz, oh], -1))
                    nxt = decoded_to_states(dec(zz), scale, prev=cur, moved_calib=MC)
                    for i, t in enumerate(trs):
                        t.step(cur[i], nxt[i])
                    cur = nxt
                d_end = [nearest(st) for st in cur]
            k = np.mean([t.ep["defeated"] for t in trs])
            cl = np.mean([t.ep["end"] == "success" for t in trs])
            lf = np.mean([t.ep["end"] == "left_room" for t in trs])
            ret = np.mean([t.ep["return"] for t in trs])
            # Can the dream represent APPROACHING and ORIENTING, not just leaving? These are the shaping
            # terms the reward actually pays: `approach` pays w*(d0-d1)/100 while NOT locked on, `lockon`
            # pays once per acquisition. Room exit got scripted data and a verification sweep; these never did.
            dd = float(np.mean([d_end[i] - d0[i] for i in range(len(trs))
                                if np.isfinite(d0[i]) and np.isfinite(d_end[i])]))
            appr = float(np.mean([t.ep["r_parts"].get("approach", 0.0) for t in trs]))
            lock = float(np.mean([t.ep["lockons"] for t in trs]))
            lockr = float(np.mean([t.ep["r_parts"].get("lockon", 0.0)
                                   + t.ep["r_parts"].get("lock_hold", 0.0) for t in trs]))
            # COMBAT audit: can the dream represent dealing damage and taking it? Same question as
            # approach -- the reward pays enemy_damage / kill / link_damage / block / dodge, and none of
            # those can fire unless the predictor moves decoded HP.
            dmg = float(np.mean([t.ep["damage_dealt"] for t in trs]))
            hpl = float(np.mean([t.ep["hp_lost"] for t in trs]))
            rp = lambda key: float(np.mean([t.ep["r_parts"].get(key, 0.0) for t in trs]))
            rows.append((k, cl, lf, ret, act_id, dd, appr, lock, lockr,
                         dmg, hpl, rp("enemy_damage"), rp("kill"), rp("link_damage"),
                         rp("block") + rp("dodge")))
        hdr = (f"{'action':<14} {'d_end-d0':>9} {'r_appr':>7} {'lockons':>8} | {'dmg':>5} {'kills':>6} "
               f"{'hpLost':>7} | {'r_edmg':>7} {'r_kill':>7} {'r_ldmg':>7} {'r_blk':>6} | {'left%':>6} {'ret':>8}")
        line = lambda r: (f"{action_name(r[4], 'combat'):<14} {r[5]:>+9.1f} {r[6]:>7.2f} {r[7]:>8.2f} | "
                          f"{r[9]:>5.2f} {r[0]:>6.2f} {r[10] / 16:>7.2f} | {r[11]:>7.2f} {r[12]:>7.2f} "
                          f"{r[13]:>7.2f} {r[14]:>6.2f} | {r[2]:>6.0%} {r[3]:>8.2f}")
        print("\n--- ranked by distance CLOSED (approach) ---")
        print(hdr)
        rows.sort(key=lambda r: r[5])
        for r in rows[:8]:
            print(line(r))
        print("\n--- ranked by DAMAGE DEALT (combat) ---")
        print(hdr)
        for r in sorted(rows, key=lambda r: -r[9])[:8]:
            print(line(r))
        print("\n--- ranked by Link HP LOST (taking damage) ---")
        print(hdr)
        for r in sorted(rows, key=lambda r: -r[10])[:5]:
            print(line(r))
        print(f"\nAcross all {len(rows)} actions: max damage dealt "
              f"{max(r[9] for r in rows):.2f} (a Baba has 2 HP, 3 in the room = 6 available), "
              f"max kills {max(r[0] for r in rows):.2f}, max hearts lost {max(r[10] for r in rows)/16:.2f} "
              f"(Link has 3).")
        rows.sort(key=lambda r: r[5])
        best = min(rows, key=lambda r: r[5])
        print(f"\nBest distance closed by a single held action: {action_name(best[4], 'combat')} "
              f"{best[5]:+.1f} units over {a.steps} steps (r_approach {best[6]:.2f}).")
        print("In the emulator a held direction closes hundreds of units. If nothing here closes distance,")
        print("the dream cannot represent approaching a Baba and the `approach` term is unearnable.")
        mx = max(rows, key=lambda r: r[7])
        print(f"Most lock-ons by a held action: {action_name(mx[4], 'combat')} {mx[7]:.2f} "
              f"(cap is 6/episode). A Z-press that never acquires means `lockon` is unearnable too.")
        worst = rows[0]
        print(f"\nBest single held action: {action_name(worst[4], 'combat')} -> {worst[0]:.2f} kills, "
              f"{worst[1]:.0%} clear, return {worst[3]:.2f} in {a.steps} steps.")
        print("A real Deku Baba fight cannot be won by holding one button, so anything above ~0 here is\n"
              "the world model being exploitable by a constant policy.")
        return

    rng = np.random.default_rng(0)
    with torch.no_grad():
        p0 = dec(z)
        prev_states = decoded_to_states({k: v for k, v in p0.items()}, scale)
        print("t=0 DECODED    : " + "  ".join(
            f"{sum(1 for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0)}b" for s in prev_states))
        print(f"\n{'t':>3} {'baba_count(mean)':>17} {'live Babas/state':>18} {'min row_hp':>11} "
              f"{'rows present':>13} {'kills booked':>13} {'cleared':>8}")
        for t in range(a.steps):
            act = torch.from_numpy(rng.integers(0, NA, len(z)))
            z = pred(torch.cat([z, torch.nn.functional.one_hot(act, NA).float()], -1))
            p = dec(z)
            states = decoded_to_states({k: v for k, v in p.items()}, scale,
                                       prev=prev_states, moved_calib=MC)
            bc_ = p["baba_count"].argmax(-1).numpy()
            rhp = np.asarray(ROW_HP_VALUES)[p["row_hp"].argmax(-1).numpy()]
            pres = (p["row_present"].numpy() > 0)
            live = [sum(1 for e in s.enemies.values() if e.id == DEKU_BABA and e.hp > 0) for s in states]
            kills, clear = [], 0
            for i, (tr, s) in enumerate(zip(trks, states)):
                tr.step(prev_states[i], s)
                kills.append(tr.ep["defeated"])
                clear += tr.ep["end"] == "success"
            prev_states = states
            print(f"{t+1:>3} {bc_.mean():>17.2f} {np.mean(live):>18.2f} "
                  f"{float(np.min(np.where(pres, rhp, 9))):>11.2f} {pres.sum(1).mean():>13.2f} "
                  f"{np.mean(kills):>13.2f} {clear:>8}")
            if t > 25 and np.mean(kills) > 0.9:
                break
    print("\nIf `live Babas/state` falls and `kills booked` rises while the policy is RANDOM, the kills are\n"
          "the predictor drifting, not the policy fighting.")


if __name__ == "__main__":
    main()
