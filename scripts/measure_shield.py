"""Does the Gohma policy ever raise the shield? Measured directly, not through a Baba-scoped counter.

Survival is the blocker: `runs/gohma-r*/episodes.jsonl` shows a 64.5% death rate on engaged slots that is FLAT
across all 56 updates while aim keeps improving. The two logged defensive metrics cannot tell us why:

  * `shield_steps` (combat.py) only counts a shielded step when a live target is within `shield_radius` = 150
    units -- a Deku Baba lunge distance. Gohma starts 188-448 away and spends much of the fight on the ceiling,
    so this reads 0.0 whether the policy shields constantly or never.
  * `boss_blocks` (gohma.py) requires a whole attack cycle that was shielded AND took zero damage AND happened
    inside `g_block_range`. It counts defensive SUCCESSES, not attempts.

So "shield_steps is 0" is not evidence about behaviour. This script logs the two things that are:

  1. the action the POLICY CHOSE (button bucket over the 90-action v5/v7 space), and
  2. whether `SF1_SHIELD` was actually set in RAM afterwards,

which separates three different worlds: the policy never picks shield (exploration problem), it picks shield
and the bit never sets (the action does not work), or it shields and dies anyway (a real tactical gap).

    DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
      .venv/bin/python-dbg scripts/measure_shield.py checkpoints/inbox/gohma-r0_u00050.pt \
      --slots 3,4,5,7,9 --envs 10 --episodes 20
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# ONE THREAD. Dolphin needs ~2 threads per instance (emulation + the Null GPU thread) and this Mac has 4
# performance cores; torch defaults to taking all 4 for a forward pass, which starves the emulation thread past
# the 10 s Null-backend stall timeout and raises "emulation stalled (VI stuck)". 27 scripts in this repo
# already set this -- the three I added did not, and I spent three runs blaming the hardware. AGENTS.md 28.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass            # already initialised in this process; the intra-op setting is the one that matters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import SF1_DEAD, SF1_SHIELD, SF1_ZTARGET  # noqa: E402
from zelda_rl.env import COMBAT_ACTIONS_V5, COMBAT_BUTTON_NAMES_V5, DIR_NAMES, DIRECTIONS  # noqa: E402
from zelda_rl.gohma import GOHMA_ATTACK, GOHMA_VULNERABLE, find_gohma, gohma_state  # noqa: E402
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR, load_policy, make_env, resolve_checkpoint  # noqa: E402

SHIELD_B = COMBAT_BUTTON_NAMES_V5.index("+Shield")   # derived, never hand-counted (AGENTS.md 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--slots", default="3,4,5,6,7,8,9")
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=20, help="total episodes across all envs")
    ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args()

    slots = [int(s) for s in a.slots.split(",")]
    model, cfg, _ = load_policy(resolve_checkpoint(a.checkpoint))
    if a.episode_seconds:
        cfg = {**cfg, "episode_seconds": a.episode_seconds}
    n_act = cfg["n_actions"]
    if n_act != len(COMBAT_ACTIONS_V5):
        print(f"!! checkpoint has {n_act} actions, v5/v7 space has {len(COMBAT_ACTIONS_V5)}; "
              f"button buckets assume the v5 space")
    print(f"checkpoint obs_v{cfg.get('obs_version')} actions {n_act} recurrent {cfg.get('recurrent')}\n"
          f"shield = button index {SHIELD_B} ('{COMBAT_BUTTON_NAMES_V5[SHIELD_B]}'), "
          f"{sum(1 for d, b in COMBAT_ACTIONS_V5 if b == SHIELD_B)} of {len(COMBAT_ACTIONS_V5)} actions "
          f"-> uniform-random baseline {sum(1 for d, b in COMBAT_ACTIONS_V5 if b == SHIELD_B)/len(COMBAT_ACTIONS_V5)*100:.1f}%")

    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"shield_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)

    def build(i):
        env, _ = make_env(cfg, root / f"i{i}", slots=slots, slots_dir=a.slots_dir,
                          video_backend="Null", unlimited_speed=True, idx=i)
        return env

    t0 = time.time()
    with ThreadPoolExecutor(a.envs) as pool:
        envs = list(pool.map(build, range(a.envs)))
    print(f"{a.envs} Dolphins up in {time.time()-t0:.1f}s")

    # counters
    btn = Counter()                 # button bucket the policy chose
    st = {"steps": 0, "shield_act": 0, "bit_set": 0, "act_and_bit": 0, "act_no_bit": 0, "bit_no_act": 0,
          "attack_steps": 0, "attack_shield_bit": 0, "within150": 0, "within150_bit": 0,
          "shield_dists": [], "deaths": 0, "eps": 0}
    # per-direction: does "+Shield" only register when the stick is NEUTRAL? 8 of the 9 shield actions carry a
    # direction, and in OoT Link does not raise the shield while walking, so most of them may be no-ops.
    sh_dir = Counter(); sh_dir_bit = Counter(); sh_dir_z = Counter(); sh_dir_z_bit = Counter()
    rng = np.random.default_rng(0)
    recurrent = bool(cfg.get("recurrent"))

    done_eps = 0
    while done_eps < a.episodes:
        slot = int(rng.choice(slots))
        obs = np.stack([e.reset(slot) for e in envs])
        h = model.init_hidden(len(envs)) if recurrent else None
        first = torch.ones(len(envs))
        alive = [True] * len(envs)
        while any(alive):
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                if recurrent:
                    act, _, _, _, h = model.act(x, h, first, deterministic=a.deterministic)
                else:
                    act, _, _, _ = model.act(x, deterministic=a.deterministic)
            first = torch.zeros(len(envs))
            acts = act.numpy().tolist()
            # send to every live env FIRST, then wait, then observe: that is what makes 8 Dolphins run in
            # parallel rather than in series (zelda_rl/env.py is send_action -> ready() -> observe()).
            live_i = [i for i in range(len(envs)) if alive[i]]
            for i in live_i:
                ai = int(acts[i])
                _, b = COMBAT_ACTIONS_V5[ai] if ai < len(COMBAT_ACTIONS_V5) else (None, -1)
                btn[b] += 1
                envs[i].send_action(ai)
            for i in live_i:
                while not envs[i].ready():
                    time.sleep(0.0005)
            for i in live_i:
                e = envs[i]
                ai = int(acts[i])
                _, b = COMBAT_ACTIONS_V5[ai] if ai < len(COMBAT_ACTIONS_V5) else (None, -1)
                chose = (b == SHIELD_B)
                o, r, term, trunc, info = e.observe()
                obs[i] = o
                cs = getattr(e, "cstate", None)
                bit = bool(cs.bit(SF1_SHIELD)) if cs is not None else False
                st["steps"] += 1
                st["shield_act"] += chose
                st["bit_set"] += bit
                st["act_and_bit"] += (chose and bit)
                st["act_no_bit"] += (chose and not bit)
                st["bit_no_act"] += (bit and not chose)
                if chose:
                    dn = DIR_NAMES[DIRECTIONS.index(COMBAT_ACTIONS_V5[ai][0])]
                    sh_dir[dn] += 1; sh_dir_bit[dn] += bit
                    zt = bool(cs.bit(SF1_ZTARGET)) if cs is not None else False
                    sh_dir_z[(dn, zt)] += 1; sh_dir_z_bit[(dn, zt)] += bit
                if cs is not None:
                    g = find_gohma(cs)
                    if g is not None:
                        d = math.dist(g.pos, cs.pos)
                        if gohma_state(g) in GOHMA_ATTACK:
                            st["attack_steps"] += 1
                            st["attack_shield_bit"] += bit
                        if d < 150.0:
                            st["within150"] += 1
                            st["within150_bit"] += bit
                        if bit:
                            st["shield_dists"].append(d)
                if term or trunc:
                    alive[i] = False
                    st["eps"] += 1
                    st["deaths"] += bool(cs is not None and cs.bit(SF1_DEAD))
                    done_eps += 1
        print(f"  slot {slot}: {done_eps}/{a.episodes} episodes, {st['steps']:,} steps, "
              f"{st['steps']/(time.time()-t0):.0f} steps/s")

    for e in envs:
        try:
            e.close()
        except Exception:
            pass

    n = max(st["steps"], 1)
    print(f"\n=== {st['steps']:,} steps over {st['eps']} episodes, slots {slots} ===")
    print("\nbutton bucket the POLICY CHOSE:")
    for b, c in btn.most_common():
        nm = COMBAT_BUTTON_NAMES_V5[b] if 0 <= b < len(COMBAT_BUTTON_NAMES_V5) else f"?{b}"
        print(f"  {nm or '(none)':<10} {c:8,}  {c/n*100:6.2f}%{'   <-- SHIELD' if b == SHIELD_B else ''}")
    print(f"\nshield action chosen      {st['shield_act']:8,}  {st['shield_act']/n*100:6.2f}%")
    print(f"SF1_SHIELD bit set        {st['bit_set']:8,}  {st['bit_set']/n*100:6.2f}%")
    print(f"  chose shield AND bit    {st['act_and_bit']:8,}  "
          f"{st['act_and_bit']/max(st['shield_act'],1)*100:6.2f}% of shield actions registered")
    print(f"  chose shield, NO bit    {st['act_no_bit']:8,}  (if this dominates, the action does not work)")
    print(f"  bit set, other action   {st['bit_no_act']:8,}  (shield held across steps / auto-shield)")
    print(f"\nwhile Gohma is ATTACKING  {st['attack_steps']:8,} steps, shielded {st['attack_shield_bit']:,} "
          f"({st['attack_shield_bit']/max(st['attack_steps'],1)*100:.2f}%)")
    print(f"within 150 units of her   {st['within150']:8,} steps ({st['within150']/n*100:.2f}% of all steps) "
          f"-- this is what `shield_steps` requires, which is why it reads 0")
    if st["shield_dists"]:
        d = np.array(st["shield_dists"])
        print(f"distance to Gohma when shielding: median {np.median(d):.0f}, "
              f"p10 {np.percentile(d,10):.0f}, p90 {np.percentile(d,90):.0f}, min {d.min():.0f}")
    print("\nshield action -> SF1_SHIELD, by stick direction ('-' is neutral):")
    for dn, c in sorted(sh_dir.items(), key=lambda kv: -kv[1]):
        print(f"  dir {dn:<3} chosen {c:6,}  bit set {sh_dir_bit[dn]:6,}  {sh_dir_bit[dn]/max(c,1)*100:6.2f}%")
    print("\n  ... split by Z-target held:")
    for (dn, zt), c in sorted(sh_dir_z.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  dir {dn:<3} ztarget {str(zt):<5} chosen {c:6,}  bit {sh_dir_z_bit[(dn,zt)]:6,}  "
              f"{sh_dir_z_bit[(dn,zt)]/max(c,1)*100:6.2f}%")
    print(f"\ndeaths {st['deaths']}/{st['eps']}")


if __name__ == "__main__":
    main()
