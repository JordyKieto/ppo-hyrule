"""Is every start slot a genuine ENGAGED combat state, or is one of them pre-fight?

Starting the fight is a different task from winning it. The cutscene is triggered by looking up at the
ceiling, and a savestate taken before that puts the policy in `ceiling_wait` with Gohma a thousand units
away -- so it spends its episodes learning to trigger a cutscene instead of learning to fight. Three slots
(1, 2, 10) were exactly this and were rebuilt as engaged states; `docs/GOHMA_FOCUS.md` records the audit.

That audit is a DOCUMENT. This is the check. The savestates a fleet trains on come from S3, not from the
Mac, so the only trustworthy answer is read out of the emulator on the machine that will use them --
AGENTS 16, verify the artefact rather than the command, and the memory note that the live Dolphin slots have
been clobbered before.

    [xvfb-run -a] python scripts/audit_gohma_slots.py --slots 3,4,5,6,7,8
    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/audit_gohma_slots.py        # on the Mac

Exit code is non-zero if any slot is pre-fight, has no live Gohma, or sits beyond --max-dist, so this can
gate a launch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

torch.set_num_threads(1)          # AGENTS 28: never let torch take the cores Dolphin needs

from zelda_rl.combat import CombatConfig, read_combat_state                      # noqa: E402
from zelda_rl.gohma import GOHMA_CEILING, find_gohma, gohma_state, hdist, kind    # noqa: E402
from zelda_rl.obs import read_state_file_v2                                       # noqa: E402
from zelda_rl.policy_io import DEFAULT_SLOTS_DIR                                  # noqa: E402

PREFIGHT = {"ceiling_wait"}       # the intro state: she hangs there until the cutscene is triggered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--slots-dir", default=DEFAULT_SLOTS_DIR)
    ap.add_argument("--settle", type=int, default=6, help="steps to run before reading, so state decodes")
    ap.add_argument("--max-dist", type=float, default=900.0,
                    help="fail a slot further than this: the pre-fight entrance states were 1,143-1,286 units")
    ap.add_argument("--out", default="runs/gohma_slot_audit.json")
    a = ap.parse_args()

    from zelda_rl.env import DolphinGoalEnv, EnvConfig
    slots = [int(x) for x in a.slots.split(",")]
    d = Path(a.slots_dir)
    files = {k: d / f"NACE01.s{k:02d}" for k in slots if (d / f"NACE01.s{k:02d}").exists()}
    missing = [k for k in slots if k not in files]
    if missing:
        print(f"MISSING savestates for slots {missing} in {d}")
    if not files:
        print("no savestates found")
        return 2
    states = {k: read_state_file_v2(f) for k, f in files.items()}
    cfg = EnvConfig(episode_vi=36000, vi_per_step=6, video_backend="Null", unlimited_speed=True,
                    obs_version=7, task="combat",
                    combat=CombatConfig(profile="gohma", max_seconds=300.0, target_ids=(0x28, 0x55)),
                    stall_timeout=30.0)
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"slotaudit_{os.getpid()}"
    env = DolphinGoalEnv(0, root / "i0", files, states, states[min(files)], cfg)

    rows, bad = [], []
    try:
        for k in sorted(files):
            env.reset(k)
            for _ in range(a.settle):                 # a few steps so the action function decodes
                env.send_action(0)
                while not env.ready():
                    time.sleep(0.0005)
                env.observe()
            st = read_combat_state(env.ram)
            g = find_gohma(st)
            gs = gohma_state(g) if g else "ABSENT"
            dist = hdist(g.pos, st.pos) if g else float("nan")
            larvae = sum(1 for e in st.enemies.values() if kind(e) == "larva" and e.hp > 0)
            why = []
            if g is None:
                why.append("no live Gohma in the state")
            elif gs in PREFIGHT:
                why.append(f"PRE-FIGHT ({gs}): needs the cutscene triggered by looking up")
            elif math.isfinite(dist) and dist > a.max_dist:
                why.append(f"{dist:.0f}u away, beyond --max-dist {a.max_dist:.0f}")
            rows.append({"slot": k, "gohma_state": gs, "gohma_hp": (g.hp if g else 0),
                         "dist_h": None if not math.isfinite(dist) else round(dist, 1),
                         "larvae_alive": larvae, "link_hp": st.hp, "link_hp_cap": st.hp_cap,
                         "on_ceiling": gs in GOHMA_CEILING, "ok": not why,
                         "why": "; ".join(why), "file": str(files[k]),
                         "bytes": files[k].stat().st_size})
            if why:
                bad.append(k)
    finally:
        try:
            env.close()
        except Exception:                                                        # noqa: BLE001
            pass

    print(f"\n{'slot':>5}{'gohma state':>18}{'hp':>4}{'dist_h':>9}{'larvae':>8}{'linkHP':>8}"
          f"{'ceiling':>9}  verdict")
    for r in rows:
        print(f"{r['slot']:>5}{r['gohma_state']:>18}{r['gohma_hp']:>4}"
              f"{(str(r['dist_h']) + 'u') if r['dist_h'] is not None else '-':>9}{r['larvae_alive']:>8}"
              f"{str(r['link_hp']) + '/' + str(r['link_hp_cap']):>8}{'yes' if r['on_ceiling'] else 'no':>9}"
              f"  {'ENGAGED' if r['ok'] else 'FAIL: ' + r['why']}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"slots_dir": str(d), "rows": rows, "bad": bad}, indent=2))
    print(f"\nwrote {out}")
    if bad:
        print(f"NOT SAFE TO TRAIN ON: slots {bad}. Starting the fight is a different task from winning it; "
              f"drop them from --start-slots or rebuild them (docs/GOHMA_FOCUS.md).")
        return 1
    print(f"all {len(rows)} slots are engaged combat states")
    return 0


if __name__ == "__main__":
    sys.exit(main())
