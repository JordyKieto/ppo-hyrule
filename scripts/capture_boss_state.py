"""Create new Gohma start states at the moment right before a stunning slingshot hit.

Runs its own headless Dolphin, loads a source slot, holds Z-target and keeps the slingshot drawn, and polls Gohma's
decoded state from RAM every ~2 ms. When her eye is red (the only vulnerable states) with Link locked on and the
slingshot ready to fire, it presses "Save State Slot 1" on the hotkey pipe and immediately FIRES. The save is kept
only if that shot knocks her down (stun within --hit-window s); otherwise it waits for the next red-eye window.
A kept state is then reloaded and the same shot is replayed --replays times to measure the hit rate from it.

    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/capture_boss_state.py --target floor_eye_red --source-slot 7 --out assets/boss_states
    DOLPHIN_BIN=... .venv/bin/python-dbg scripts/capture_boss_state.py --target ceiling_eye_red --source-slot 2 --out assets/boss_states
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.combat import read_combat_state  # noqa: E402
from zelda_rl.dolphin import HOTKEY_MAP  # noqa: E402
from zelda_rl.env import DolphinGoalEnv, EnvConfig  # noqa: E402
from zelda_rl.gohma import HELD_SLINGSHOT, SF1_ITEM_OUT, find_gohma, gohma_state, larvae, link_aim  # noqa: E402
from zelda_rl.obs import read_state, read_state_file_v2  # noqa: E402

D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
KNOCKED = {"fall", "land", "stunned", "stunned_hit"}


def fire_and_watch(env, window: float, hold_lock: bool = True) -> tuple[bool, list[str]]:
    """Fire one seed now (C-down press, release) and watch Gohma for `window` seconds. True if she is knocked down."""
    pad, r = env.inst.pad, env.ram
    s0 = read_combat_state(r).seeds
    pad.send("SET C 0.5 0.0"); time.sleep(0.1); pad.send("SET C 0.5 0.5")
    t0, seq, fired = time.perf_counter(), [], None
    while time.perf_counter() - t0 < window + 1.0:
        s = read_combat_state(r)
        g = find_gohma(s)
        st = gohma_state(g) if g else "-"
        if not seq or seq[-1] != st:
            seq.append(st)
        if fired is None and s.seeds < s0:
            fired = time.perf_counter()
        if fired is not None and st in KNOCKED:
            return True, seq
        if fired is None and time.perf_counter() - t0 > 0.6:        # press didn't fire: try one more press
            pad.send("SET C 0.5 0.0"); time.sleep(0.1); pad.send("SET C 0.5 0.5")
            t0 = time.perf_counter() - 0.2
        if fired is not None and time.perf_counter() - fired > window:
            break
        time.sleep(0.005)
    return False, seq + ([] if fired else ["(no seed fired)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Gohma state to capture, e.g. floor_eye_red | ceiling_eye_red | stunned")
    ap.add_argument("--source-slot", type=int, required=True)
    ap.add_argument("--out", default="assets/boss_states")
    ap.add_argument("--name", default=None)
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--attempt-seconds", type=float, default=150.0)
    ap.add_argument("--max-dist", type=float, default=700.0)
    ap.add_argument("--no-slingshot", action="store_true", help="don't draw the slingshot (e.g. for a sword-window state)")
    ap.add_argument("--hit-window", type=float, default=1.0, help="seconds after the seed within which she must go down")
    ap.add_argument("--replays", type=int, default=5, help="reload the kept state and replay the shot this many times")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"gohma_{args.target}_from_s{args.source_slot:02d}"

    files = {args.source_slot: D / f"NACE01.s{args.source_slot:02d}"}
    states = {k: read_state_file_v2(f) for k, f in files.items()}
    env = DolphinGoalEnv(0, Path("runs/instances/capture_boss"), files, states, states[args.source_slot],
                         EnvConfig(video_backend="Null", unlimited_speed=False, obs_version=2))
    pad, hk, r = env.inst.pad, env.inst.hk, env.ram
    save_key = HOTKEY_MAP["Save State/Save State Slot 1"]
    saved = None
    try:
        for attempt in range(1, args.attempts + 1):
            env.load_slot(args.source_slot)
            pad.send("SET MAIN 0.5 0.5", "SET C 0.5 0.5", "PRESS L")
            t0, last_tap, c_down, prev_state = time.perf_counter(), 0.0, False, None
            print(f"attempt {attempt}: waiting for {args.target}", flush=True)
            while time.perf_counter() - t0 < args.attempt_seconds:
                s = read_combat_state(r)
                g = find_gohma(s)
                if s.dead or g is None:
                    print("  Link died / no Gohma -> retry", flush=True)
                    break
                st = gohma_state(g)
                if st != prev_state:
                    print(f"  {time.perf_counter() - t0:6.1f}s gohma {st:18s} hp {g.hp} d {math.dist(g.pos, s.pos):5.0f} "
                          f"link {s.hp / 16:.1f}hp held {s.held_item} locked {s.targeted == g.ptr}", flush=True)
                    prev_state = st
                now = time.perf_counter()
                # keep the slingshot drawn (single taps: alternating press/release would fire)
                if not args.no_slingshot and s.held_item != HELD_SLINGSHOT and now - last_tap > 0.3:
                    pad.send("SET C 0.5 0.0"); c_down, last_tap = True, now
                elif c_down and now - last_tap > 0.08:
                    pad.send("SET C 0.5 0.5"); c_down = False
                ready = args.no_slingshot or (s.held_item == HELD_SLINGSHOT and s.bit(SF1_ITEM_OUT))
                d = math.dist(g.pos, s.pos)
                locked = s.targeted == g.ptr
                if st == args.target and ready and locked and d <= args.max_dist and not s.dead and now - t0 > 1.0:
                    vi = read_state(r).vi
                    hk.send(f"PRESS {save_key}"); time.sleep(0.05); hk.send(f"RELEASE {save_key}")
                    time.sleep(0.05)
                    aim_h, aim_v = link_aim(s, g.focus)
                    cand = {"name": name, "target_state": st, "source_slot": args.source_slot, "vi_at_trigger": vi,
                            "gohma_hp": g.hp, "gohma_dist": round(d), "gohma_on_ceiling": g.pos[1] > -490,
                            "link_hearts": s.hp / 16, "seeds": s.seeds, "held_item": s.held_item, "locked_on_gohma": locked,
                            "link_aim_to_eye_cos": round(aim_h, 3), "link_aim_elevation": round(aim_v, 3),
                            "larvae_alive": sum(1 for e in larvae(s) if e.hp > 0), "seconds_after_load": round(now - t0, 2)}
                    hit, after = fire_and_watch(env, args.hit_window)
                    print(f"  candidate at {st} d {round(d)} aim {aim_h:+.2f}/{aim_v:+.2f}: shot -> {' > '.join(after)} "
                          f"{'HIT (kept)' if hit else 'miss (discarded)'}", flush=True)
                    if hit:
                        saved = cand
                        break
                    prev_state = None
                time.sleep(0.002)
            pad.send("RELEASE L", "SET C 0.5 0.5")
            if saved:
                break
        if not saved:
            sys.exit(f"no {args.target} moment captured in {args.attempts} attempts")

        src = env.inst.user_dir / "StateSaves" / "NACE01.s01"
        t_wait = time.time()
        while time.time() - t_wait < 20 and (not src.exists() or time.time() - src.stat().st_mtime > 30):
            time.sleep(0.2)
        time.sleep(2.0)                                   # let Dolphin finish compressing / writing
        dest = out_dir / f"{name}.sav"
        shutil.copy(src, dest)
        print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)

        # verify: reload the new file through the normal slot path and replay the shot (lock held, fire at once)
        st_file = read_state_file_v2(dest)
        env.slot_states[1] = st_file
        shutil.copy(dest, env.inst.user_dir / "StateSaves" / "NACE01.s01")
        hits, runs = 0, []
        for i in range(args.replays):
            env.load_slot(1)
            pad.send("SET MAIN 0.5 0.5", "PRESS L")
            s = read_combat_state(r)
            g = find_gohma(s)
            loaded = {"gohma_state": gohma_state(g) if g else None, "hp": g.hp if g else None, "held": s.held_item,
                      "locked": bool(g) and s.targeted == g.ptr}
            hit, seq = fire_and_watch(env, args.hit_window)
            pad.send("RELEASE L")
            hits += hit
            runs.append({"loaded": loaded, "hit": hit, "sequence": seq})
            print(f"  replay {i + 1}: loaded {loaded} -> {' > '.join(seq)} {'HIT' if hit else 'miss'}", flush=True)
        env.load_slot(1)
        chk = env.input_check(wait_vi=60)
        saved["verify"] = {"replay_hits": hits, "replays": args.replays, "stick_input_ok": chk["game_sees_stick"], "runs": runs}
        (out_dir / f"{name}.json").write_text(json.dumps(saved, indent=2))
        print(f"verified: {hits}/{args.replays} replayed shots stunned her; stick input ok = {chk['game_sees_stick']}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
