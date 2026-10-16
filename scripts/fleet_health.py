"""One command that answers "is every arm alive, and will it stay alive": liveness, progress, AND DISK.

WHY THIS EXISTS. On 2026-10-16 nineteen of twenty-four sweep arms and all five hybrid arms died and nobody
noticed until the author asked. Two causes, neither visible from any existing command:

  * the pods ran out of disk (collected episodes + a downloaded corpus + 8 Dolphin instance dirs each
    copying a 228 MB NAND) and training died writing a savestate;
  * a crash in the RvS buffer merge killed the arm, leaving the pod idle and billing.

`fleet audit` reports WORKING/IDLE but says nothing about disk, and nothing was polling it. This prints one
line per pod -- alive, last update, free disk -- and exits NON-ZERO when any pod is dead or below a disk
floor, so it can be the command a Monitor watches (AGENTS 19, 21).

    .venv/bin/python scripts/fleet_health.py --experiment gohma-s3 --min-free-gb 3
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fleet.cli import load_state, run, select  # noqa: E402

# Deliberately boring: no arithmetic, no stat, no nested quoting. The first version used all three and
# silently returned four fields instead of six, which made every pod look "unreachable" -- a health check
# that lies about health is worse than none (AGENTS 23).
PROBE = (
    "cd /workspace/zelda-RL 2>/dev/null; "
    "P=$(ps -eo args | grep -c '[t]rain_combat\\|[e]val_fleet\\|[t]rain_bc_human\\|[t]rain_rvs'); "
    "D=$(df -BG / | awk 'NR==2{gsub(/G/,\"\",$4); print $4}'); "
    "L=$(ls -t logs/*.log 2>/dev/null | head -1); "
    "U=$(grep -cE '^[0-9:]+ INFO +[0-9]+ ' \"$L\" 2>/dev/null); "
    "F=$(find \"$L\" -mmin -8 2>/dev/null | wc -l); "
    "X=$(grep -cE 'Traceback|No space left|Error:' \"$L\" 2>/dev/null); "
    "echo \"P=$P D=$D U=$U FRESH=$F X=$X LOG=$(basename \"$L\" 2>/dev/null)\""
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", action="append", default=None,
                    help="repeatable. DEFAULT: every experiment with a live state file, because a check that "
                         "has to be pointed at the right arm is a check that misses the other nine")
    ap.add_argument("--min-free-gb", type=int, default=3)
    ap.add_argument("--stale-min", type=int, default=8, help="a log untouched this long means stuck")
    ap.add_argument("--quiet-ok", action="store_true", help="print nothing when everything is healthy")
    a = ap.parse_args()
    # Only experiments whose pods still EXIST. State files outlive their pods, and probing retired ones
    # produced twelve "unreachable" alarms for fleets terminated weeks ago -- a checker that cries wolf
    # gets ignored, which is how the real failures stayed invisible.
    live_ids = set()
    try:
        from fleet.runpod_api import RunPod
        live_ids = {p_.get("id") for p_ in (RunPod()._req("GET", "/pods") or [])}
    except Exception as e:                                                       # noqa: BLE001
        print(f"(could not list live pods: {str(e)[:60]}; probing all state files)")
    exps = a.experiment
    if not exps:
        # Discover everything. Nineteen arms died unnoticed because nothing was watching the experiment they
        # happened to be in; the default must cover all of them.
        exps = []
        for f in sorted((Path(__file__).resolve().parents[1] / "runs" / "fleet").glob("*.json")):
            try:
                import json
                pods_ = json.loads(f.read_text()).get("pods") or []
                if pods_ and (not live_ids or any(q.get("id") in live_ids for q in pods_)):
                    exps.append(f.stem)
            except Exception:                                                    # noqa: BLE001
                pass
    bad, lines = [], []
    for exp in exps:
        try:
            st = load_state(exp)
        except Exception:                                                        # noqa: BLE001
            continue
        pods = select(st, None)

        def probe(pod):
            try:
                return pod["rank"], run(pod, PROBE, capture=True).strip().splitlines()[-1]
            except Exception as e:                                                # noqa: BLE001
                return pod["rank"], f"ERR|{str(e)[:40]}"

        with ThreadPoolExecutor(max_workers=min(24, len(pods) or 1)) as ex:
            res = sorted(ex.map(probe, pods))
        for rank, raw in res:
            kv = dict(t.split("=", 1) for t in raw.split() if "=" in t)
            if "P" not in kv:
                bad.append(f"{exp} r{rank}: unreachable ({raw[:50]})")
                lines.append(f"  {exp} r{rank:<3} UNREACHABLE  {raw[:50]}")
                continue
            procs = int(kv.get("P") or 0)
            free = int(kv.get("D") or 0)
            upd = kv.get("U") or "0"
            fresh = kv.get("FRESH") == "1"
            flag = ""
            crashed = int(kv.get("X") or 0) > 0
            # No process is only a problem if the log ended in an error. gohma-s2 was reported "DEAD" on
            # 2026-10-16 when all 7 arms had in fact finished: train -> 80-episode eval -> clean exit.
            if procs == 0 and crashed:
                flag = "CRASHED (log has a traceback / disk error)"
            elif procs == 0:
                flag = ""
                lines.append(f"  {exp} r{rank:<3} upd {upd:>4}  {free:>3} GB  FINISHED (clean exit)  "
                             f"{kv.get('LOG','-')[:30]}")
                if free < a.min_free_gb:
                    bad.append(f"{exp} r{rank}: finished, but DISK {free} GB FREE")
                continue
            elif not fresh:
                flag = "STUCK (log not written in 8 min)"
            if free < a.min_free_gb:
                flag = (flag + "; " if flag else "") + f"DISK {free} GB FREE"
            if flag:
                bad.append(f"{exp} r{rank}: {flag}")
            lines.append(f"  {exp} r{rank:<3} upd {upd:>4}  {free:>3} GB  {procs} proc  "
                         f"{'fresh' if fresh else 'stale'}  {kv.get('LOG','-')[:30]}  {flag}")
    if bad or not a.quiet_ok:
        print("\n".join(lines))
    if bad:
        print(f"\n{len(bad)} PROBLEM(S):")
        for b in bad:
            print(f"  !! {b}")
        return 1
    if not a.quiet_ok:
        print("\nall pods healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
