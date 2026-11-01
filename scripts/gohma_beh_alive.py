"""Trainer-shaped liveness for a TRAINING fleet: is every pod actually running an update loop?

EXPERIMENT-CONTRACT: exempt -- a read-only reporter; it trains nothing and evaluates nothing.

WHY THIS EXISTS, rather than `fleet audit`. `fleet audit` greps tmux session **`zrl`**, which is the
COLLECTOR's session, and counts episodes under a collector's output directory. `fleet launch` runs a trainer
in tmux session **`train`**. So on a training fleet audit's `P` is always 0 and its liveness collapses to
`pgrep -c dolphin-emu` -- which once reported four pods idle that were mid-Dolphin-boot, the exact AGENTS 22
trap, and which is also true of a pod whose trainer has died leaving its Dolphins orphaned.

AGENTS 22: measure the signal on a machine in each state before automating on it. The signals here, in
descending order of how much they actually prove:

  updates     rows in runs/<run>/updates.jsonl -- the trainer COMPLETED that many update loops. The only
              signal that proves training, because the trainer writes a row at the end of each update.
  log_age_s   seconds since the train log was last written. A healthy run writes every rollout.
  train_tmux  the tmux session the trainer runs in exists. True the instant it starts, so it proves
              STARTED, never healthy.
  dolphins    pgrep -c dolphin-emu. 8 expected. Orphaned Dolphins outlive a dead trainer, so a high count
              is not evidence either.
  exit/crash  the tail of the log, when there are no updates and no tmux -- which is the case worth reading.

    .venv/bin/python scripts/gohma_beh_alive.py --experiment gohma-beh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fleet.cli import R_REPO, for_pods, load_state, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="gohma-beh")
    ap.add_argument("--ctl-tag", default="ctl")
    ap.add_argument("--tail", type=int, default=0, help="print this many log lines for each unhealthy pod")
    a = ap.parse_args()
    st = load_state(a.experiment)

    def one(pod):
        name = pod.get("run_name") or ""
        cmd = (f"N={R_REPO}/runs/{name}; "
               f"U=$(wc -l < $N/updates.jsonl 2>/dev/null || echo 0); "
               f"L=$(ls {R_REPO}/logs/train_{name}.log 2>/dev/null); "
               f"A=$([ -n \"$L\" ] && echo $(( $(date +%s) - $(stat -c %Y \"$L\") )) || echo -1); "
               f"T=$(tmux has-session -t train 2>/dev/null && echo 1 || echo 0); "
               # bracket the pattern: an unbracketed pgrep -f matches the remote shell's OWN argv (AGENTS 23)
               f"D=$(ps -eo cmd | grep -c '[d]olphin-emu' || true); "
               f"echo \"$U|$A|$T|$D\"; "
               f"[ \"$U\" = 0 ] && [ -n \"$L\" ] && tail -c 400 \"$L\" | tr '\\n' ' '; true")
        try:
            out = run(pod, cmd, capture=True, check=False, timeout=90)
        except Exception as e:                                                   # noqa: BLE001
            return (pod["rank"], name, -1, -1, -1, -1, f"ssh failed: {e}")
        head, _, rest = out.strip().partition("\n")
        try:
            u, age, t, d = (int(x) for x in head.split("|"))
        except ValueError:
            return (pod["rank"], name, -1, -1, -1, -1, f"unparsable: {out[:120]!r}")
        return (pod["rank"], name, u, age, t, d, rest.strip())

    res = sorted(for_pods(st["pods"], one, workers=33))
    print(f"{'rank':>4} {'arm':>5} {'updates':>7} {'log_age_s':>9} {'tmux':>4} {'dolphins':>8}  note")
    healthy = []
    for rank, name, u, age, t, d, note in res:
        arm = "ctl" if f"-{a.ctl_tag}-r" in name else "beh" if name else "?"
        ok = u > 0 or (t == 1 and age >= 0 and age < 600)
        healthy.append(ok)
        flag = "" if ok else "  <<< NOT TRAINING"
        print(f"{rank:>4} {arm:>5} {u:>7} {age:>9} {t:>4} {d:>8}{flag}")
        if not ok and a.tail and note:
            print(f"       log tail: {note[:a.tail * 80]}")

    n = len(res)
    good = sum(healthy)
    cost = sum(p.get("cost_per_hr") or 0 for p in st["pods"])
    with_updates = sum(1 for _, _, u, _, _, _, _ in res if u > 0)
    print(f"\n{good}/{n} pods are training; {with_updates}/{n} have COMPLETED at least one update")
    print(f"${cost:.2f}/hr total; ${cost * (n - good) / max(n, 1):.2f}/hr on pods that are not training")
    print("`updates` is the only column that proves an update loop ran. tmux proves STARTED, and a dolphin "
          "count can be orphans of a dead trainer (AGENTS 19/22).")
    return 0 if good == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
