#!/bin/bash
# Live tabular metrics for the AUTOREGRESSIVE token world model on pod 2.
#   scripts/token_watch.sh          one shot
#   scripts/token_watch.sh -w       refresh every 60s (macOS has no `watch`, so this loops)
#   scripts/token_watch.sh -w 20    refresh every 20s
#   RUN=wm_v5_100 scripts/token_watch.sh    a different run directory
POD_IP="${POD_IP:-103.207.149.118}"
POD_PORT="${POD_PORT:-12084}"
RUN="${RUN:-wm_ar}"
SSH=(ssh -p "$POD_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o ConnectTimeout=15 "root@$POD_IP")
shot() { "${SSH[@]}" "python /root/token_table.py /root/zelda-RL/runs/$RUN" 2>&1; }
if [ "${1:-}" = "-w" ]; then
  n="${2:-60}"
  while true; do printf '\033[H\033[2J'; shot; echo; echo "  (refresh ${n}s — ctrl-C to stop)"; sleep "$n"; done
else
  shot
fi
