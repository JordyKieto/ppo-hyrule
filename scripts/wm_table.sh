#!/bin/bash
# Epoch x horizon skill table for the running world-model job on the GPU pod.
#   scripts/wm_table.sh          one shot
#   scripts/wm_table.sh -w       refresh every 60s (macOS has no `watch`, so this loops)
#   scripts/wm_table.sh -w 20    refresh every 20s
#   RUN=jepa_deep scripts/wm_table.sh      a different run directory
POD_IP="${POD_IP:-213.181.105.225}"
POD_PORT="${POD_PORT:-12722}"
RUN="${RUN:-jepa_h5}"
SSH=(ssh -p "$POD_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o ConnectTimeout=15 "root@$POD_IP")
shot() { "${SSH[@]}" "python /root/table.py /root/zelda-RL/runs/$RUN" 2>&1; }
if [ "${1:-}" = "-w" ]; then
  n="${2:-60}"
  while true; do printf '\033[H\033[2J'; shot; echo; echo "  (refresh ${n}s — ctrl-C to stop)"; sleep "$n"; done
else
  shot
fi
