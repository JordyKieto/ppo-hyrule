#!/usr/bin/env bash
# Scale zelda-RL PPO training out to a RunPod pod: push code + game assets over rsync/ssh, build
# Dolphin 2603a on the pod, run scripts/train_ppo.py with N Dolphin instances in a detached tmux
# session, watch it, and sync checkpoints back to this Mac for local evaluation.
#
# Same conventions as sunnypilot's scripts/push_to_runpod.sh / push_cf_planning_to_runpod.sh:
# direct TCP SSH (not the ssh.runpod.io proxy), connection from env vars (IP:port change on every
# pod restart), everything that must survive a restart lives under /workspace, credentials go over as
# a mode-600 file (never on a remote command line, where `ps` shows them), tmux kill and new-session
# as two separate commands, and success is only reported after checking for evidence.
#
#   export RUNPOD_HOST=103.207.149.58 RUNPOD_PORT=13494
#
# Usage (run locally, from anywhere):
#   ./scripts/runpod.sh                         # push code only
#   ./scripts/runpod.sh --setup                 # + apt (rsync/tmux/python venv/Dolphin runtime libs) + pip deps
#   ./scripts/runpod.sh --build-dolphin         # + build Dolphin 2603a from source on the pod (once per volume)
#   ./scripts/runpod.sh --assets                # + push NAND (installed OoT VC title + save) and NACE01.s0* states
#   ./scripts/runpod.sh --verify                # + run scripts/verify_pod_dolphin.py on the pod (acceptance test)
#   ./scripts/runpod.sh --train                 # + (re)start training in tmux session 'zelda_train'
#   ./scripts/runpod.sh --status                # ALIVE/DEAD, last log lines, log age, wandb URL, checkpoints
#   ./scripts/runpod.sh --watch                 # tail -f the training log (Ctrl-C stops watching, not training)
#   ./scripts/runpod.sh --sync                  # pull runs/<run>/ (checkpoints, logs, plots) back to ./runs/<run>/
#   ./scripts/runpod.sh --stop                  # kill the tmux session + Dolphin processes, then --sync
# First launch on a fresh pod:
#   ./scripts/runpod.sh --setup --build-dolphin --assets --verify --train
#
# Knobs (env): RUN_NAME (default pod-<timestamp> on --train, else the last run started on the pod),
#   NUM_ENVS (default: pod CPUs / 2), TOTAL_UPDATES (1000), VIDEO_BACKEND (Null), RESUME=1 (--resume),
#   DISPLAY_MODE=offscreen|xvfb, TORCH_THREADS (4), EXTRA_ARGS='--lr 1e-4 ...', SYNC_EVERY=600
#   (--sync loops), TMUX_SESSION (zelda_train), LOCAL_DOLPHIN_DIR (~/Library/Application Support/Dolphin).
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

: "${RUNPOD_HOST:?set RUNPOD_HOST first, e.g. export RUNPOD_HOST=103.207.149.58}"
: "${RUNPOD_PORT:?set RUNPOD_PORT first, e.g. export RUNPOD_PORT=13494}"
RUNPOD_USER="${RUNPOD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_TARGET="${RUNPOD_USER}@${RUNPOD_HOST}"

REMOTE_DIR="/workspace/zelda-RL"          # code, assets, runs: under /workspace so they survive restarts
REMOTE_VENV="/workspace/venv"
DOLPHIN_ROOT="/workspace/dolphin"
REMOTE_DOLPHIN_BIN="$DOLPHIN_ROOT/install/bin/dolphin-emu"   # DolphinQt: nogui has no hotkeys (see pod_build_dolphin.sh)
# pgrep/pkill -f pattern that matches Dolphin but not the remote `bash -c` running pkill itself
# (whose command line contains the literal "[d]olphin-emu", which this regex does not match).
DOLPHIN_PAT="$DOLPHIN_ROOT/install/bin/[d]olphin-emu"
TMUX_SESSION="${TMUX_SESSION:-zelda_train}"
LOCAL_DOLPHIN_DIR="${LOCAL_DOLPHIN_DIR:-$HOME/Library/Application Support/Dolphin}"
TITLE_DIR_REL="Wii/title/00010001/4e414345"   # NACE: OoT (USA) VC, installed in the NAND

TOTAL_UPDATES="${TOTAL_UPDATES:-1000}"
VIDEO_BACKEND="${VIDEO_BACKEND:-Null}"
DISPLAY_MODE="${DISPLAY_MODE:-offscreen}"
TORCH_THREADS="${TORCH_THREADS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NUM_ENVS="${NUM_ENVS:-}"
RESUME="${RESUME:-0}"
# CPU torch wheels by default: RAM-feature observations + a small MLP don't need a GPU (docs/RUNPOD.md).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
REMOTE_DEPS="numpy lz4 wandb pillow matplotlib"

SETUP=0; BUILD=0; ASSETS=0; VERIFY=0; TRAIN=0; STATUS=0; WATCH=0; SYNC=0; STOP=0
for arg in "$@"; do
  case "$arg" in
    --setup) SETUP=1 ;;
    --build-dolphin) BUILD=1 ;;
    --assets) ASSETS=1 ;;
    --verify) VERIFY=1 ;;
    --train) TRAIN=1 ;;
    --status) STATUS=1 ;;
    --watch) WATCH=1 ;;
    --sync) SYNC=1 ;;
    --stop) STOP=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) log "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."  # repo root, so this works no matter where it's invoked from
[[ -f scripts/train_ppo.py && -d zelda_rl ]] || { log "run this from the zelda-RL repo" >&2; exit 1; }

# accept-new: a fresh pod is never in known_hosts, but a CHANGED key is still refused.
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30)
SSH_CMD=(ssh "${SSH_OPTS[@]}" -i "$SSH_KEY" -p "$RUNPOD_PORT")
RSYNC_SSH="ssh ${SSH_OPTS[*]} -i $SSH_KEY -p $RUNPOD_PORT"
# --no-owner --no-group: rsync's chown fails (exit 23) on RunPod NetworkVolume mounts.
RSYNC=(rsync -rlptz --no-owner --no-group -e "$RSYNC_SSH")

remote() { "${SSH_CMD[@]}" "$SSH_TARGET" "$@"; }

ONLY_LOCAL_OPS=0
if [[ "$SETUP$BUILD$ASSETS$VERIFY$TRAIN" == "00000" && ( "$STATUS" == "1" || "$WATCH" == "1" || "$SYNC" == "1" || "$STOP" == "1" ) ]]; then
  ONLY_LOCAL_OPS=1   # status/watch/sync/stop don't push code (so they never disturb a running job)
fi

if [[ "$ONLY_LOCAL_OPS" == "0" ]]; then
  log "ensuring $REMOTE_DIR exists on pod ($SSH_TARGET:$RUNPOD_PORT)"
  remote "mkdir -p '$REMOTE_DIR/assets/states' '$REMOTE_DIR/assets/dolphin_user' '$REMOTE_DIR/logs'"

  log "checking remote rsync"
  if ! remote "command -v rsync >/dev/null 2>&1"; then
    log "rsync missing on pod, installing (apt output below is live, not silent)"
    remote "apt-get update && apt-get install -y rsync"
  fi

  # Code only: no venv, no local runs/ (checkpoints, Dolphin instance dirs), no assets, no caches.
  log "pushing code"
  "${RSYNC[@]}" --exclude '.venv/' --exclude 'runs/' --exclude 'assets/' --exclude '__pycache__/' \
    --exclude '.git/' --exclude '*.pt' --exclude '.DS_Store' --exclude '.zelda_secrets' \
    ./ "$SSH_TARGET:$REMOTE_DIR/"
  log "push done"
fi

if [[ "$SETUP" == "1" ]]; then
  log "apt: tmux, python venv, Dolphin runtime/build libraries (live output)"
  remote "apt-get update && apt-get install -y --no-install-recommends tmux python3 python3-venv python3-pip"
  remote "bash '$REMOTE_DIR/scripts/pod_build_dolphin.sh' --deps-only"

  # --system-site-packages: reuse a torch the base image already ships (RunPod pytorch templates).
  log "python venv at $REMOTE_VENV"
  remote "[[ -x '$REMOTE_VENV/bin/python' ]] || python3 -m venv --system-site-packages '$REMOTE_VENV'"
  remote "'$REMOTE_VENV/bin/python' -c 'import torch' 2>/dev/null || '$REMOTE_VENV/bin/pip' install --index-url '$TORCH_INDEX' torch"
  remote "'$REMOTE_VENV/bin/pip' install $REMOTE_DEPS"
  log "python deps done"
  remote "'$REMOTE_VENV/bin/python' -c 'import torch, numpy, lz4, wandb, matplotlib; print(\"stack OK: torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"cpus\", __import__(\"os\").cpu_count())'"
fi

if [[ "$BUILD" == "1" ]]; then
  if remote "SKIP_APT=1 DOLPHIN_ROOT='$DOLPHIN_ROOT' bash '$REMOTE_DIR/scripts/pod_build_dolphin.sh' --check" >/dev/null 2>&1; then
    log "Dolphin already built at $DOLPHIN_ROOT/install (rm -rf it on the pod to rebuild)"
  else
    remote "command -v tmux >/dev/null 2>&1 || (apt-get update && apt-get install -y tmux)"
    log "building Dolphin 2603a on the pod in tmux session 'dolphin_build' (log: $DOLPHIN_ROOT/build.log)"
    remote "mkdir -p '$DOLPHIN_ROOT' && tmux kill-session -t dolphin_build 2>/dev/null || true"
    remote "tmux new-session -d -s dolphin_build \"DOLPHIN_ROOT=$DOLPHIN_ROOT bash $REMOTE_DIR/scripts/pod_build_dolphin.sh > $DOLPHIN_ROOT/build.log 2>&1; echo BUILD_EXIT=\\\$? >> $DOLPHIN_ROOT/build.log\""
    # Poll instead of streaming: a 10-30 min build should not die with the SSH connection.
    while remote "tmux has-session -t dolphin_build 2>/dev/null"; do
      sleep 30
      log "  $(remote "tail -n 1 '$DOLPHIN_ROOT/build.log' 2>/dev/null | cut -c1-150" || echo '(ssh hiccup)')"
    done
    remote "tail -n 15 '$DOLPHIN_ROOT/build.log'"
    remote "grep -q '^BUILD_EXIT=0' '$DOLPHIN_ROOT/build.log'" || { log "Dolphin build FAILED (see $DOLPHIN_ROOT/build.log on the pod)" >&2; exit 1; }
    log "Dolphin build done"
  fi
fi

if [[ "$ASSETS" == "1" ]]; then
  # The NAND dir, not a WAD: the save file ("LINK") the savestates were made with lives in the
  # title's NAND data, and the env boots the installed title with `-n 000100014e414345`.
  [[ -d "$LOCAL_DOLPHIN_DIR/$TITLE_DIR_REL" ]] || { log "no OoT VC title at $LOCAL_DOLPHIN_DIR/$TITLE_DIR_REL" >&2; exit 1; }
  log "pushing NAND ($LOCAL_DOLPHIN_DIR/Wii, $(du -sh "$LOCAL_DOLPHIN_DIR/Wii" | cut -f1))"
  "${RSYNC[@]}" "$LOCAL_DOLPHIN_DIR/Wii" "$SSH_TARGET:$REMOTE_DIR/assets/dolphin_user/"
  STATES_FOUND=""
  for f in "$LOCAL_DOLPHIN_DIR"/StateSaves/NACE01.s0*; do
    [[ -f "$f" ]] && STATES_FOUND="$STATES_FOUND $(basename "$f")"
  done
  [[ -n "$STATES_FOUND" ]] || { log "no NACE01.s0* savestates in $LOCAL_DOLPHIN_DIR/StateSaves" >&2; exit 1; }
  log "pushing savestates:$STATES_FOUND"
  "${RSYNC[@]}" "$LOCAL_DOLPHIN_DIR"/StateSaves/NACE01.s0* "$SSH_TARGET:$REMOTE_DIR/assets/states/"
  remote "ls -la '$REMOTE_DIR/assets/states' && du -sh '$REMOTE_DIR/assets/dolphin_user/Wii'"
fi

# zelda_rl/dolphin.py copies the NAND from ~/Library/Application Support/Dolphin/Wii. On the pod
# that path is a symlink to the pushed assets (root's $HOME is on the container disk, so recreate it
# every time). DOLPHIN_HOST_USER_DIR is exported too, for a dolphin.py that reads it.
POD_ENV="export DOLPHIN_BIN=$REMOTE_DOLPHIN_BIN DOLPHIN_HOST_USER_DIR=$REMOTE_DIR/assets/dolphin_user PYTHONUNBUFFERED=1; \
mkdir -p \"\$HOME/Library/Application Support\" && ln -sfn $REMOTE_DIR/assets/dolphin_user \"\$HOME/Library/Application Support/Dolphin\""
if [[ "$DISPLAY_MODE" == "xvfb" ]]; then
  PY_WRAP="xvfb-run -a -s '-screen 0 640x480x24'"
else
  POD_ENV="$POD_ENV; export QT_QPA_PLATFORM=offscreen"
  PY_WRAP=""
fi

if [[ "$VERIFY" == "1" ]]; then
  log "running the pod acceptance test (Null backend, savestate load, stick input)"
  if ! remote "$POD_ENV; cd '$REMOTE_DIR' && $PY_WRAP '$REMOTE_VENV/bin/python' -u scripts/verify_pod_dolphin.py --states-dir assets/states --backend '$VIDEO_BACKEND'"; then
    log "VERIFY FAILED -- not training on this pod until it passes (see docs/RUNPOD.md 'To verify')" >&2
    exit 1
  fi
fi

# Resolve the run name: explicit RUN_NAME, else a new one on --train, else the pod's last run.
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "$TRAIN" == "1" && "$RESUME" != "1" ]]; then
    RUN_NAME="pod-$(date +%Y%m%d-%H%M%S)"
  elif [[ "$TRAIN$STATUS$WATCH$SYNC$STOP" != "00000" ]]; then
    RUN_NAME="$(remote "cat '$REMOTE_DIR/.last_run' 2>/dev/null" || true)"
    [[ -n "$RUN_NAME" ]] || { log "RUN_NAME not set and no previous run recorded on the pod" >&2; exit 1; }
  fi
fi
REMOTE_LOG="$REMOTE_DIR/logs/train_${RUN_NAME:-none}.log"

if [[ "$TRAIN" == "1" ]]; then
  remote "command -v tmux >/dev/null 2>&1 || (apt-get update && apt-get install -y tmux)"
  if ! remote "SKIP_APT=1 DOLPHIN_ROOT='$DOLPHIN_ROOT' bash '$REMOTE_DIR/scripts/pod_build_dolphin.sh' --check"; then
    log "Dolphin missing or its libraries are gone (pod restarted?): run with --setup (and --build-dolphin)" >&2
    exit 1
  fi
  remote "ls '$REMOTE_DIR'/assets/states/NACE01.s0* >/dev/null 2>&1" || { log "no savestates on pod: run with --assets" >&2; exit 1; }

  # Credentials as a file (mode 600): WANDB_API_KEY from the environment, else from ~/.netrc
  # (what `wandb login` writes). Never logged, never on a command line.
  CRED_TMP="$(mktemp)"
  trap 'rm -f "$CRED_TMP"' EXIT
  if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.netrc" ]]; then
    WANDB_API_KEY="$(python3 -c 'import netrc; a = netrc.netrc().authenticators("api.wandb.ai"); print(a[2] if a else "")' 2>/dev/null || true)"
  fi
  FOUND=""
  for key in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT; do
    if [[ -n "${!key:-}" ]]; then
      printf '%s=%s\n' "$key" "${!key}" >> "$CRED_TMP"
      FOUND="$FOUND $key"
    fi
  done
  WANDB_FLAG="--no-wandb"
  if [[ -n "$FOUND" && -n "${WANDB_API_KEY:-}" ]]; then
    log "pushing credentials for:$FOUND (values not logged)"
    "${RSYNC[@]}" "$CRED_TMP" "$SSH_TARGET:$REMOTE_DIR/.zelda_secrets"
    remote "chmod 600 '$REMOTE_DIR/.zelda_secrets'"
    WANDB_FLAG="--wandb"
  else
    log "no WANDB_API_KEY (env or ~/.netrc) -- training runs with --no-wandb"
    remote "rm -f '$REMOTE_DIR/.zelda_secrets'"
  fi

  RESUME_FLAG=""
  if [[ -n "$NUM_ENVS" ]]; then
    NUM_ENVS_LINE="NUM_ENVS=$NUM_ENVS"
  else
    NUM_ENVS_LINE='NUM_ENVS=$(( cpus / 2 > 0 ? cpus / 2 : 1 ))'
  fi
  [[ "$RESUME" == "1" ]] && RESUME_FLAG="--resume"

  # The launch command goes over as a script file: no nested tmux/ssh quoting, and it documents
  # exactly what ran. NUM_ENVS defaults to half the CPUs *granted to the container* (cgroup
  # cpu.max), not nproc, which reports the host's cores on many container runtimes.
  LAUNCH_TMP="$(mktemp)"
  cat > "$LAUNCH_TMP" <<EOF
#!/usr/bin/env bash
# generated by scripts/runpod.sh at $(date '+%Y-%m-%d %H:%M:%S') for run $RUN_NAME
set -uo pipefail
cd $REMOTE_DIR
if [[ -f .zelda_secrets ]]; then
  while IFS='=' read -r k v; do [[ -n "\$k" ]] && export "\$k=\$v"; done < .zelda_secrets
fi
$POD_ENV
cpus=\$(nproc)
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  [[ "\$quota" != "max" ]] && cpus=\$(( (quota + period - 1) / period ))
fi
$NUM_ENVS_LINE
export OMP_NUM_THREADS=$TORCH_THREADS MKL_NUM_THREADS=$TORCH_THREADS
echo "[launch] \$(date) run=$RUN_NAME cpus=\$cpus num_envs=\$NUM_ENVS backend=$VIDEO_BACKEND display=$DISPLAY_MODE shm=\$(df -h /dev/shm | tail -1)"
$PY_WRAP $REMOTE_VENV/bin/python -u scripts/train_ppo.py \\
  --num-envs "\$NUM_ENVS" --run-name $RUN_NAME --total-updates $TOTAL_UPDATES \\
  --headless --video-backend $VIDEO_BACKEND $WANDB_FLAG --out-dir runs \\
  --slots-dir assets/states $RESUME_FLAG $EXTRA_ARGS
echo "[launch] train_ppo.py exited with \$? at \$(date)"
# Don't leave orphaned emulators burning CPU after a crash.
pkill -f '$DOLPHIN_PAT' 2>/dev/null || true
EOF
  "${RSYNC[@]}" "$LAUNCH_TMP" "$SSH_TARGET:$REMOTE_DIR/logs/launch_${RUN_NAME}.sh"
  rm -f "$LAUNCH_TMP"
  remote "echo '$RUN_NAME' > '$REMOTE_DIR/.last_run'"

  log "launching run '$RUN_NAME' in detached tmux session '$TMUX_SESSION'"
  # Two INDEPENDENT commands: chaining kill + new-session can take the tmux server down with it.
  remote "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true"
  remote "pkill -f '$DOLPHIN_PAT' 2>/dev/null || true"
  remote "tmux new-session -d -s $TMUX_SESSION \"bash $REMOTE_DIR/logs/launch_${RUN_NAME}.sh 2>&1 | tee -a $REMOTE_LOG\""
  sleep 3
  if ! remote "tmux has-session -t $TMUX_SESSION 2>/dev/null"; then
    log "FAILED to start tmux session '$TMUX_SESSION' on $SSH_TARGET; last log lines:" >&2
    remote "tail -n 30 '$REMOTE_LOG' 2>/dev/null" || true
    exit 1
  fi
  if [[ "$WANDB_FLAG" == "--wandb" ]]; then
    log "waiting up to 120s for the wandb run URL"
    for _ in $(seq 1 24); do
      URL="$(remote "grep -oE 'https://wandb.ai/[^ ]+/runs/[A-Za-z0-9]+' '$REMOTE_LOG' 2>/dev/null | tail -1" || true)"
      if [[ -n "$URL" ]]; then log "wandb: $URL"; break; fi
      remote "tmux has-session -t $TMUX_SESSION 2>/dev/null" || { log "training died during startup:" >&2; remote "tail -n 40 '$REMOTE_LOG'"; exit 1; }
      sleep 5
    done
  fi
  log "launched. attach with: ssh -i $SSH_KEY -p $RUNPOD_PORT $SSH_TARGET -t 'tmux attach -t $TMUX_SESSION'"
  log "follow the log with:  ./scripts/runpod.sh --watch     (or --status for a snapshot)"
  log "pull checkpoints with: RUN_NAME=$RUN_NAME ./scripts/runpod.sh --sync"
fi

if [[ "$STATUS" == "1" ]]; then
  remote "cd '$REMOTE_DIR'; \
    tmux has-session -t $TMUX_SESSION 2>/dev/null && echo 'tmux $TMUX_SESSION: ALIVE' || echo 'tmux $TMUX_SESSION: DEAD'; \
    echo \"run: $RUN_NAME   dolphin processes: \$(pgrep -f '$DOLPHIN_PAT' | wc -l)   load: \$(cut -d' ' -f1-3 /proc/loadavg)   shm: \$(df -h /dev/shm | tail -1 | tr -s ' ' | cut -d' ' -f3,2)\"; \
    [ -f '$REMOTE_LOG' ] && echo \"log last written \$(( \$(date +%s) - \$(stat -c %Y '$REMOTE_LOG') ))s ago (heartbeat)\"; \
    echo \"wandb: \$(grep -oE 'https://wandb.ai/[^ ]+/runs/[A-Za-z0-9]+' '$REMOTE_LOG' 2>/dev/null | tail -1)\"; \
    echo \"checkpoints: \$(ls runs/$RUN_NAME/checkpoints/*.pt 2>/dev/null | wc -l), newest: \$(ls -t runs/$RUN_NAME/checkpoints/*.pt 2>/dev/null | head -1)\"; \
    echo ---; tail -n 20 '$REMOTE_LOG' 2>/dev/null; true"
fi

do_sync() {
  mkdir -p "runs/$RUN_NAME"
  # Everything except instances/ (per-env Dolphin user dirs with NAND copies, frame dumps, logs).
  "${RSYNC[@]}" --exclude 'instances/' "$SSH_TARGET:$REMOTE_DIR/runs/$RUN_NAME/" "runs/$RUN_NAME/"
  "${RSYNC[@]}" "$SSH_TARGET:$REMOTE_LOG" "runs/$RUN_NAME/pod_train.log" 2>/dev/null || true
  local newest
  newest="$(ls -t "runs/$RUN_NAME"/checkpoints/*.pt 2>/dev/null | head -1 || true)"
  log "synced runs/$RUN_NAME ($(ls "runs/$RUN_NAME"/checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ') checkpoints)"
  if [[ -n "$newest" ]]; then
    log "evaluate locally on the Mac's Dolphin:"
    log "  DOLPHIN_BIN=\${DOLPHIN_BIN:-~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin} .venv/bin/python-dbg scripts/eval_policy.py --checkpoint $newest"
  fi
}

if [[ "$STOP" == "1" ]]; then
  log "stopping tmux session '$TMUX_SESSION' and Dolphin processes on the pod"
  remote "tmux kill-session -t $TMUX_SESSION 2>/dev/null || true"
  remote "pkill -f '$DOLPHIN_PAT' 2>/dev/null || true"
  SYNC=1
  log "the pod itself keeps billing until you stop/terminate it in the RunPod console"
fi

if [[ "$SYNC" == "1" ]]; then
  if [[ -n "${SYNC_EVERY:-}" && "$STOP" != "1" ]]; then
    log "syncing runs/$RUN_NAME every ${SYNC_EVERY}s (Ctrl-C to stop)"
    while true; do do_sync || log "sync failed (pod unreachable?), retrying"; sleep "$SYNC_EVERY"; done
  else
    do_sync
  fi
fi

if [[ "$WATCH" == "1" ]]; then
  remote "tmux has-session -t $TMUX_SESSION 2>&1 >/dev/null && echo ALIVE || echo DEAD; tail -f -n 50 '$REMOTE_LOG'"
fi

log "done"
log "ssh in manually with: ssh -i $SSH_KEY -p $RUNPOD_PORT $SSH_TARGET"
