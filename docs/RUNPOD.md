# RunPod workflow (zelda-RL PPO on headless Dolphin)

## ⭐ Fleet CLI (current path): `python -m fleet`

Ported from sunnypilot's `runpod_fleet`:
- RunPod REST API for CPU pods
- SSH bootstrap
- Upstash for heartbeats, published policies and cross-pod weight averaging

Credentials are read in code from `$ZRL_SECRETS_FILE` (default: sunnypilot's `secrets.yaml`) and `~/.netrc`. They are never printed. Pods get them as a mode-600 `.fleet_secrets` file.

```sh
python -m fleet create    --experiment kokiri-v2 --count 1 --vcpu 16 --confirm   # dry run without --confirm
python -m fleet bootstrap --experiment kokiri-v2      # apt, code, NAND + savestates, Dolphin 2603a, venv, secrets
python -m fleet verify    --experiment kokiri-v2 --instances 8                     # acceptance test + speed probe
python -m fleet launch    --experiment kokiri-v2 --num-envs 8 --train-args "--total-updates 400"
python -m fleet status    --experiment kokiri-v2      # Upstash heartbeats (no SSH) + per-pod tail
python -m fleet sync      --experiment kokiri-v2      # runs/<run>/ + newest checkpoint -> checkpoints/inbox/
python -m fleet create    --experiment kokiri-v2 --count 15 --vcpu 16 --confirm    # scale out (fills rank gaps)
python -m fleet bootstrap --experiment kokiri-v2 --via-seeds 0,2 --ranks 5,...,19  # seeds rsync assets pod->pod
python -m fleet launch    --experiment kokiri-v2      # relaunch all 5 with --fleet-world-size 5
python -m fleet kill      --experiment kokiri-v2 --confirm
```

**Feedback loop (Mac):**

```sh
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/play.py --upstash kokiri-v2 --watch   # newest fleet policy, plays in local Dolphin, auto-refreshes
```

**How the fleet trains:**
- Every pod runs `train_ppo.py` with `--fleet-experiment E --fleet-rank R --fleet-world-size N`, N Dolphin envs, and the Null backend.
- Every pod starts from identical weights.
- After each PPO update, every rank uploads its weights to Upstash, waits up to 180 s for its peers, and loads the average. This is synchronous FedAvg once per update, used because RunPod pods cannot reach each other directly (see sunnypilot's README).
- `weight_hash` in the logs and heartbeats must match across ranks.
- Rank heartbeats: `zrl:E:hb:R`, 300 s TTL.
- Published policy: `zrl:E:policy:latest`, every `--publish-every` updates.
- All runs share one wandb group = experiment.

## Measured (2026-09-12, experiment `kokiri-v2`)

### Fleet / infrastructure

| Item | Result |
|---|---|
| Pod type | CPU pod, 16 vCPU, `cpu5c`/`cpu3c` flavors, 32 GB RAM, `runpod/base:1.2.0-ubuntu2404`, Secure Cloud |
| Cost | $0.56/hr or $0.48/hr per pod (flavor-dependent); **5-pod fleet $2.64/hr** |
| Create → SSH ready | ~50 s (the first ~45 s shows RUNNING with no IP/port) |
| Create failures | 2 of 4 creates hit RunPod `500 Something went wrong`; both succeeded on retry 20 s later |
| Dolphin 2603a build on the pod | **269 s** (1957 ninja steps, 16 cores); cached as a 42 MB tarball at `~/.cache/zelda-rl/` |
| First-pod bootstrap (apt + code + assets + build + venv) | **9 min**; later pods skip the build |
| Asset push (NAND 46 MB + 4 × ~40 MB savestates) | ~3 min per pod, bound by the Mac's upload link |
| `/dev/shm` on pod | **15 GB** (Dolphin needs ~97 MB per instance) |
| CPUs visible | `nproc` = 16 (cgroup), `os.cpu_count()` = **48** (host): size envs from the pod's vCPU count, not `os.cpu_count()` |
| Upstash from the Mac | PING 292 ms; 1.3 MB blob up 0.95 s / down 0.41 s |
| Published policy (v2) | 0.98 MB; pod → Upstash → Mac `checkpoints/inbox/` → loads locally ✅ |
| 2-rank weight averaging | identical weight hash on both ranks, average correct to 1e-6 ✅ |

### Emulation / env on the pod (`fleet verify`, `fleet launch` smoke)

| Check | Result |
|---|---|
| Boot + RAM attach | **0.8 s**, zero-copy mmap of `/dev/shm/dolphin-emu.<pid>` (works with `ptrace_scope=1`: the trainer owns the fd) |
| Savestate made on the Mac (ARM) loads on x86 | ✅ scene 85, Link at exactly (−1307.8, −62.1, −263.4); load 114 ms, reload 70 ms |
| Null backend runs game logic + sees pipe input | ✅ stick moved Link 175 units (under Xvfb) |
| 1 instance, uncapped | 236 VI/s = **3.9×** real time |
| 8 instances, uncapped, idle | 204–247 VI/s each, **1736 VI/s total = 28.9× real time** (load avg 6.5) |
| Trainer, 12 envs, 10 s episodes, 2 updates | 12 instances booted in 2.5 s; **214–224 env steps/s**, 1.8–1.9× per env; slots 1–3 pass the input check |
| M1 comparison | 3 envs headless: 92–100 env steps/s at 3.1–3.3× → **one 16-vCPU pod ≈ 2.5–3× the M1** |

Envs per pod: 12 envs saturated the 16 vCPUs (per-env speed halved vs 8 idle instances). The real run uses **8 envs per pod** (target ~300 env steps/s per pod, ~1,500/s fleet-wide). Confirm with `perf/env_steps_per_sec` in wandb.

### Scaling to 20 pods (2026-09-13)

| Item | Result |
|---|---|
| 15-pod create burst | several `500 Something went wrong` / `no longer any instances available`; all 15 created within 3 attempts; fleet **$10.24/hr** |
| Asset push from the Mac | ~0.7 MB/s upload (5 min per pod for ~200 MB) → would take hours for 15 pods |
| **Seed bootstrap** (`bootstrap --via-seeds 0,2`) | pod→pod rsync of code + NAND + 10 savestates + Dolphin install (~540 MB): **4–48 s per pod**; all 19 pods fully bootstrapped in ~3.5 min |
| Pod→pod reachability | public IP:SSH-port works across different public IPs (60–125 ms connect); **fails between pods sharing a public IP** (NAT hairpin) → seeds are chosen on a different IP |
| Upstash reachability | **one pod could not reach Upstash at all** (DNS gave us-east-2 IPs, every connect timed out). Bootstrap now PINGs Upstash and fails that pod. 19 new pods: all PONG in 0.06–0.48 s |
| Savestate verification on x86 (`scripts/verify_slots.py`) | slots 1–3, 5–10 ✅; **slot 4 fails intermittently** (loads correctly, but the game ignores input) → excluded |
| Launch | parallel prepare + start: all 20 ranks started within the same second |

### Deku combat run `gru-v2` → `gru-v2b` (2026-09-13, 20 pods): task solved
- **Speed:** 3,300–3,800 env steps/s fleet-wide; 41.8 M env steps over 154 global updates. $10.38/hr: gru-v2b ≈3 h 40 min ≈ $38.
- **Result:** 77% success over all 10 start slots, 3/3 successes watched locally. Full table: `docs/COMBAT_TASK.md` § Results.
- **Pod speeds:** ranks 1, 2 and 13 ended 32–48 updates behind the fastest (92, 86, 76 vs 124). Straggler-tolerant sync kept them on the current average for the whole run with **zero real divergence**, and progress was never gated on them.
- **Credit:** $42 at 12:16 → $13 at 15:02. Balance is checked with GraphQL `myself { clientBalance currentSpendPerHr }`. Checkpoints were pulled to the Mac every ~20 updates, and the fleet was stopped with `fleet kill --confirm` (0 pods left, spend $0.02/hr).

### Fleet sync design (current)

- **Straggler-tolerant sync (2026-09-13, after `gru-v2` diverged at update 38).**
  - **Seen live:** pods differ ~3.5× in speed (71–252 env steps/s: $0.48 cpu3c pods sharing a busy host vs $0.56 cpu5c). The mid-speed aggregator waited up to 120 s for the slowest rank, so the *fastest* ranks timed out waiting for the average and trained alone (5 of 20 hashes diverged).
  - **Now:** the aggregator (`--fleet-aggregator`, pick a fast pod) waits at most **30 s** for stragglers, averages what arrived, and publishes `avg` plus a `latest` pointer. Followers wait up to 300 s for their update's average, or **catch up to any newer average immediately**. Late ranks skip a round but never drift; blobs are kept for 5 rounds.
  - **Relaunch without losing progress:** verify identical weights on every pod (all 20 had `update_00030.pt` = `8a95bd0ad2e2`), then launch a new tag with `--init-from runs/<old-run>-r{rank}/checkpoints/update_00030.pt` (`{rank}` is filled per pod).
- **Aggregator weight averaging.** Every rank uploads its post-update weights. Rank 0 waits for the *live* ranks (fresh heartbeat), averages them in rank order in float64 (bit-identical everywhere), and publishes one `avg` blob. Followers download it.
  - Upstash traffic per update: ~N MB up + ~N MB to the aggregator + ~N MB of averages, instead of N² MB for all-to-all.
  - If the aggregator is slow or dead, followers keep their local weights after the timeout.
- **Per-launch keys.** Sync keys are namespaced by launch id (a relaunch once averaged with a stale previous launch's blobs). They carry a 10 min TTL, and the previous update's keys are deleted.
- **Strict timeouts.** Upstash calls use connect 5 s / read 20 s with bounded retries. Heartbeats and policy publishing run on a background thread, so an unreachable Upstash never blocks training. Stale-launch and dead-rank hangs were both seen live.
- **Watchdog.** An update taking >15 min dumps all Python stacks to the log. `kill -USR1 <pid>` dumps them on demand. py-spy is blocked in RunPod containers (no ptrace).
- **Identical-weights check.** `scripts/fleet_watch.py <experiment> <N>` alerts on real divergence, stale ranks, missing ranks, and lagging ranks.
  - With straggler catch-up, a slow rank at local update 11 legitimately holds the update-16 average. So **don't compare hashes by update number**: that gave false "hashes differ" alerts.
  - The watcher instead collects every hash shared by ≥2 ranks as a fleet average. It alerts only if a rank's hash matches none for 3 of that rank's own syncs.
  - Lag warnings fire once per rank (again only after 10 more updates of lag). Progress lines show `fleet update <fastest> (slowest <n>)`.
- **Published policy pointer.** `policy:latest` is no longer overwritten by a slow rank publishing an older update. Before this fix, `fetch_latest_policy` returned rank 2's u30 while the fleet was at u46. When in doubt, pull the aggregator's `runs/<run>-r4/checkpoints/update_XXXXX.pt` over `fleet ssh` (base64 through the SSH channel).

### Quirks of this setup (all verified the hard way)

1. **Qt `offscreen` does not boot the game.** DolphinQt starts and logs init, but never creates the RAM arena. Run everything under **`xvfb-run -a`** (the fleet does). `zelda_rl/dolphin.py` now refuses to launch on Linux without `DISPLAY`.
2. **`dolphin-emu-nogui` can't do resets.** It has no hotkey handling, so savestate loads over the hotkey pipe need the Qt binary (`dolphin-emu`), from the 2603a source.
3. **`pkill -f dolphin-emu` over SSH kills your own session**: the remote shell's command line contains the pattern. Use `pkill -x dolphin-emu` / `pgrep -x`.
4. **Pods can't reach each other** (sunnypilot finding), so the fleet coordinates only through Upstash: heartbeats, policies, weight averaging.
5. **Identical init on every rank** (`torch.manual_seed(1234)` before building the model); env RNG seeds differ per rank.
6. **Savestates are portable ARM Mac → x86 Linux** when the Dolphin version matches (2603a, state version 177).
7. **Home upload bandwidth dominates bootstrap.** A RunPod network volume with the assets and Dolphin build would make new pods near-instant (sunnypilot's `prime-volume` pattern). Not done yet.
8. **Slot 4 is unreliable** (passes the input check sometimes, fails others): not used as a start slot. Slot 5 is the goal. Start slots: 1, 2, 3, 6, 7, 8, 9, 10.
9. **Secrets:** read from sunnypilot's `secrets.yaml` / `~/.netrc` by code, shipped as a mode-600 `.fleet_secrets` file, sourced by the launch script. Never on a command line, in logs, or in this repo.

---|---|
| Upstash from the Mac | PING 292 ms; 1.3 MB blob up 0.95 s / down 0.41 s |
| 2-rank weight averaging | identical hash on both ranks, average correct to 1e-6 |
| First pod | `zrl-kokiri-v2-r0`: 16 vCPU (cpu5c/cpu3c), 32 GB, $0.56/hr, SSH ready in ~50 s |

---


The Mac stays the dev and evaluation box. A RunPod Linux pod runs `scripts/train_ppo.py` with many Dolphin instances. Checkpoints come back to the Mac, where you watch the policy play in a local Dolphin window.

All pod work goes through `scripts/runpod.sh`. It follows the same conventions as sunnypilot's `scripts/push_to_runpod.sh`:
- direct TCP SSH from env vars
- everything that must survive a restart lives under `/workspace`
- secrets go over as a mode-600 file
- tmux runs the job
- nothing reports success without evidence

**Status: nothing here has run on a real pod yet.** Section 8 lists what has been verified and what is still assumed. Section 4's acceptance test is the first thing to run on a new pod.

Remote layout:

```
/workspace/zelda-RL/                 code (rsynced), logs/, runs/<run>/ (checkpoints, jsonl, plots, instances/)
/workspace/zelda-RL/assets/dolphin_user/Wii   NAND copy (OoT VC title 000100014e414345 + the "LINK" save)
/workspace/zelda-RL/assets/states/NACE01.s0*  savestates (train_ppo.py's Linux default --slots-dir)
/workspace/dolphin/{src,build,install}        Dolphin 2603a built from source (install/bin/dolphin-emu)
/workspace/venv                               Python venv (torch CPU, numpy, lz4, wandb, matplotlib, pillow)
```

As in sunnypilot, everything outside `/workspace` sits on the **container disk**, which a real pod restart wipes. That includes apt packages (Qt runtime, tmux) and `~/.ssh/authorized_keys`. The Dolphin build and the venv live on the volume and survive. After a restart, run `--setup` again (it re-installs the apt libraries and reuses the build).

## 0. Pick a pod: CPU, not GPU (to be confirmed by section 4)

- **Observations are RAM-only.** They are read zero-copy from Dolphin's shared-memory arena, so nothing has to be rendered. The plan is Dolphin's `Null` video backend: no GPU, no display.
- **Throughput is emulation speed × instances, which is CPU-bound.** Each Dolphin uses about 2 threads (CPU thread + GPU thread; the Null GPU thread should be light). Start with `NUM_ENVS` = vCPUs / 2 and tune it with `verify_pod_dolphin.py --instances K --unlimited`.
- **The policy is a small MLP on a feature vector.** PPO updates on CPU are cheap. The default is CPU torch wheels (`TORCH_INDEX`).
- **Use x86_64 pods.** Dolphin's JIT exists for x86_64 and ARM64. RunPod's pods are x86_64, and `docker/Dockerfile` targets linux/amd64.
- **Choose a GPU pod only if** section 4 shows the Null backend breaks game logic, so a real renderer (Vulkan/OGL under `DISPLAY_MODE=xvfb`) is needed. The other reason is later pixel observations.
- **Before the first run, check `df -h /dev/shm` on the pod.** Each Dolphin needs ~97 MB of shm (section 8). Also check the CPU count the container really gets: `nproc` can report the host's cores. The launcher reads the cgroup's `cpu.max`.
- **For multiple pods, run independent runs** (different `RUN_NAME` per pod). There is no gradient sync; compare the runs in wandb.

## 1. SSH key + connection vars (every new pod / every restart)

Add your public key on the pod through the RunPod web terminal. Paste the contents of `~/.ssh/id_ed25519.pub`:

```sh
mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "<contents of ~/.ssh/id_ed25519.pub>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

Then set the connection vars on the Mac. They change on every restart, so take them from the "SSH over exposed TCP" panel:

```sh
export RUNPOD_HOST=103.207.149.58
export RUNPOD_PORT=13494
# optional: RUNPOD_USER (root), SSH_KEY (~/.ssh/id_ed25519)
ssh-keygen -R "[$RUNPOD_HOST]:$RUNPOD_PORT"   # only if the host key changed after a restart
```

## 2. First launch on a fresh pod

```sh
./scripts/runpod.sh --setup --build-dolphin --assets --verify --train
```

| Flag | What it does |
|---|---|
| (always) | Pushes the code to `/workspace/zelda-RL`. Excludes `.venv/`, `runs/`, `assets/`, `*.pt` and caches. |
| `--setup` | apt: `tmux`, `python3-venv`, plus Dolphin's build/runtime libraries (`scripts/pod_build_dolphin.sh --deps-only`: Qt6 + offscreen QPA plugin, GL/EGL, X11, udev/evdev, ALSA/Pulse, xvfb). Creates the venv with CPU torch (reuses a preinstalled torch), then numpy, lz4, wandb, matplotlib and pillow, and prints a stack check. |
| `--build-dolphin` | Builds Dolphin **2603a** from source in tmux session `dolphin_build` (log: `/workspace/dolphin/build.log`) and polls until done. Skipped if a working install already exists. It is a one-time cost per volume: roughly 10 min on 32 vCPU, longer on small pods (estimate). |
| `--assets` | Rsyncs `~/Library/Application Support/Dolphin/Wii` (the installed NACE title and its save data, ~45 MB) and `StateSaves/NACE01.s0*` (~40 MB each). The NAND is pushed instead of a WAD because the savestates reference the NAND save and the env boots the installed title with `-n`. Override the source with `LOCAL_DOLPHIN_DIR`. |
| `--verify` | Runs `scripts/verify_pod_dolphin.py` on the pod (section 4). The script exits non-zero, and nothing trains, if any check fails. |
| `--train` | Starts training in tmux session `zelda_train` (details below). |

`--train` in detail:
- It pushes `WANDB_API_KEY` (from your env, or from `~/.netrc` as written by `wandb login`) plus `WANDB_ENTITY`/`WANDB_PROJECT` to `/workspace/zelda-RL/.zelda_secrets` (mode 600).
- It writes the exact launch command to `logs/launch_<run>.sh` and runs it in tmux session `zelda_train`, with output going to `logs/train_<run>.log`. The command is:
  ```
  python scripts/train_ppo.py --num-envs $NUM_ENVS --run-name $RUN_NAME --total-updates $TOTAL_UPDATES \
    --headless --video-backend Null --wandb|--no-wandb --out-dir runs --slots-dir assets/states $EXTRA_ARGS
  ```
- Its environment is `DOLPHIN_BIN=/workspace/dolphin/install/bin/dolphin-emu` and `QT_QPA_PLATFORM=offscreen` (or wrapped in `xvfb-run` with `DISPLAY_MODE=xvfb`), plus `OMP_NUM_THREADS=$TORCH_THREADS`. `~/Library/Application Support/Dolphin` is symlinked to the pushed assets, because that is where `zelda_rl/dolphin.py` copies the NAND from.
- It checks that the tmux session exists, then waits up to 120 s for the **wandb run URL** and prints it.

Knobs: `RUN_NAME`, `NUM_ENVS`, `TOTAL_UPDATES` (1000), `VIDEO_BACKEND` (Null), `RESUME=1` (adds `--resume`, which continues from `runs/<run>/checkpoints/latest.pt`), `DISPLAY_MODE`, `TORCH_THREADS` (4), `EXTRA_ARGS`, `TMUX_SESSION`.

```sh
RUN_NAME=kokiri-goal-pod1 NUM_ENVS=14 TOTAL_UPDATES=2000 EXTRA_ARGS='--lr 1e-4' ./scripts/runpod.sh --train
```

## 3. Relaunch

- **Pod still up, code edited or run crashed:** `./scripts/runpod.sh --train`. To continue a run instead of starting a new one: `RUN_NAME=<run> RESUME=1 ./scripts/runpod.sh --train`.
- **Pod restarted** (container disk wiped): step 1, then `./scripts/runpod.sh --setup --train`. The Dolphin build, venv, assets and runs are on `/workspace` and survive.
- **Push code only:** `./scripts/runpod.sh`.

## 4. To verify on the first pod (acceptance test)

Run `./scripts/runpod.sh --verify`, or run the script on the pod directly:

```sh
cd /workspace/zelda-RL && QT_QPA_PLATFORM=offscreen DOLPHIN_BIN=/workspace/dolphin/install/bin/dolphin-emu \
  /workspace/venv/bin/python scripts/verify_pod_dolphin.py --states-dir assets/states --backend Null
```

It stops at the first failure. Each check tests an assumption that could not be verified from the Mac:

1. `DOLPHIN_BIN` and the NAND exist. It prints the free space in `/dev/shm` and `ptrace_scope`.
2. DolphinQt boots `-b -n 000100014e414345 -v Null` with no display, and `zelda_rl.linuxmem` attaches to the N64 RAM. It prints the backend used: `shm mmap (zero-copy)` expected, `vm_readv` as the fallback.
3. Emulation runs under Null: the VI counter at `0x8011DBB0` rises.
4. The **savestate loads through the hotkey pipe**: the scene and Link's position in RAM match the ones decoded from `NACE01.s01`. This proves savestate-version compatibility (x86-64 JIT loading a state made on the Mac's ARM64 build) and that DolphinQt hotkeys work offscreen.
5. **Stick input moves Link:** `SET MAIN 0.5 1.0` for 1 s must move him more than 20 units in RAM. This is the Null-backend game-logic check.
6. A second load puts Link back at the saved position.

Then size the pod:

```sh
... verify_pod_dolphin.py --instances 8 --unlimited     # VI/s per instance + total; raise K until the total stops growing
```

If step 2 or 4 fails with `offscreen`, retry with `DISPLAY_MODE=xvfb ./scripts/runpod.sh --verify`. If step 5 fails under Null, try `VIDEO_BACKEND=Vulkan DISPLAY_MODE=xvfb`; Vulkan needs a GPU pod or lavapipe.

## 5. Watch training from the Mac

- **wandb** (no SSH needed): `--train` prints the run URL; `--status` prints it again.
- **Snapshot:** `./scripts/runpod.sh --status` shows:
  - tmux ALIVE/DEAD
  - Dolphin process count, load average and `/dev/shm` usage
  - log age in seconds (the heartbeat: a healthy run writes every rollout)
  - wandb URL
  - checkpoint count and newest checkpoint
  - last 20 log lines
- **Live log:** `./scripts/runpod.sh --watch`. Ctrl-C stops watching, not training.
- **Raw tmux:** `ssh -i ~/.ssh/id_ed25519 -p "$RUNPOD_PORT" root@"$RUNPOD_HOST" -t 'tmux attach -t zelda_train'`. Detach with `Ctrl-b d`.

`--status`, `--watch`, `--sync` and `--stop` never push code, so they can't disturb a running job. Without `RUN_NAME` they use the pod's last started run (`/workspace/zelda-RL/.last_run`).

## 6. Sync checkpoints back and evaluate on the Mac

```sh
./scripts/runpod.sh --sync                        # runs/<run>/ -> ./runs/<run>/ (minus instances/), + pod_train.log
SYNC_EVERY=600 ./scripts/runpod.sh --sync         # keep pulling every 10 min
```

It prints the evaluation command for the newest checkpoint. Evaluation uses the Mac's signed Dolphin copy and `python-dbg` (FINDINGS §3.1):

```sh
DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin \
  .venv/bin/python-dbg scripts/eval_policy.py --checkpoint runs/<run>/checkpoints/update_00100.pt
```

## 7. Stop / teardown

```sh
./scripts/runpod.sh --stop      # kill tmux session + all Dolphin processes, then a final --sync
```

**The pod keeps billing until you stop or terminate it in the RunPod console.** A network volume bills separately until it is deleted.

## 8. Verified vs assumed

**Verified**, against the Dolphin source at tag `2603a` (commit `5e7cc91d`) or locally:
- **Savestate version.** The user's `NACE01.s01` header has state version **177** ("Dolphin 2603a"). `State.cpp` at 2603a has `STATE_VERSION = 177`, and loading compares only that number, not the revision string. `pod_build_dolphin.sh` refuses to build a tree with a different value.
- **Linux RAM layout.**
  - `MemArenaUnix.cpp` (used on Linux per `Common/CMakeLists.txt`) does `shm_open("/dolphin-emu.<pid>")`, then `shm_unlink`, then `ftruncate`, and keeps the fd open.
  - In `Memmap.cpp`, MEM1 is physical region 0, so it sits at **shm offset 0**. The arena is MEM1 0x2000000 + L1 0x40000 + MEM2 0x4000000 = 0x6040000 bytes.
  - `zelda_rl/linuxmem.py` was tested in a linux/arm64 Docker container against a fake process that does the same shm dance. Tested: MEM1 at offset 0 and at a non-zero offset; `auto`/`shm`/`vm_readv` methods; live updates visible through the zero-copy view; the writable mode; and the dead-pid and timeout paths.
- **`/dev/shm` limit.** Docker's default `/dev/shm` is **64 MB**. Touching a 0x6040000-byte arena there kills the process with **SIGBUS** (exit 135); with `--shm-size=1g` it works. Any container needs about 100 MB of `/dev/shm` per Dolphin instance.
- **nogui can't reset.** `dolphin-emu-nogui` has **no hotkey handling**: `HotkeyScheduler` is DolphinQt-only, headless `MainLoop` only dispatches jobs, and X11 nogui only maps raw F1-F8 keys. So the savestate-load hotkey pipe can't work there. nogui also rejects `-b`, which is GUI-only in `CommandLineParse.cpp`. **So the pod runs DolphinQt (`dolphin-emu`)** with `QT_QPA_PLATFORM=offscreen`. `DolphinQt/Main.cpp` only forces `xcb` when the variable is unset or `wayland`. nogui is still built, as `-p headless`.
- **Frontend pieces exist in 2603a.** The `Null` video backend exists (`VideoBackends/Null`, config name "Null"). The Pipes input backend is compiled on all UNIX builds.
- **Build requirements (2603a).** GCC >= 12, C++23, CMake >= 3.20, Qt6. DolphinQt guards Qt >= 6.5/6.6 features, so Ubuntu 24.04's Qt 6.4.2 should build. The apt package lists in `pod_build_dolphin.sh` and `docker/Dockerfile` resolve on `ubuntu:24.04` (checked with `apt-get install -s`).

**Assumed / unverified**. Section 4 checks most of these:
- **Qt offscreen.** DolphinQt boots and processes pipe hotkeys under `QT_QPA_PLATFORM=offscreen`. The fallback is `DISPLAY_MODE=xvfb`.
- **Null backend.** The game logic runs correctly under Null (no GPU readbacks that OoT VC depends on). Link moves, and the VI counter and scene update.
- **Cross-architecture savestates.** States saved by the Mac's ARM64 JIT load under the x86-64 JIT and replay deterministically enough for training. The state holds PowerPC state, not JIT code, so this should work.
- **The build itself.** The full from-source build of 2603a on Ubuntu 24.04 with `USE_SYSTEM_LIBS=OFF` has not been run, and neither has the Dockerfile.
- **RunPod containers:**
  - `/dev/shm` size: unknown. Check `df -h`.
  - Opening `/proc/<pid>/fd/N` works for same-uid root. It uses ptrace READ mode, which Yama does not restrict.
  - The `process_vm_readv` fallback needs ptrace ATTACH plus a seccomp profile that allows it.
- **Throughput.** Cloud vCPUs are slower per core than the M1. `NUM_ENVS` = vCPUs/2 is a starting guess.
- **Network volume I/O.** Per-env Dolphin user dirs (`runs/<run>/instances/`, a 45 MB NAND copy each) sit on `/workspace`. If that is a slow NetworkVolume mount, boot times suffer. Consider pointing instance dirs at the container disk.

## Known gotchas carried over from sunnypilot

- **rsync exit 23 on NetworkVolume.** rsync's chown fails, so the script uses `rsync -rlptz --no-owner --no-group` everywhere.
- **Host keys.** `StrictHostKeyChecking=accept-new` handles fresh pods. Run `ssh-keygen -R` after a restart that rotated the host key.
- **tmux restarts.** tmux kill and new-session run as two separate ssh commands, because chaining them can take the tmux server down.
- **Secrets.** Secrets never go on a remote command line (visible in `ps`). `.zelda_secrets` is read by the generated launch script.
- **`pkill -f` over ssh.** It matches its own `bash -c` parent. The script uses the pattern `/workspace/dolphin/install/bin/[d]olphin-emu`, which does not match that parent.

## Alternative: prebuilt image (docker/Dockerfile)

```sh
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t zelda-rl:2603a .
```

- It is a two-stage build. The first stage runs the same `scripts/pod_build_dolphin.sh` with prefix `/opt/dolphin`. The runtime stage has only the runtime libraries: Qt6 + offscreen plugin, xvfb, and a `/opt/venv` with CPU torch. It also runs `pod_build_dolphin.sh --check`, which fails the image build on a missing shared library.
- It sets `DOLPHIN_BIN=/opt/dolphin/install/bin/dolphin-emu` and `QT_QPA_PLATFORM=offscreen`.
- It starts sshd with RunPod's `PUBLIC_KEY` convention.
- It contains **no game assets or secrets**; those still arrive via `--assets` / `--train`.
- Pushing it to a registry and making a RunPod template is up to you. With the image, `REMOTE_DOLPHIN_BIN`/`REMOTE_VENV` in `runpod.sh` would point at `/opt/...`; that switch is not wired yet.
- Local runs need `--shm-size` (e.g. `docker run --shm-size=4g ...`).
- On an Apple Silicon Mac, build `--platform linux/arm64` to smoke-test at native speed. amd64 under emulation is far too slow.
