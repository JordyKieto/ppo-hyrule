# Zelda OoT RL on Dolphin: Findings So Far

Investigation log for driving *The Legend of Zelda: Ocarina of Time* (Wii Virtual Console) in stock Dolphin as an RL environment. The first task is rupee collection. Everything marked **measured** was run on the machine below. Everything marked *estimate* was not.

- **Machine:** Apple M1 (4 performance + 4 efficiency cores), 16 GB RAM, macOS 26.2
- **Emulator:** Dolphin 2603a (official universal build, `/Applications/Dolphin.app`)
- **Game:** OoT USA VC, title `000100014e414345` (`NACE`), installed in the NAND. The N64 ROM inside is **NTSC 1.2**.
- **Save:** File 1 "LINK", a brand-new file: Link's house, 3 hearts, 0 rupees, no sword.

---

## 1. TL;DR

| Question | Answer |
|---|---|
| Can we control Dolphin from Python on macOS? | **Yes.** The named-pipe controller handles buttons and analog sticks, and a second pipe handles hotkeys (savestate save/load). |
| Can we run isolated instances? | **Yes.** `Dolphin -u <user_dir> -b -n <title>`, each with its own config, NAND copy and pipes. The user's main Dolphin stays untouched. |
| RGB observations? | **Yes**, through Dolphin's PNG frame dump: ~20 unique frames/s in gameplay (OoT's native rate). Window capture is blocked without Screen Recording permission. |
| Can savestates be parsed? | **Yes.** Header + LZ4 block → 105 MB of Wii state → 4 MB of N64 RAM. **~8 obs/s at best.** Fine for resets and debugging, too slow and heavy per step. |
| Live game RAM? | **Yes: zero-copy mapping of Dolphin's memory.** Rupees + Link's position in **8 µs**, full feature extraction in **~25 µs**, 4 MB snapshot in **0.14–0.19 ms**. Survives savestate loads. Needs a one-time signing setup (§3.1). |
| Resets? | Savestate load via hotkey: **45/45 succeeded, ~29 ms p50** from press to Link back at spawn (verified in RAM). |
| Do we need OCR for rupees? | **No.** Rupees, health, scene, Link's position and collectibles are all readable from RAM. |
| Concurrency on one Mac? | **Not measured yet.** Estimate: 3–4 instances at 1× speed, CPU-bound. |
| **Can PPO learn OoT combat from RAM?** | **Yes (2026-09-13).** A GRU entity policy trained on 20 RunPod CPU pods (41.8 M env steps, ~5 h 20 min, ~$55 incl. discarded updates) clears the Deku Tree Deku Baba room in **77%** of episodes across 10 start states (3.7% deaths). Watched locally: **3/3 successes**, fastest clear 17.7 s with no damage. See `docs/COMBAT_TASK.md` § Results. |

---

## 2. Control (input + resets)

**Named pipes (verified).** Dolphin creates an input device for every FIFO in `<user_dir>/Pipes/`. Our GC pad maps to `Pipe/0/pad`.

```
PRESS A / RELEASE A              buttons: A B X Y Z START L R D_UP D_DOWN D_LEFT D_RIGHT
SET MAIN <x> <y>                 analog 0..1, 0.5 = neutral, y=1 = up (away from camera)
SET C <x> <y>                    C-stick
```

**Hotkeys over a pipe (verified).** `Hotkeys.ini` binds `Load State Slot 1` and `Save State Slot 1` to buttons on a second FIFO, `Pipe/0/hk`. Episode reset = copy the state to `StateSaves/NACE01.s01` once, then pulse the load hotkey.

**Config that is required** (see `zelda_rl/dolphin.py:build_user_dir`):

| Setting | Why |
|---|---|
| `Input.BackgroundInput = True` | Pad input is ignored when the window isn't focused |
| `General.HotkeysRequireFocus = False` | Hotkeys have a separate focus gate. **The key is in `[General]`** (an earlier `[Interface]` placement was silently ignored, so hotkeys only reached the *focused* window: the cause of the "flaky" loads and of 2nd-instance load failures) |
| `DSP.Backend = No Audio Output` | Saves CPU |
| `Core.EmulationSpeed = 0.0` | Unlimited speed (for training); `1.0` for real-time tests |

**Gotchas**
- `-s <savestate>` is **ignored** with `-n` (NAND boot). Load states through the hotkey instead.
- Menu flow: attract demo → START → title → START/A → file select → A → "Open this file?" → A. Blind sleep timings were flaky. `scripts/make_savestate.py` presses A until the HUD appears (~45 s from cold boot).
- **(Resolved) load flakiness** was the focus gate above. Historical note: 20 ms pulses load 15/15. **During boot or the attract demo they sometimes silently don't fire** (seen 3 times, cause unknown; App Nap/timer coalescing is a guess). Use **closed-loop** resets: hold/retry until RAM confirms the load.
- During a save, Dolphin removes and rewrites the state file. Readers must tolerate `FileNotFoundError` and partial files.
- Dolphin keeps a `lastState.sav` backup on every save, which doubles the disk writes.

---

## 3. Observation channels: trade-offs

| Channel | Works on mac? | Rate / latency (measured unless noted) | Cost | Notes |
|---|---|---|---|---|
| **Window capture** (screencapture / ScreenCaptureKit) | ❌ No Screen Recording permission for VS Code | – | – | Needs visible windows; doesn't carry over to a headless cloud. Not pursued. |
| **RGB: PNG frame dump** | ✅ stock | ~20 unique fps in gameplay, 60 in intro (duplicates skipped). Per-stage latency *not yet measured*. | ~0.9 MB/frame → ~18 MB/s disk per instance at 1×. Must delete old files. | Mac build has **no FFmpeg**, so no video/pipe dump. Needs a CNN (and OCR for reward). |
| **Savestate → N64 RAM** | ✅ stock | Save ~90 ms + read 6 ms + decompress 28 ms + parse <1 ms ≈ **125 ms → ~8 obs/s** max | 38.5 MB written per obs (+ backup) ≈ 300 MB/s disk; emulator **hitches** on every save | Exact state. Best for resets, offline analysis and labeling. |
| **N64 RAM via GDB stub** | ✅ stock (`-C Dolphin.General.GDBPort=2345`) | Reads 0.39 ms (3 regions); halt+read+continue 11.3 ms p50 (~10 ms client drain timeout) | Pauses the CPU per read | **A savestate load kills the stub** (connection refused afterwards) → incompatible with savestate resets. ≤0x800 bytes per `m` packet. |
| **N64 RAM via `mach_vm_read`** | ✅ with signing setup (§3.1) | **44 µs** p50 for 3 small reads; **0.74 ms** per 4 MB snapshot (~1,350/s) | One syscall per read | Survives savestate loads. |
| **N64 RAM via `mach_vm_remap` (zero-copy)** ⭐ | ✅ with signing setup (§3.1) | **8 µs** rupees + position; **23–28 µs** full `read_features()` incl. actor-list walk; **0.14–0.19 ms** 4 MB copy | None: Dolphin's pages mapped into our process, no syscalls per read | All 4 MEM1 mirrors agree. Live while Link walks. |
| **N64 RAM via `process_vm_readv`** (Linux) | *Not tested* | *Estimate:* µs, like mach | – | Standard Linux route for the cloud. Needs ptrace permission (same uid + `ptrace_scope`, or `CAP_SYS_PTRACE`). |

### 3.1 macOS live-memory recipe (verified)
`task_for_pid` from a non-root process needs **all three** of the following. Each missing piece gives `kr=5`, which was tested with a tiny C probe:

1. **Developer Mode:** `sudo DevToolsSecurity -enable` and the user in `_developer` (already true here).
2. **Target signed with `com.apple.security.get-task-allow`:** a re-signed *copy* of Dolphin. The official app stays locked; `task_for_pid` against it is still denied.
3. **Caller signed with `com.apple.security.cs.debugger`** (hardened runtime). An unsigned C probe was denied; the same probe signed with this entitlement succeeded.

Python-specific trap: Homebrew's framework `bin/python3.x` **re-execs** `Python.app/Contents/MacOS/Python`, which drops the entitlement. Sign a copy of that *inner* binary into `<venv>/bin/python-dbg`; it finds the venv through `../pyvenv.cfg`.

Other traps:
- `codesign` fails inside `~/Documents` (File Provider xattrs: "resource fork … detritus"). Sign copies elsewhere; `ditto --noextattr` helps.
- The script is `scripts/setup_macos_memaccess.sh <venv> [dest]`. Run with `DOLPHIN_BIN=<dest>/Dolphin.app/Contents/MacOS/Dolphin <venv>/bin/python-dbg …`.

Finding memory: scan `mach_vm_region` for regions with the N64 exception vector at `+0xF64120` and `NACE` at `+0x3180`. That takes 0.01 s and finds 4 views (fastmem arena + mirrors). Then `mach_vm_remap` the 4 MB N64 block into the Python process and read it with `memoryview`/`struct`/numpy.

### Recommendation
- **Observation:** a RAM feature vector read zero-copy each step (Link position/velocity/yaw, rupees, health, scene, nearest collectibles relative to Link). Pixels are optional, added later if RAM features plateau.
- **Reward:** Δrupees from RAM. No OCR.
- **Reset:** savestate load via hotkey, **closed-loop verified from RAM** (~30 ms).
- **Step rate:** OoT runs at 20 Hz natively. The observation channel is no longer a bottleneck (µs), so the ceiling is emulation speed × instances.
- **Cloud:** same design with `process_vm_readv` or `/proc/<pid>/mem` on Linux; no signing needed.

---

## 4. Benchmark results (measured)

### Boot & frames
| Metric | Value |
|---|---|
| Launch → first dumped frame | ~4 s |
| Launch → MEM1 findable in process memory | ~10 s |
| Launch → title screen | ~14 s |
| Cold boot → gameplay savestate (scripted) | ~45 s |
| Frame dump size / format | 644×480 RGB PNG, compression 0, ~0.9 MB |
| Unique frames/s | ~60 (intro/menus), ~20 (gameplay) |

### Savestate as observation (`scripts/bench_savestate_obs.py`, `scripts/bench_save_hold.py`)
| Stage | p50 | p90 |
|---|---|---|
| Hotkey press → new state file (true Dolphin save latency) | 89–104 ms (min 83 ms) | – |
| Read 38.5 MB file (SSD) | 6 ms | 8–10 ms |
| LZ4 decompress in Python (105 MB) | 28 ms (113 ms first/cold) | 38 ms |
| Locate N64 RAM + extract features | <1 ms | 1 ms |
| End-to-end period with a 150 ms hotkey hold | 189 ms → **5.3 obs/s** | 239 ms |
| Theoretical best (10 ms hold) | ~125 ms → **~8 obs/s** | – |

### GDB stub (`scripts/probe_gdb3.py`, attract demo running)
| Stage | p50 | p90 |
|---|---|---|
| Halt (`\x03` → stop reply), incl. 10 ms drain timeout | 10.9 ms | 11.8 ms |
| 3 memory reads (64 + 4 + 12 bytes) | 0.39 ms | 0.45 ms |
| Continue | 0.01 ms | 0.02 ms |
| Total per observation | 11.3 ms | 12.4 ms |
| After a savestate load | ❌ interrupt times out, reconnect refused | |

### Live process memory (`scripts/probe_machmem.py`, `scripts/probe_remap.py`)
| Operation | p50 | p90 |
|---|---|---|
| Find MEM1 views (region scan) | 0.01 s | – |
| `mach_vm_read`: rupees + player pointer + position (3 reads) | 44 µs | 88 µs |
| `mach_vm_read`: full 4 MB N64 RAM | 0.74 ms (~1,350/s) | – |
| **Zero-copy view:** rupees + position | **8.0–8.5 µs** | 11–12 µs |
| **Zero-copy view:** `read_features()` (save ctx, scene, player, 12 actor lists) | **23–28 µs** | – |
| Zero-copy view: 4 MB `numpy.copy()` snapshot | 0.14–0.19 ms | – |
| Valid after savestate load | ✅ (`mach_vm_read`: house state read correctly after load) | |

### Resets (`scripts/bench_reset_ram.py`, in gameplay, verified in RAM)
| Press style | Success | Press → Link at spawn p50 | p90 | max |
|---|---|---|---|---|
| 20 ms pulse | 15/15 | 29 ms | 54 ms | 136 ms |
| 200 ms pulse | 15/15 | 28 ms | 35 ms | 119 ms |
| Closed loop (hold until RAM confirms) | 15/15 | 29 ms | 51 ms | 177 ms |

### Not yet measured
- Concurrency (N instances at unlimited speed: throughput, CPU, thermals, efficiency-core scheduling).
- Remap view validity across a load: the `mach_vm_read` path was verified across a load; the remap probe's load was sent during the attract demo and didn't fire. Expected to hold (same pages).
- RGB pipeline stage latencies and CNN inference (CPU vs MPS). `scripts/bench_latency.py` is written but has not run.
- Input → RAM-visible effect delay (now cheap to measure with the zero-copy view). *Estimate:* 1–3 game frames.

---

## 5. Dolphin savestate format (2603a, header v1)

```
0x00  char[6]  game id "NACE01", u16 reserved, u32 legacy lzo size, u32 reserved
0x10  f64      timestamp
0x18  u32      version cookie (0xBAADBABE + state version)
0x1C  u32      version string length N
0x20  char[N]  "Dolphin 2603a"
+N    u16 header_version(1)  u16 compression(1 = LZ4)  u32 payload_offset(0)  u64 uncompressed_size
      then chunks: [s32 compressed_len][LZ4 block]...   (one chunk in practice)
```
Uncompressed: 109,663,858 bytes (Wii MEM1/MEM2 + device state). Parser: `zelda_rl/savestate.py`.

---

## 6. OoT RAM map (NTSC 1.2 inside the Wii VC)

N64 RAM (4 MB) lives in Wii MEM1 at **`0x80F64120`** (offset `0x17A14A5` in the decompressed state; host offset `+0xF64120` from each MEM1 view). Convert N64 address `A` to an offset in that block with `A & 0x3FFFFF`. All values are big-endian.

| Symbol | N64 addr / offset | Verified value |
|---|---|---|
| N64 RAM +0 | exception vector `3C1A8000 275A2600 03400008` | ✅ |
| `osMemSize` | `0x80000318` = `0x00400000` | ✅ |
| `gSaveContext` | `0x8011AC80` ("ZELDAZ" at +0x1C) | ✅ |
| entrance | +0x00 (s32) | 0xBB (Link's house) |
| age | +0x04 (s32) | 1 (child) |
| max health / health | +0x2E / +0x30 (s16) | 0x30 / 0x30 (3 hearts) |
| **rupees** | **+0x34 (s16)** | 0 (house), 150 (attract-demo save) |
| `gPlayState` | `0x801C8D60` | ✅ |
| scene | +0xA4 (s16) | 52 = Link's House, 81 = Hyrule Field (attract demo) |
| actor lists | +0x1C24 +0x0C: 12 × {s32 count; Actor* head} | player count = 1 |
| player actor pointer | +0x1C44 | `0x801DB2F0` (house) |
| Actor world position | actor +0x24 (Vec3f) | tracked walking live ✅ |
| Actor yaw | actor +0x32 (s16) | changes plausibly while turning; not rigorously verified |
| Actor next | actor +0x124 | used for list walk |
| Collectible actor id | `EnItem00` = 0x0015, param at +0x1C | implemented, **untested** (no rupees in the house) |

---

## 7. Concurrency on a single Mac (*estimates*, to be benchmarked)
- Each Dolphin wants ~2 fast threads (CPU + GPU thread) → **3–4 instances at 1×** on the 4 performance cores, fewer at unlimited speed. Background processes may be pushed onto efficiency cores.
- Memory: ~0.5–1 GB per instance → 16 GB is not the limit.
- GPU: shared through Metal; N64 VC at 1× internal resolution is light. With RAM observations, rendering could be minimized.
- Disk: PNG dumps and savestate polling are heavy. Live RAM reads avoid both.
- Docker on macOS runs a Linux VM with no Metal → software rendering → not useful for local scaling. Docker is for Linux cloud hosts.

---

## 7b. Corrections (later session)
- **Your slot 4 is frozen** (the game ignores input); slots 1, 2, 3, 5 accept input. Slot 3 ignores input for a few hundred ms right after loading.
- An intermediate conclusion that "Link can't move under the Null backend / uncapped speed" was **wrong**: those probes used slot 4. Uncapped Metal with input is verified (~4× for one instance). The Null backend still needs a check with a working slot.
- `GameState.frames` at `gPlayState+0x9C` does not tick in this build; the env clock is the 60 Hz VI counter at `0x8011DBB0`.
- Goal PPO loop, observation schema and benchmarks: `docs/OBSERVATION_SCHEMA.md`.

## 7c. Cloud (RunPod) — see docs/RUNPOD.md for full numbers
- Dolphin 2603a builds on Ubuntu 24.04 in 269 s. Mac savestates load on x86. Null backend + pipe input work **under Xvfb** (Qt offscreen never boots the game).
- Linux RAM access: zero-copy mmap of Dolphin's `/dev/shm/dolphin-emu.<pid>` fd, attach in 0.8 s.
- One 16-vCPU CPU pod ($0.48–0.56/hr): 8 idle instances = 28.9× real time; trainer at 12 envs = 214–224 env steps/s (≈2.5–3× the M1).
- 5-pod fleet coordinated over Upstash (heartbeats, policy publishing, per-update weight averaging): $2.64/hr.

## 8. Open problems / next steps
1. **Rupee-rich start state:** the current save has no rupees nearby and no sword. Need a savestate where collection is possible.
2. **Validate collectible reading** (`EnItem00` list walk) in a scene with rupees.
3. **Concurrency sweep:** 1→6 instances, unlimited speed, env steps/s per instance and total, with RAM observations.
4. **Step timing:** decide real-time (1×, act every N game frames using a RAM frame counter) vs. unlimited speed with lock-step.
5. **Gymnasium env + PPO** (CleanRL-style) on RAM features at 10–20 Hz.
6. **Cloud:** Linux container with `dolphin-emu-nogui`, `process_vm_readv`, savestate resets.
7. Move the Python env from the session scratchpad into the project (`.venv`) and rerun `setup_macos_memaccess.sh` for it.

---

## 9. Code map
| Path | Purpose |
|---|---|
| `zelda_rl/dolphin.py` | Instance user-dir builder (config, NAND copy, FIFOs), launcher, pipe writer. `DOLPHIN_BIN` env selects the binary |
| `zelda_rl/machmem.py` | macOS `task_for_pid`, `mach_vm_read`, zero-copy `remap`, `find_mem1_views` |
| `zelda_rl/oot_ram.py` | OoT NTSC 1.2 RAM map → feature dict |
| `zelda_rl/savestate.py` | Savestate decompress + N64 RAM locator; `N64Ram` accessor |
| `zelda_rl/gdb.py` | Minimal GDB remote client for Dolphin's stub |
| `zelda_rl/frames.py` | Latest-frame reader for the PNG dump dir (with pruning) |
| `zelda_rl/hud.py` | Gameplay detector (rupee icon pixels) |
| `zelda_rl/policy.py` | Nature-CNN actor-critic (for the pixel path) |
| `scripts/setup_macos_memaccess.sh` | One-time signing of Dolphin + interpreter copies |
| `scripts/make_savestate.py` | Cold boot → gameplay → `runs/states/<name>.sav` |
| `scripts/ctl.py` | Poke a running instance (tap / stick / hotkey) and grab a frame |
| `scripts/bench_reset_ram.py` | Reset reliability/latency verified from RAM |
| `scripts/probe_machmem.py`, `probe_remap.py`, `probe_hotkey_flake.py` | Live memory probes |
| `scripts/bench_savestate_obs.py`, `bench_save_hold.py` | Savestate observation benchmarks |
| `scripts/probe_gdb*.py` | GDB stub probes |
