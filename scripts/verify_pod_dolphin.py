"""First-pod acceptance test: does headless Linux Dolphin run OoT correctly for training?

Checks the assumptions docs/RUNPOD.md flags as unverified, in order, and stops at the first failure:
  1. environment: DOLPHIN_BIN exists, /dev/shm is big enough, ptrace_scope, Qt platform
  2. Dolphin boots the NAND title with the requested video backend (default Null) and the N64 RAM
     can be attached (zelda_rl.linuxmem; prints which backend: shm zero-copy or vm_readv copy)
  3. emulation advances (VI counter rises)
  4. savestate slot 1 loads through the hotkey pipe: RAM scene + Link position match the .s01 file
     (proves the savestate version is compatible and DolphinQt hotkeys work without a display)
  5. stick input through the pad pipe moves Link (his RAM position changes)
  6. a second load puts Link back at the saved position (resets are repeatable)
  7. with --instances K: K instances side by side, VI rate of each (quick concurrency probe)

    QT_QPA_PLATFORM=offscreen DOLPHIN_BIN=/workspace/dolphin/install/bin/dolphin-emu \
      python scripts/verify_pod_dolphin.py --states-dir assets/states [--backend Null] [--instances 4 --unlimited]

Exit code 0 = everything passed. Works on macOS too (uses zelda_rl.live there), which makes it easy to
compare with the Mac numbers.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import DOLPHIN_BIN, HOST_USER_DIR, HOTKEY_MAP, DolphinInstance, build_user_dir  # noqa: E402
from zelda_rl.savestate import N64Ram, decompress_state, find_n64_ram  # noqa: E402

SAVE_CTX, PLAY_STATE, VI = 0x8011AC80, 0x801C8D60, 0x8011DBB0


def attach(pid: int, timeout: float):
    if sys.platform == "linux":
        from zelda_rl.linuxmem import attach_n64_ram_linux, backend_of
        ram = attach_n64_ram_linux(pid, timeout)
        return ram, backend_of(ram)
    from zelda_rl.live import attach_n64_ram
    return attach_n64_ram(pid, timeout), "mach remap (zero-copy)"


class Probe:
    def __init__(self, ram):
        self.ram, self._refresh = ram, getattr(ram, "refresh", None)

    def _r(self):
        if self._refresh:
            self._refresh()
        return self.ram

    def vi(self): return struct.unpack_from(">I", self._r(), VI & 0x3FFFFF)[0]
    def scene(self): return struct.unpack_from(">h", self._r(), (PLAY_STATE + 0xA4) & 0x3FFFFF)[0]

    def link(self):
        r = self._r()
        p = struct.unpack_from(">I", r, (PLAY_STATE + 0x1C44) & 0x3FFFFF)[0]
        return struct.unpack_from(">fff", r, (p & 0x3FFFFF) + 0x24) if 0x80000000 <= p < 0x80400000 else None


def expected_from_state(path: Path):
    raw = decompress_state(path.read_bytes())
    off = find_n64_ram(raw)
    r = N64Ram(raw[off:off + 0x400000])
    p = r.u32(PLAY_STATE + 0x1C44)
    pos = r.vec3(p + 0x24) if 0x80000000 <= p < 0x80400000 else None
    return r.s16(PLAY_STATE + 0xA4), pos


def near(a, b, tol):
    return a is not None and b is not None and math.dist(a, b) <= tol


def step(ok: bool, msg: str):
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        raise SystemExit(1)


def env_report():
    print(f"DOLPHIN_BIN={DOLPHIN_BIN}  HOST_USER_DIR={HOST_USER_DIR}  QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM')}"
          f"  DISPLAY={os.environ.get('DISPLAY')}  cpus={os.cpu_count()}")
    step(Path(DOLPHIN_BIN).exists(), f"Dolphin binary exists: {DOLPHIN_BIN}")
    step((HOST_USER_DIR / "Wii").is_dir(), f"NAND present at {HOST_USER_DIR / 'Wii'}")
    if sys.platform == "linux":
        st = os.statvfs("/dev/shm")
        free_mb = st.f_bavail * st.f_frsize / 2**20
        print(f"/dev/shm free: {free_mb:.0f} MB (Dolphin needs ~97 MB per instance, else SIGBUS)")
        try:
            print("ptrace_scope:", Path("/proc/sys/kernel/yama/ptrace_scope").read_text().strip())
        except OSError:
            print("ptrace_scope: (no yama)")
        return free_mb
    return float("inf")


def load_slot(inst, probe: Probe, want_scene: int, want_pos, timeout: float = 20.0) -> float | None:
    """Closed-loop reset: hold the load hotkey until RAM matches the savestate. Returns seconds or None."""
    key = HOTKEY_MAP["Load State/Load State Slot 1"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        inst.hk.send(f"PRESS {key}")
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < 0.25:
            if probe.scene() == want_scene and near(probe.link(), want_pos, 5.0):
                inst.hk.send(f"RELEASE {key}")
                return time.perf_counter() - t0
            time.sleep(0.002)
        inst.hk.send(f"RELEASE {key}")
        time.sleep(0.25)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states-dir", default="assets/states")
    ap.add_argument("--slot-file", default="NACE01.s01")
    ap.add_argument("--backend", default="Null")
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--unlimited", action="store_true", help="unlimited emulation speed (training setting)")
    ap.add_argument("--work-dir", default="runs/verify_pod")
    ap.add_argument("--boot-timeout", type=float, default=90.0)
    args = ap.parse_args()

    free_mb = env_report()
    if free_mb < 100 * args.instances:
        print(f"WARNING: /dev/shm has {free_mb:.0f} MB for {args.instances} instance(s); expect SIGBUS crashes")

    state = Path(args.states_dir) / args.slot_file
    step(state.exists(), f"savestate exists: {state}")
    want_scene, want_pos = expected_from_state(state)
    print(f"savestate says: scene={want_scene} link_pos={want_pos}")

    work = Path(args.work_dir)
    shutil.rmtree(work, ignore_errors=True)
    insts = []
    try:
        for i in range(args.instances):
            ud = build_user_dir((work / f"i{i}").resolve(), dump_frames=False, unlimited_speed=args.unlimited)
            shutil.copy(state, ud / "StateSaves" / "NACE01.s01")
            insts.append(DolphinInstance(ud, video_backend=args.backend))
        t0 = time.time()
        inst = insts[0]
        try:
            ram, how = attach(inst.proc.pid, args.boot_timeout)
        except Exception as e:  # noqa: BLE001
            print((inst.user_dir / "dolphin_stdout.log").read_text(errors="replace")[-3000:])
            step(False, f"attach N64 RAM: {e!r} (Dolphin exit code {inst.proc.poll()})")
        step(True, f"booted + attached N64 RAM in {time.time() - t0:.1f}s via {how}")
        probe = Probe(ram)

        v0 = probe.vi(); time.sleep(2.0); v1 = probe.vi()
        step(v1 > v0, f"emulation advances under -v {args.backend}: VI {v0} -> {v1} ({(v1 - v0) / 2:.0f}/s)")

        dt = load_slot(inst, probe, want_scene, want_pos)
        step(dt is not None, f"savestate load via hotkey pipe matches file (scene {probe.scene()}, link {probe.link()})"
             + (f" in {dt * 1e3:.0f} ms" if dt else ""))

        p0 = probe.link()
        inst.pad.send("SET MAIN 0.5 1.0"); time.sleep(1.0); inst.pad.send("SET MAIN 0.5 0.5")
        time.sleep(0.2)
        p1 = probe.link()
        moved = math.dist(p0, p1) if p0 and p1 else 0.0
        step(moved > 20.0, f"stick input moves Link: {p0} -> {p1} ({moved:.0f} units)")

        dt2 = load_slot(inst, probe, want_scene, want_pos)
        step(dt2 is not None, "second load returns Link to the saved position" + (f" in {dt2 * 1e3:.0f} ms" if dt2 else ""))

        if len(insts) > 1:
            probes = []
            for j, other in enumerate(insts):
                r, _ = (ram, how) if j == 0 else attach(other.proc.pid, args.boot_timeout)
                probes.append(Probe(r))
                if j > 0 and load_slot(other, probes[-1], want_scene, want_pos) is None:
                    print(f"WARNING: instance {j} did not load the savestate; its VI rate is from menus/attract demo")
            v0s = [p.vi() for p in probes]; time.sleep(5.0); v1s = [p.vi() for p in probes]
            rates = [(b - a) / 5.0 for a, b in zip(v0s, v1s)]
            print("VI/s per instance:", " ".join(f"{r:.0f}" for r in rates), f"| total {sum(rates):.0f} VI/s "
                  f"({sum(rates) / 60:.1f}x realtime-equivalents)")
            try:
                print(subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip())
            except OSError:
                pass
        print("ALL CHECKS PASSED")
    finally:
        for i in insts:
            i.stop()


if __name__ == "__main__":
    main()
