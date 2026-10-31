"""Live RAM attachment to a running Dolphin instance (macOS: mach; Linux: zelda_rl.linuxmem).

EXPERIMENT-CONTRACT: exempt -- attaches to a Dolphin that someone else launched; it never
opens one, and it has no entry point whose thread policy it could set.
"""
from __future__ import annotations

import struct
import sys
import time

import numpy as np

N64_RAM_SIZE = 0x400000


def attach_n64_ram(pid: int, timeout: float = 60.0) -> np.ndarray:
    """Wait for Dolphin to map MEM1, then return a zero-copy uint8 view of the 4 MB N64 RAM."""
    if sys.platform.startswith("linux"):
        from zelda_rl.linuxmem import attach_n64_ram_linux
        return attach_n64_ram_linux(pid, timeout)

    from zelda_rl.machmem import MachTask, find_mem1_views, remap

    t0 = time.time()
    task = None
    while time.time() - t0 < timeout:
        try:
            task = task or MachTask(pid)
            views = find_mem1_views(task)
            if views:
                return np.frombuffer(remap(task, views[0] + 0xF64120, N64_RAM_SIZE), np.uint8)
        except PermissionError:
            pass
        time.sleep(0.25)
    # Naming the emulator here cost ~40 minutes of a hands-on session: the window, the controller, the NAND
    # and a refactor were each blamed before docs/MAP.md lines 66 and 73 were re-read. Both causes below
    # present IDENTICALLY as this timeout, so the message lists them (AGENTS 21, 23).
    import os
    hint = ""
    if sys.platform == "darwin":
        from pathlib import Path as _P
        why = []
        if _P(sys.executable).name != "python-dbg":
            why.append(f"interpreter is {_P(sys.executable).name}, not .venv/bin/python-dbg -- task_for_pid "
                       f"needs the signed one (docs/MAP.md line 66)")
        if "/.cache/zelda-rl/" not in os.environ.get("DOLPHIN_BIN", ""):
            why.append("DOLPHIN_BIN is not the cached build -- /Applications/Dolphin.app boots the Wii System "
                       "Menu instead of the game and refuses memory reads "
                       "(export DOLPHIN_BIN=~/.cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin, "
                       "docs/MAP.md line 73)")
        hint = "".join(f"\n  - {w}" for w in why) or "\n  - both prerequisites look right; check the Dolphin window"
    raise TimeoutError(f"N64 RAM not found in Dolphin process after {timeout:.0f}s{hint}")


def be_u32(ram, addr): return struct.unpack_from(">I", ram, addr & 0x3FFFFF)[0]
def be_s16(ram, addr): return struct.unpack_from(">h", ram, addr & 0x3FFFFF)[0]
