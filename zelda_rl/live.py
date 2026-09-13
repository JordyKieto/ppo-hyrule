"""Live RAM attachment to a running Dolphin instance (macOS: mach; Linux: zelda_rl.linuxmem)."""
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
    raise TimeoutError("N64 RAM not found in Dolphin process")


def be_u32(ram, addr): return struct.unpack_from(">I", ram, addr & 0x3FFFFF)[0]
def be_s16(ram, addr): return struct.unpack_from(">h", ram, addr & 0x3FFFFF)[0]
