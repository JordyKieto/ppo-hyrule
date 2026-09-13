"""Live process-memory reads on macOS (needs target signed with get-task-allow)."""
from __future__ import annotations

import ctypes
import ctypes.util

_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_self = ctypes.c_uint.in_dll(_libc, "mach_task_self_")
VM_REGION_BASIC_INFO_64 = 9


class _BasicInfo(ctypes.Structure):
    _fields_ = [("protection", ctypes.c_int), ("max_protection", ctypes.c_int),
                ("inheritance", ctypes.c_uint), ("shared", ctypes.c_uint), ("reserved", ctypes.c_uint),
                ("offset", ctypes.c_ulonglong), ("behavior", ctypes.c_int), ("user_wired_count", ctypes.c_ushort)]


class MachTask:
    def __init__(self, pid: int):
        self.task = ctypes.c_uint()
        kr = _libc.task_for_pid(_self, pid, ctypes.byref(self.task))
        if kr != 0:
            raise PermissionError(f"task_for_pid({pid}) failed kr={kr}")

    def regions(self):
        addr = ctypes.c_ulonglong(0)
        while True:
            size = ctypes.c_ulonglong(0)
            info = _BasicInfo()
            count = ctypes.c_uint(ctypes.sizeof(_BasicInfo) // 4)
            obj = ctypes.c_uint()
            kr = _libc.mach_vm_region(self.task, ctypes.byref(addr), ctypes.byref(size), VM_REGION_BASIC_INFO_64,
                                      ctypes.byref(info), ctypes.byref(count), ctypes.byref(obj))
            if kr != 0:
                return
            yield addr.value, size.value, info.protection
            addr.value += size.value

    def read_into(self, addr: int, buf) -> int:
        out = ctypes.c_ulonglong(0)
        kr = _libc.mach_vm_read_overwrite(self.task, ctypes.c_ulonglong(addr), ctypes.c_ulonglong(len(buf)),
                                          ctypes.c_ulonglong(ctypes.addressof(ctypes.c_char.from_buffer(buf))),
                                          ctypes.byref(out))
        if kr != 0:
            raise OSError(f"mach_vm_read_overwrite({addr:#x}, {len(buf)}) kr={kr}")
        return out.value

    def read(self, addr: int, n: int) -> bytes:
        buf = bytearray(n)
        self.read_into(addr, buf)
        return bytes(buf)


VM_FLAGS_ANYWHERE = 0x1
VM_INHERIT_NONE = 2


def remap(task: MachTask, addr: int, size: int):
    """Zero-copy: map `size` bytes of the target's memory at `addr` into this process.

    Returns a writable ctypes char array aliasing the target's pages (reads cost no syscalls).
    """
    target = ctypes.c_ulonglong(0)
    cur, mx = ctypes.c_int(0), ctypes.c_int(0)
    page = 0x4000
    start = addr & ~(page - 1)
    length = (addr + size - start + page - 1) & ~(page - 1)
    kr = _libc.mach_vm_remap(_self, ctypes.byref(target), ctypes.c_ulonglong(length), ctypes.c_ulonglong(0),
                             VM_FLAGS_ANYWHERE, task.task, ctypes.c_ulonglong(start), 0,
                             ctypes.byref(cur), ctypes.byref(mx), VM_INHERIT_NONE)
    if kr != 0:
        raise OSError(f"mach_vm_remap({addr:#x}, {size:#x}) kr={kr}")
    return (ctypes.c_char * size).from_address(target.value + (addr - start))


N64_MAGIC = bytes.fromhex("3c1a8000275a260003400008")
VC_N64_RAM_IN_MEM1 = 0xF64120   # OoT NACE: N64 RAM at Wii 0x80F64120


def find_mem1_views(task: MachTask, n64_off: int = VC_N64_RAM_IN_MEM1, game_code: bytes = b"NACE") -> list[int]:
    """Host addresses where Dolphin maps Wii MEM1 (fastmem arena + views)."""
    out = []
    for addr, size, prot in task.regions():
        if size < n64_off + 12 or not prot & 1:
            continue
        try:
            if task.read(addr + n64_off, 12) == N64_MAGIC and task.read(addr + 0x3180, 4) == game_code:
                out.append(addr)
        except OSError:
            pass
    return out
