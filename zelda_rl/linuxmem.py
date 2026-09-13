"""Live N64 RAM attachment to a running Dolphin instance on Linux (RunPod / Docker backend).

Same contract as ``zelda_rl.live.attach_n64_ram`` on macOS: ``attach_n64_ram_linux(pid, timeout)`` returns
a uint8 ``np.ndarray`` of length 0x400000 aliasing the emulated N64 RAM of OoT (Wii VC ``NACE``).

Backends, in order of preference:

1. ``shm`` (zero-copy, no ptrace). Dolphin's Linux ``MemArena`` creates all emulated physical memory as
   one POSIX shm object and unlinks it immediately, but keeps the fd open. Verified against the
   Dolphin source at tag ``2603a`` (commit 5e7cc91d8c9a43ca189b288937f65c9763af9c22):

   - ``Source/Core/Common/CMakeLists.txt``: non-Apple, non-Android, non-Windows builds compile
     ``MemArenaUnix.cpp`` (macOS uses ``MemArenaDarwin.cpp``, which is why ``machmem.py`` differs).
   - ``Common/MemArenaUnix.cpp`` ``GrabSHMSegment``: ``shm_open("/<base_name>.<getpid()>", O_RDWR|O_CREAT|O_EXCL, 0600)``,
     then ``shm_unlink`` and ``ftruncate(size)``. ``m_shm_fd`` stays open until ``ReleaseSHMSegment``
     (emulation shutdown). Views are ``mmap(..., MAP_SHARED, m_shm_fd, offset)``.
   - ``Core/HW/Memmap.cpp`` ``MemoryManager::Init``: ``m_arena.GrabSHMSegment(mem_size, "dolphin-emu")``.
     Physical regions are packed in order RAM (MEM1), L1 cache, fake VMEM (GameCube without MMU
     only), EXRAM (MEM2, Wii only), so **MEM1 is at shm offset 0** on Wii. Sizes are rounded up to a
     power of two: MEM1 0x01800000 -> 0x02000000, L1 cache 0x40000, MEM2 0x04000000, so the whole
     segment is 0x06040000 bytes (~96 MB) per instance.

   With glibc, ``shm_open("/dolphin-emu.<pid>")`` lives at ``/dev/shm/dolphin-emu.<pid>``, so the fd
   shows up as ``/proc/<pid>/fd/N -> /dev/shm/dolphin-emu.<pid> (deleted)``. Opening that magic link
   re-opens the unlinked inode. procfs gates this with ``ptrace_may_access(PTRACE_MODE_READ_FSCREDS)``,
   which Yama's ``ptrace_scope`` does not restrict (Yama only restricts ATTACH mode), so a same-uid
   process or root in the container is enough. We still verify the signatures instead of trusting
   the layout: the N64 exception vector ``3c1a8000 275a2600 03400008`` at MEM1+0xF64120 and ASCII
   ``NACE`` at MEM1+0x3180. The fast path checks offset 0 and then scans page-aligned offsets with
   ``pread``. Reads of unallocated tmpfs holes return zeros without allocating pages, unlike faulting
   through an mmap. Only the 4 MB N64 block is then mmapped.

2. ``vm_readv`` (copying fallback). This path parses ``/proc/<pid>/maps``, finds a readable mapping
   holding MEM1 (``dolphin-emu`` mappings first, then any mapping large enough), and copies the 4 MB
   with ``process_vm_readv`` into a buffer. It needs ptrace ATTACH permission: same uid, and a Yama
   ``ptrace_scope`` <= 1 with Dolphin as our descendant (true when we Popen it), or ``CAP_SYS_PTRACE``.
   It also needs a seccomp profile that allows ``process_vm_readv`` (Docker's default allows it on
   kernels >= 4.8). **Not zero-copy**: the returned array is a ``PolledN64Ram`` whose ``refresh()``
   must be called once per env step, before reading. Callers can stay backend-agnostic with
   ``refresh = getattr(ram, "refresh", None)``.

Both paths are dependency-free (numpy + stdlib + ctypes). None of this was run against a real Linux
Dolphin yet; see docs/RUNPOD.md "To verify on the first pod".
"""
from __future__ import annotations

import ctypes
import errno
import mmap
import os
import struct
import time

import numpy as np

N64_RAM_SIZE = 0x400000
N64_MAGIC = bytes.fromhex("3c1a8000275a260003400008")
VC_N64_RAM_IN_MEM1 = 0xF64120   # OoT NACE: N64 RAM at Wii 0x80F64120
GAME_CODE_OFF = 0x3180          # ASCII game code inside MEM1
SHM_BASE_NAME = "dolphin-emu"   # Memmap.cpp: m_arena.GrabSHMSegment(mem_size, "dolphin-emu")
PAGE = mmap.ALLOCATIONGRANULARITY


# ------------------------------------------------------------------------------------------------
# Backend 1: shm fd -> mmap (zero-copy)
# ------------------------------------------------------------------------------------------------
def find_dolphin_shm_fds(pid: int) -> list[tuple[str, str]]:
    """(``/proc/<pid>/fd/N``, link target) for every fd of ``pid`` that points at Dolphin's shm arena."""
    out = []
    try:
        entries = list(os.scandir(f"/proc/{pid}/fd"))
    except FileNotFoundError:
        raise ProcessLookupError(f"pid {pid} does not exist") from None
    for e in entries:
        try:
            target = os.readlink(e.path)
        except OSError:          # fd closed while scanning, or no permission
            continue
        if SHM_BASE_NAME in target:
            out.append((e.path, target))
    return out


def _is_mem1(read4, read12, base: int, n64_off: int, game_code: bytes) -> bool:
    return read4(base + GAME_CODE_OFF) == game_code and read12(base + n64_off) == N64_MAGIC


def locate_mem1_in_fd(fd: int, n64_off: int = VC_N64_RAM_IN_MEM1, game_code: bytes = b"NACE") -> int | None:
    """Offset of Wii MEM1 inside the shm file, found by signatures (offset 0 is checked first)."""
    size = os.fstat(fd).st_size
    last = size - (n64_off + len(N64_MAGIC))
    if last < 0:
        return None

    def read4(off): return os.pread(fd, 4, off)
    def read12(off): return os.pread(fd, 12, off)

    if _is_mem1(read4, read12, 0, n64_off, game_code):
        return 0
    for base in range(PAGE, last + 1, PAGE):
        if read4(base + GAME_CODE_OFF) == game_code and read12(base + n64_off) == N64_MAGIC:
            return base
    return None


def _mmap_n64_from_fd(fd: int, mem1_off: int, n64_off: int, writable: bool) -> np.ndarray:
    start = mem1_off + n64_off
    page_off = start - start % PAGE
    delta = start - page_off
    prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
    mm = mmap.mmap(fd, delta + N64_RAM_SIZE, flags=mmap.MAP_SHARED, prot=prot, offset=page_off)
    # np.frombuffer keeps `mm` alive through arr.base; mmap dup()s the fd, so the caller may close theirs.
    arr = np.frombuffer(mm, np.uint8, count=N64_RAM_SIZE, offset=delta)
    return arr


def attach_via_shm(pid: int, *, writable: bool = False, n64_off: int = VC_N64_RAM_IN_MEM1,
                   game_code: bytes = b"NACE") -> np.ndarray | None:
    """Zero-copy view, or None if the arena exists but MEM1 is not populated yet.

    Raises FileNotFoundError if no dolphin-emu shm fd is visible, PermissionError if it can't be opened.
    """
    fds = find_dolphin_shm_fds(pid)
    if not fds:
        raise FileNotFoundError(f"no '{SHM_BASE_NAME}' shm fd in /proc/{pid}/fd")
    last_err: OSError | None = None
    for path, _target in fds:
        try:
            fd = os.open(path, os.O_RDWR if writable else os.O_RDONLY)
        except OSError as e:
            last_err = e
            continue
        try:
            off = locate_mem1_in_fd(fd, n64_off, game_code)
            if off is not None:
                return _mmap_n64_from_fd(fd, off, n64_off, writable)
        finally:
            os.close(fd)
    if last_err is not None and last_err.errno in (errno.EACCES, errno.EPERM):
        raise PermissionError(f"cannot open Dolphin shm fd of pid {pid}: {last_err}")
    return None


# ------------------------------------------------------------------------------------------------
# Backend 2: /proc/<pid>/maps + process_vm_readv (copying)
# ------------------------------------------------------------------------------------------------
class _IOVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


_libc = None


def _process_vm_readv():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        fn = _libc.process_vm_readv
        fn.argtypes = [ctypes.c_int, ctypes.POINTER(_IOVec), ctypes.c_ulong,
                       ctypes.POINTER(_IOVec), ctypes.c_ulong, ctypes.c_ulong]
        fn.restype = ctypes.c_ssize_t
    return _libc.process_vm_readv


class ProcessMemReader:
    """Reads a fixed range of another process's memory with process_vm_readv."""

    def __init__(self, pid: int):
        self.pid = pid
        self._fn = _process_vm_readv()

    def read_into(self, addr: int, buf: np.ndarray) -> None:
        assert buf.flags.c_contiguous and buf.dtype == np.uint8
        local = _IOVec(buf.ctypes.data, buf.nbytes)
        remote = _IOVec(addr, buf.nbytes)
        n = self._fn(self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
        if n != buf.nbytes:
            err = ctypes.get_errno()
            if n < 0 and err in (errno.EPERM, errno.EACCES):
                raise PermissionError(f"process_vm_readv(pid={self.pid}) denied: {os.strerror(err)} "
                                      "(needs same uid + ptrace_scope<=1 as parent, or CAP_SYS_PTRACE)")
            if n < 0 and err == errno.ESRCH:
                raise ProcessLookupError(f"pid {self.pid} is gone")
            raise OSError(err, f"process_vm_readv({addr:#x}, {buf.nbytes:#x}) returned {n}: {os.strerror(err)}")

    def read(self, addr: int, n: int) -> bytes:
        buf = np.empty(n, np.uint8)
        self.read_into(addr, buf)
        return buf.tobytes()


def iter_maps(pid: int):
    """(start, end, perms, file_offset, path) for each line of /proc/<pid>/maps."""
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split(maxsplit=5)
            start, end = (int(x, 16) for x in parts[0].split("-"))
            path = parts[5].rstrip("\n") if len(parts) > 5 else ""
            yield start, end, parts[1], int(parts[2], 16), path


def find_mem1_addrs(pid: int, reader: ProcessMemReader | None = None, n64_off: int = VC_N64_RAM_IN_MEM1,
                    game_code: bytes = b"NACE") -> list[int]:
    """Host virtual addresses in ``pid`` where Wii MEM1 is mapped (dolphin-emu shm mappings first)."""
    reader = reader or ProcessMemReader(pid)
    maps = [m for m in iter_maps(pid) if "r" in m[2] and m[1] - m[0] >= n64_off + len(N64_MAGIC)]
    maps.sort(key=lambda m: SHM_BASE_NAME not in m[4])
    out = []
    for start, end, _perms, _pgoff, _path in maps:
        try:
            if reader.read(start + GAME_CODE_OFF, 4) == game_code and reader.read(start + n64_off, 12) == N64_MAGIC:
                out.append(start)
        except PermissionError:
            raise
        except OSError:
            continue
    return out


class PolledN64Ram(np.ndarray):
    """4 MB copy of N64 RAM that is refreshed explicitly. Call ``refresh()`` before each read (~1 syscall)."""

    def __new__(cls, reader: ProcessMemReader, addr: int):
        obj = np.zeros(N64_RAM_SIZE, np.uint8).view(cls)
        obj._reader, obj._addr = reader, addr
        obj.refresh()
        return obj

    def __array_finalize__(self, obj):
        self._reader = getattr(obj, "_reader", None)
        self._addr = getattr(obj, "_addr", None)

    def refresh(self) -> None:
        if self._reader is None or self.shape != (N64_RAM_SIZE,):
            raise RuntimeError("refresh() needs the full 4 MB PolledN64Ram, not a slice")
        self._reader.read_into(self._addr, self.view(np.ndarray))


def attach_via_vm_readv(pid: int, *, n64_off: int = VC_N64_RAM_IN_MEM1, game_code: bytes = b"NACE") -> PolledN64Ram | None:
    reader = ProcessMemReader(pid)
    addrs = find_mem1_addrs(pid, reader, n64_off, game_code)
    return PolledN64Ram(reader, addrs[0] + n64_off) if addrs else None


# ------------------------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------------------------
def backend_of(ram: np.ndarray) -> str:
    return "vm_readv (copy; call refresh())" if isinstance(ram, PolledN64Ram) else "shm mmap (zero-copy)"


def attach_n64_ram_linux(pid: int, timeout: float = 60.0, *, method: str = "auto", writable: bool = False,
                         poll: float = 0.25) -> np.ndarray:
    """Wait for Dolphin to populate MEM1 with the VC N64 RAM, then return a uint8 view of the 4 MB.

    method: "auto" (shm, falling back to vm_readv only if the shm fd is missing or unreadable),
            "shm" or "vm_readv".
    writable: open the shm read-write (zero-copy path only); by default the array is read-only.
    """
    if method not in ("auto", "shm", "vm_readv"):
        raise ValueError(method)
    t0 = time.monotonic()
    notes: dict[str, str] = {}
    while True:
        if not os.path.exists(f"/proc/{pid}"):
            raise ProcessLookupError(f"Dolphin pid {pid} exited before N64 RAM appeared")
        use_fallback = method == "vm_readv"
        if method in ("auto", "shm"):
            try:
                arr = attach_via_shm(pid, writable=writable)
                if arr is not None:
                    return arr
                notes["shm"] = "arena found, MEM1/N64 signatures not present yet (still booting?)"
            except (FileNotFoundError, PermissionError) as e:
                notes["shm"] = str(e)
                use_fallback = method == "auto"
        if use_fallback:
            try:
                arr = attach_via_vm_readv(pid)
                if arr is not None:
                    return arr
                notes["vm_readv"] = "no mapping with MEM1/N64 signatures yet"
            except PermissionError as e:
                notes["vm_readv"] = str(e)
        if time.monotonic() - t0 >= timeout:
            detail = "; ".join(f"{k}: {v}" for k, v in notes.items())
            raise TimeoutError(f"N64 RAM not found in Dolphin pid {pid} after {timeout:.0f}s ({detail}). "
                               "Check `df -h /dev/shm` (needs ~100 MB per instance) and that the game booted.")
        time.sleep(poll)


# ------------------------------------------------------------------------------------------------
# Diagnostics: python -m zelda_rl.linuxmem <dolphin_pid> [--method auto|shm|vm_readv] [--watch]
# ------------------------------------------------------------------------------------------------
_SAVE_CTX, _PLAY_STATE, _VI_COUNTER = 0x8011AC80, 0x801C8D60, 0x8011DBB0   # see zelda_rl/oot_ram.py


def _summary(ram) -> str:
    u32 = lambda a: struct.unpack_from(">I", ram, a & 0x3FFFFF)[0]
    s16 = lambda a: struct.unpack_from(">h", ram, a & 0x3FFFFF)[0]
    player = u32(_PLAY_STATE + 0x1C44)
    pos = struct.unpack_from(">fff", ram, (player & 0x3FFFFF) + 0x24) if 0x80000000 <= player < 0x80400000 else None
    fmt_pos = "-" if pos is None else "(" + ", ".join(f"{v:.1f}" for v in pos) + ")"
    return (f"vi={u32(_VI_COUNTER)} scene={s16(_PLAY_STATE + 0xA4)} rupees={s16(_SAVE_CTX + 0x34)} "
            f"health={s16(_SAVE_CTX + 0x30)} link_pos={fmt_pos}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Attach to a Linux Dolphin's N64 RAM and print OoT state.")
    ap.add_argument("pid", type=int)
    ap.add_argument("--method", default="auto", choices=["auto", "shm", "vm_readv"])
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--watch", action="store_true", help="print state every 0.5 s until Ctrl-C")
    args = ap.parse_args(argv)
    for path, target in find_dolphin_shm_fds(args.pid):
        print(f"shm fd: {path} -> {target}")
    t0 = time.monotonic()
    ram = attach_n64_ram_linux(args.pid, args.timeout, method=args.method)
    print(f"attached in {time.monotonic() - t0:.2f}s via {backend_of(ram)}")
    refresh = getattr(ram, "refresh", None)
    try:
        while True:
            if refresh:
                refresh()
            print(_summary(ram), flush=True)
            if not args.watch:
                return 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
