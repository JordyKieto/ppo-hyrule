"""Pod debug: boot Dolphin like the env does, then inspect fds, big mappings, MEM1 signature, and Dolphin's log."""
import os, sys, time, shutil, ctypes, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance
ud = Path("runs/instances/debug"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
(ud / "Config/Logger.ini").write_text("[Options]\nWriteToFile = True\nVerbosity = 4\n[Logs]\nBOOT = True\nCORE = True\nMASTER = True\nIOS = True\nIOS_ES = True\nCOMMON = True\n")
inst = DolphinInstance(ud, video_backend="Null")
pid = inst.proc.pid
for t in (5, 15, 30):
    time.sleep(t - (0 if t == 5 else (5 if t == 15 else 15)))
    alive = inst.proc.poll() is None
    fds = []
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try: fds.append(os.readlink(f"/proc/{pid}/fd/{fd}"))
            except OSError: pass
    except OSError as e: fds = [repr(e)]
    shm = [f for f in fds if "shm" in f or "memfd" in f or "dolphin" in f.lower()]
    big = []
    for line in open(f"/proc/{pid}/maps"):
        a, b = [int(x, 16) for x in line.split()[0].split("-")]
        if b - a >= 16 << 20: big.append((hex(a), (b - a) >> 20, line.split()[1], line.split()[-1] if len(line.split()) > 5 else ""))
    print(f"t={t}s alive={alive} shm-ish fds={shm[:6]} big maps (addr, MB, perms, path)={big[:12]}", flush=True)
# signature search via process_vm_readv in big rw mappings
libc = ctypes.CDLL("libc.so.6", use_errno=True)
class iovec(ctypes.Structure): _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]
def read(addr, n):
    buf = ctypes.create_string_buffer(n)
    l, r = iovec(ctypes.addressof(buf), n), iovec(addr, n)
    got = libc.process_vm_readv(pid, ctypes.byref(l), 1, ctypes.byref(r), 1, 0)
    return buf.raw[:got] if got > 0 else b""
MAGIC = bytes.fromhex("3c1a8000275a260003400008")
found = []
for line in open(f"/proc/{pid}/maps"):
    parts = line.split(); a, b = [int(x, 16) for x in parts[0].split("-")]
    if "r" not in parts[1] or b - a < 0xF64120 + 12 or b - a > 8 << 30: continue
    head = read(a + 0x3180, 4); vec = read(a + 0xF64120, 12)
    if head == b"NACE" or vec == MAGIC:
        found.append((hex(a), (b - a) >> 20, parts[-1] if len(parts) > 5 else "", head, vec == MAGIC))
print("MEM1 candidates (addr, MB, path, @0x3180, N64 vector@0xF64120):", found[:8])
print("errno last:", ctypes.get_errno())
log = ud / "Logs/dolphin.log"
print("---- dolphin.log tail ----"); print(log.read_text(errors="replace")[-2500:] if log.exists() else "(no log)")
print("---- stdout tail ----"); print((ud / "dolphin_stdout.log").read_text(errors="replace")[-1500:])
inst.stop()
