"""Live check: how the pipe controller draws / fires the slingshot (C-down) and Z-targets Queen Gohma.
Prints held item action (Player +0x141), stateFlags1 aim bits (3, 9, 20), slingshot seeds, lock-on, Gohma state (+0x180)."""
import os, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig
from zelda_rl.obs import PLAY_STATE as PS, SAVE_CTX as SC, read_state, read_state_file_v2

D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
SLOTS = [int(s) for s in os.environ.get("SLOTS", "7 2").split()]
files = {k: D / f"NACE01.s{k:02d}" for k in SLOTS}
states = {k: read_state_file_v2(f) for k, f in files.items()}
env = DolphinGoalEnv(0, Path("runs/instances/probe_boss"), files, states, states[SLOTS[0]],
                     EnvConfig(video_backend="Null", unlimited_speed=False, obs_version=2))
r = env.ram
u32 = lambda a: struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
u16 = lambda a: struct.unpack_from(">H", r, a & 0x3FFFFF)[0]


def vi_wait(n):
    t0 = read_state(r).vi
    while read_state(r).vi < t0 + n:
        time.sleep(0.002)


def boss():
    a = u32(PS + 0x1C24 + 0x10 + 9 * 8)
    return a if 0x80000000 <= a < 0x80400000 else None


def show(label):
    pl = u32(PS + 0x1C44)
    sf1 = u32(pl + 0x66C)
    b = boss()
    tgt = u32(PS + 0x1CCC)
    print(f"{label:40s} held={struct.unpack_from('>b', r, (pl + 0x141) & 0x3FFFFF)[0]:3d} aim(3,9,20)={sf1 >> 3 & 1}{sf1 >> 9 & 1}{sf1 >> 20 & 1} "
          f"z={sf1 >> 15 & 1} seeds={r[(SC + 0x8C + 6) & 0x3FFFFF]:2d} lock={'gohma' if b and tgt == b else hex(tgt)} "
          f"gohma hp={r[(b + 0xAF) & 0x3FFFFF] if b else None} fn={u32(b + 0x180):#x} y={struct.unpack_from('>f', r, (b + 0x28) & 0x3FFFFF)[0]:.0f}", flush=True)


def tap(cmd_press, cmd_release, hold=4, after=20, label=""):
    env.inst.pad.send(cmd_press); vi_wait(hold); env.inst.pad.send(cmd_release); vi_wait(after); show(label)


for slot in SLOTS:
    env.load_slot(slot); vi_wait(20); show(f"slot {slot} loaded")
    tap("SET C 0.5 0.0", "SET C 0.5 0.5", label="tap C-down (draw?)")
    tap("SET C 0.5 0.0", "SET C 0.5 0.5", label="tap C-down again (fire?)")
    tap("PRESS B", "RELEASE B", label="tap B (put away?)")
    env.load_slot(slot); vi_wait(20); show(f"slot {slot} reloaded")
    env.inst.pad.send("SET C 0.5 0.0"); vi_wait(30); show("hold C-down 30 VIs")
    env.inst.pad.send("SET C 0.5 0.5"); vi_wait(30); show("release C-down")
    env.load_slot(slot); vi_wait(20)
    env.inst.pad.send("PRESS L"); vi_wait(20); show("hold L (Z-target) 20 VIs")
    tap("SET C 0.5 0.0", "SET C 0.5 0.5", label="L + tap C-down")
    tap("SET C 0.5 0.0", "SET C 0.5 0.5", after=30, label="L + tap C-down again")
    tap("SET C 0.5 0.0", "SET C 0.5 0.5", after=60, label="L + tap C-down x3")
    env.inst.pad.send("RELEASE L"); vi_wait(20); show("release L")
env.close()
