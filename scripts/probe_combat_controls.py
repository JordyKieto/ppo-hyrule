"""Live check: which pipe buttons shield (stateFlags1 bit 22) and Z-target a Deku Baba (targetedActor pointer)."""
import os, sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import DolphinGoalEnv, EnvConfig
from zelda_rl.obs import read_state_file_v2
from zelda_rl.combat import read_combat_state, SF1_SHIELD, SF1_ZTARGET
D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
files = {k: D / f"NACE01.s{k:02d}" for k in (6, 10)}
states = {k: read_state_file_v2(f) for k, f in files.items()}
env = DolphinGoalEnv(0, Path("runs/instances/probe_combat"), files, states, states[10], EnvConfig(video_backend="Null", unlimited_speed=False, obs_version=2))
def vi_wait(n):
    v0 = read_combat_state(env.ram)
    from zelda_rl.obs import read_state
    t0 = read_state(env.ram).vi
    while read_state(env.ram).vi < t0 + n: time.sleep(0.002)
def show(label):
    s = read_combat_state(env.ram)
    tgt = s.enemies.get(s.targeted); arr = s.enemies.get(s.arrow)
    near = min((math.dist(e.pos, s.pos), hex(e.id), e.hp) for e in s.enemies.values() if e.id == 0x55) if any(e.id == 0x55 for e in s.enemies.values()) else None
    print(f"{label:34s} shield={s.bit(SF1_SHIELD)} z={s.bit(SF1_ZTARGET)} targeted={'baba hp%d' % tgt.hp if tgt and tgt.id == 0x55 else hex(s.targeted)} "
          f"arrow={'baba' if arr and arr.id == 0x55 else hex(s.arrow)} nearest_baba={near} room={s.room} hp={s.hp}")
for slot in (10,):
    env.load_slot(slot); vi_wait(20); show(f"slot {slot} loaded")
    for btn, label in (("R", "hold R (GC R)"), ("L", "hold L (GC L)"), ("Z", "hold Z (GC Z)")):
        env.inst.pad.send(f"PRESS {btn}"); vi_wait(20); show(label + " 20 VIs")
        env.inst.pad.send(f"RELEASE {btn}"); vi_wait(20); show(f"  released {btn}")
    # walk toward nearest baba while holding L, then swing B while locked
    s = read_combat_state(env.ram)
    e = min((e for e in s.enemies.values() if e.id == 0x55), key=lambda e: math.dist(e.pos, s.pos))
    env.load_slot(slot); vi_wait(10)
    env.inst.pad.send("PRESS L"); vi_wait(30); show("L held 30 VIs (lock?)")
    env.inst.pad.send("SET MAIN 0.5 1.0"); vi_wait(40); show("L + walk forward 40 VIs")
    env.inst.pad.send("SET MAIN 0.5 0.5", "PRESS B"); vi_wait(6); env.inst.pad.send("RELEASE B"); vi_wait(30); show("L + B swing")
    env.inst.pad.send("RELEASE L"); vi_wait(10); show("released L")
env.close()
