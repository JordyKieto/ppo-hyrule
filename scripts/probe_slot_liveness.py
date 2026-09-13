"""Per savestate: do NPC actors move (game simulating)? does the game see pad input? does Link respond?"""
import os, sys, time, shutil, struct, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance, HOTKEY_MAP
from zelda_rl.live import attach_n64_ram, be_s16, be_u32
PS = 0x801C8D60
def actors(ram, cat):
    out, a = [], be_u32(ram, PS + 0x1C24 + 0x10 + cat * 8)
    while a >> 24 == 0x80 and len(out) < 64:
        out.append((a, struct.unpack_from(">fff", ram, (a & 0x3FFFFF) + 0x24))); a = be_u32(ram, a + 0x124)
    return out
slots = sys.argv[1:]
for slot in slots:
    ud = Path("runs/instances/sl"); shutil.rmtree(ud, ignore_errors=True)
    ud = build_user_dir(ud.resolve(), dump_frames=False)
    shutil.copy(os.path.expanduser(slot), ud / "StateSaves/NACE01.s01")
    inst = DolphinInstance(ud); ram = attach_n64_ram(inst.proc.pid)
    k = HOTKEY_MAP["Load State/Load State Slot 1"]
    for _ in range(40):
        if be_s16(ram, PS + 0xA4) in (52, 85): break
        inst.hk.send(f"PRESS {k}"); time.sleep(0.15); inst.hk.send(f"RELEASE {k}"); time.sleep(0.35)
    time.sleep(1.0)
    n0 = {a: p for a, p in actors(ram, 4)}; time.sleep(2.0); n1 = {a: p for a, p in actors(ram, 4)}
    npc_move = max([math.dist(n0[a], n1[a]) for a in n0 if a in n1] or [0])
    inst.pad.send("SET MAIN 1.0 0.5"); time.sleep(0.4)
    pad = ram[(PS + 0x14) & 0x3FFFFF:(PS + 0x18) & 0x3FFFFF].tobytes().hex()
    pl = be_u32(ram, PS + 0x1C44); p0 = struct.unpack_from(">fff", ram, (pl & 0x3FFFFF) + 0x24)
    time.sleep(1.0); p1 = struct.unpack_from(">fff", ram, (pl & 0x3FFFFF) + 0x24); inst.pad.send("SET MAIN 0.5 0.5")
    print(f"{Path(slot).name:12s} scene={be_s16(ram, PS+0xA4):3d} NPCs={len(n0):2d} max NPC move 2s={npc_move:7.1f}  pad={pad}  Link moved={math.dist(p0, p1):6.1f}", flush=True)
    inst.stop(); time.sleep(1.5)
