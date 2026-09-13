"""Offline check of v2 RAM fields across the user's savestates (player flags, camera, actors, save flags, dialog)."""
import os, sys, struct, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.savestate import decompress_state, find_n64_ram
PS, SC = 0x801C8D60, 0x8011AC80
D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
rams = {}
for k in (1, 2, 3, 4, 5):
    raw = decompress_state((D / f"NACE01.s{k:02d}").read_bytes()); b = find_n64_ram(raw); rams[k] = raw[b:b + 0x400000]
u8 = lambda r, a: r[a & 0x3FFFFF]
u16 = lambda r, a: struct.unpack_from(">H", r, a & 0x3FFFFF)[0]
s16 = lambda r, a: struct.unpack_from(">h", r, a & 0x3FFFFF)[0]
u32 = lambda r, a: struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
f3 = lambda r, a: struct.unpack_from(">fff", r, a & 0x3FFFFF)
for k, r in rams.items():
    pl = u32(r, PS + 0x1C44); pos = f3(r, pl + 0x24)
    sf1, sf2, sf3 = u32(r, pl + 0x66C), u32(r, pl + 0x670), u8(r, pl + 0x674)
    camptr = u32(r, PS + 0x790); active = s16(r, PS + 0x7A0)
    at, eye = f3(r, PS + 0x1E0 + 0x50), f3(r, PS + 0x1E0 + 0x5C)
    cam_fwd = math.degrees(math.atan2(at[2] - eye[2], at[0] - eye[0]))
    print(f"slot {k}: stateFlags1={sf1:#010x} (TALKING bit6={bool(sf1 & 0x40)}) sf2={sf2:#010x} sf3={sf3:#04x} | camPtrs[0]={camptr:#x} (=PS+0x1E0? {camptr == PS + 0x1E0}) activeCam={active} "
          f"| eye-Link {math.dist(eye, pos):6.0f} at-Link {math.dist(at, pos):5.0f} cam_fwd {cam_fwd:6.1f}deg link_yaw {u16(r, pl + 0x32) / 65536 * 360:6.1f}deg")
    items = bytes(r[(SC + 0x74) & 0x3FFFFF:(SC + 0x74 + 24) & 0x3FFFFF]).hex()
    print(f"         items={items} equip={u16(r, SC + 0x9C):#06x} upgrades={u32(r, SC + 0xA0):#010x} quest={u32(r, SC + 0xA4):#010x} "
          f"curEquips(btnB item)={u8(r, SC + 0x68):#04x} equipU16={u16(r, SC + 0x70):#06x}")
# save flags diff between slots
blocks = {"eventChkInf": (0xED4, 28), "itemGetInf": (0xEF0, 8), "infTable": (0xEF8, 60), "scene85flags": (0xD4 + 85 * 0x1C, 0x1C)}
for name, (off, n) in blocks.items():
    vals = {k: np.frombuffer(bytes(r[(SC + off) & 0x3FFFFF:(SC + off + n) & 0x3FFFFF]), np.uint8) for k, r in rams.items()}
    bits = {k: int(np.unpackbits(v).sum()) for k, v in vals.items()}
    diff15 = int(np.unpackbits(vals[1] ^ vals[5]).sum())
    print(f"{name:12s} bits set per slot {bits}  slot1 vs slot5 differing bits {diff15}")
# nearest actors of slot 2 (Link walking near a rupee)
r = rams[2]; pl = u32(r, PS + 0x1C44); lp = f3(r, pl + 0x24)
acts = []
for cat in range(12):
    a = u32(r, PS + 0x1C24 + 0x10 + cat * 8)
    while a >> 24 == 0x80:
        if a != pl:
            p = f3(r, a + 0x24)
            acts.append((math.dist(p, lp), u16(r, a), u8(r, a + 2), cat, s16(r, a + 0x1C), u32(r, a + 4), u8(r, a + 0xAF), struct.unpack_from(">f", r, (a + 0x90) & 0x3FFFFF)[0], p))
        a = u32(r, a + 0x124)
acts.sort()
print("slot 2 nearest actors: dist | id | cat(field) cat(list) | params | flags | health | game xzDist")
for d, aid, c, cl, prm, fl, hp, xz, p in acts[:16]:
    print(f"  {d:7.0f} id={aid:#06x} cat={c:2d}/{cl:2d} params={prm & 0xFFFF:#06x} flags={fl:#010x} hp={hp:3d} xzDist={xz:7.0f}")
# dialog byte candidates: nonzero in slot 1 (talking), zero in slots 2,3,5
cands = []
for off in range(0x20D8, 0x12600):
    a = (PS + off) & 0x3FFFFF
    if rams[1][a] != 0 and all(rams[k][a] == 0 for k in (2, 3, 5)):
        cands.append((off, rams[1][a]))
print(f"{len(cands)} dialog-candidate bytes in msgCtx region; known-guess offsets:",
      [(hex(o), v) for o, v in cands if o in (0x104BD, 0x104E4, 0x104C9, 0x10A20, 0x104CA, 0x104CB)])
print("first 40:", [(hex(o), v) for o, v in cands[:40]])
