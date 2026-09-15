"""Record a human boss fight (Queen Gohma) from live RAM at 20 Hz, printing events as they happen.

Link: hearts, magic, stateFlags bit flips, held item action, lock-on, button items, ammo, scene/room.
Every enemy/boss actor: HP, position, distance, facing, raw struct window, and auto-detected state-machine fields
(code-pointer u32s such as actionFunc, small s16s with few values such as eye state) printed on change.
Spawns/despawns of enemies, bosses, items, doors, warps.

    DOLPHIN_PID=<pid> .venv/bin/python-dbg -u scripts/monitor_boss.py
Trace: runs/manual/boss_<ts>.jsonl
"""
import json, math, os, struct, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.live import attach_n64_ram
from zelda_rl.obs import PLAY_STATE as PS, SAVE_CTX as SC, SF1_BITS
from zelda_rl.gohma import GOHMA_STATE, LARVA_STATE

u8 = lambda r, a: r[a & 0x3FFFFF]
u32 = lambda r, a: struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
u16 = lambda r, a: struct.unpack_from(">H", r, a & 0x3FFFFF)[0]
s16 = lambda r, a: struct.unpack_from(">h", r, a & 0x3FFFFF)[0]
v3 = lambda r, a: struct.unpack_from(">fff", r, a & 0x3FFFFF)
raw = lambda r, a, n: bytes(r[a & 0x3FFFFF:(a + n) & 0x3FFFFF]).hex()

CATEGORY = ["switch", "bg", "player", "explosive", "npc", "enemy", "prop", "itemaction", "misc", "boss", "door", "chest"]
WATCH_CATS = (3, 5, 7, 8, 9, 10, 11)
FIGHTERS = (5, 9)
WIN = {9: (0x14C, 0x800), 5: (0x14C, 0x400)}
NAMES = {0x0028: "Queen Gohma?", 0x0027: "Gohma larva/egg?", 0x0055: "Deku Baba", 0x0095: "Skullwalltula",
         0x0037: "Skulltula", 0x0016: "seed/arrow?", 0x005D: "blue warp?", 0x0015: "item drop"}
AMMO = {0: "deku_sticks", 1: "deku_nuts", 2: "bombs", 3: "arrows", 6: "slingshot_seeds"}
SF1 = dict((b, n) for n, b in SF1_BITS) | {22: "shield", 9: "ready_to_fire?", 20: "first_person?"}
name = lambda i: NAMES.get(i, f"actor 0x{i:04X}")


def pid():
    if os.environ.get("DOLPHIN_PID"):
        return int(os.environ["DOLPHIN_PID"])
    out = subprocess.run(["pgrep", "-f", ".cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin$"], capture_output=True, text=True).stdout.split()
    if not out:
        sys.exit("signed Dolphin not running")
    return int(out[0])


def actors(r, lp):
    out = {}
    for cat in WATCH_CATS:
        a, guard = u32(r, PS + 0x1C24 + 0x10 + cat * 8), 0
        while 0x80000000 <= a < 0x80400000 and guard < 200:
            p = v3(r, a + 0x24)
            d = {"ptr": a, "id": u16(r, a), "cat": cat, "params": u16(r, a + 0x1C), "hp": int(u8(r, a + 0xAF)),
                 "pos": [round(x, 1) for x in p], "dist": None if math.isnan(lp[0]) else round(math.dist(p, lp), 1),
                 "yaw": s16(r, a + 0x32), "shape_yaw": s16(r, a + 0xB6), "yaw_to_player": s16(r, a + 0x8A), "flags": u32(r, a + 4),
                 "upd": u32(r, a + 0x130)}
            if cat in FIGHTERS:
                d["raw"] = raw(r, a + WIN[cat][0], WIN[cat][1] - WIN[cat][0])
            out[a] = d
            a, guard = u32(r, a + 0x124), guard + 1
    return out


def snap(r):
    pl = u32(r, PS + 0x1C44)
    ok = 0x80000000 <= pl < 0x80400000
    lp = v3(r, pl + 0x24) if ok else (math.nan,) * 3
    tgt = u32(r, PS + 0x1CCC)
    return {"t": time.time(), "scene": s16(r, PS + 0xA4), "room": int(u8(r, PS + 0x11CBC)), "hp": s16(r, SC + 0x30),
            "magic": int(u8(r, SC + 0x33)), "buttons": [int(u8(r, SC + 0x68 + i)) for i in range(4)],
            "ammo": [int(u8(r, SC + 0x8C + i)) for i in range(16)], "ok": ok, "pos": [round(x, 1) for x in lp],
            "yaw": u16(r, pl + 0x32) if ok else 0, "sf1": u32(r, pl + 0x66C) if ok else 0, "sf2": u32(r, pl + 0x670) if ok else 0,
            "held_item_action": struct.unpack_from(">b", r, (pl + 0x141) & 0x3FFFFF)[0] if ok else None,
            "player_raw": raw(r, pl + 0x13C, 0x44) if ok else "", "targeted": tgt,
            "targeted_id": u16(r, tgt) if 0x80000000 <= tgt < 0x80400000 else None, "actors": actors(r, lp)}


class FieldWatch:
    def __init__(self, raw_hex, base):
        self.base, self.prev, self.vals, self.changes = base, bytes.fromhex(raw_hex), {}, {}

    def fields(self, b):
        for o in range(0, len(b) - 3, 4):
            v = struct.unpack_from(">I", b, o)[0]
            if 0x80080000 <= v < 0x80400000 and v % 4 == 0:
                yield ("ptr", o, v)
        for o in range(0, len(b) - 1, 2):
            v = struct.unpack_from(">h", b, o)[0]
            if -32 <= v <= 64:
                yield ("s16", o, v)

    def step(self, raw_hex):
        cur, ev = bytes.fromhex(raw_hex), []
        old = {(k, o): v for k, o, v in self.fields(self.prev)}
        for k, o, v in self.fields(cur):
            s = self.vals.setdefault((k, o), set()); s.add(v)
            if (k, o) in old and old[(k, o)] != v:
                n = self.changes[(k, o)] = self.changes.get((k, o), 0) + 1
                if n <= (400 if k == "ptr" else 120) and len(s) <= (24 if k == "ptr" else 8):
                    ev.append(f"ptr+0x{self.base + o:X} {old[(k, o)]:#x}->{v:#x}" if k == "ptr" else f"s16+0x{self.base + o:X} {old[(k, o)]}->{v}")
        self.prev = cur
        return ev


def main():
    p = pid(); r = attach_n64_ram(p, timeout=3600)
    out = Path("runs/manual") / f"boss_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    f, prev, t0, watches = open(out, "w"), snap(r), time.time(), {}
    say = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    fighters = [f"{name(a['id'])} hp {a['hp']} d {a['dist']}" for a in prev["actors"].values() if a["cat"] in FIGHTERS]
    print(f"MONITORING (boss recorder) pid {p} | scene {prev['scene']} room {prev['room']} | hearts {prev['hp'] / 16:.2f} | seeds {prev['ammo'][6]} "
          f"nuts {prev['ammo'][1]} sticks {prev['ammo'][0]} | fighters {fighters} | {out}", flush=True)
    for a in prev["actors"].values():
        if a["cat"] in FIGHTERS:
            watches[a["ptr"]] = FieldWatch(a["raw"], WIN[a["cat"]][0])
    while True:
        cur = snap(r); f.write(json.dumps(cur) + "\n")
        if (cur["scene"], cur["room"]) != (prev["scene"], prev["room"]):
            say(f"SCENE/ROOM {prev['scene']}/{prev['room']} -> {cur['scene']}/{cur['room']}")
        if cur["hp"] != prev["hp"]:
            say(f"LINK HEARTS {prev['hp'] / 16:.2f} -> {cur['hp'] / 16:.2f}{'  *** LINK DIED ***' if cur['hp'] <= 0 else ''}")
        for i, (a, b) in enumerate(zip(prev["ammo"], cur["ammo"])):
            if a != b:
                say(f"AMMO {AMMO.get(i, f'slot{i}')} {a} -> {b}")
        if cur["buttons"] != prev["buttons"]:
            say(f"BUTTON ITEMS [B, C<, Cv, C>] {prev['buttons']} -> {cur['buttons']}")
        if cur["ok"] and prev["ok"]:
            for key, flags in (("sf1", SF1), ("sf2", {})):
                d = cur[key] ^ prev[key]
                for bit in range(32):
                    if d >> bit & 1:
                        say(f"LINK {key} bit {bit}{' ' + flags[bit] if bit in flags else ''} {'ON' if cur[key] >> bit & 1 else 'off'}")
            if cur["held_item_action"] != prev["held_item_action"]:
                say(f"LINK held item action {prev['held_item_action']} -> {cur['held_item_action']}")
        if cur["targeted"] != prev["targeted"]:
            say(f"LOCK-ON {'-> ' + name(cur['targeted_id']) if cur['targeted_id'] is not None else 'released'}")
        pa, ca = prev["actors"], cur["actors"]
        for ptr, a in ca.items():
            o, tag = pa.get(ptr), f"{name(a['id'])}@{ptr:#x}"
            if o is None or o["id"] != a["id"]:
                if a["id"] != 0x0016:
                    say(f"SPAWN {tag} [{CATEGORY[a['cat']]}] hp {a['hp']} params {a['params']:#06x} dist {a['dist']}")
                if a["cat"] in FIGHTERS:
                    watches[ptr] = FieldWatch(a["raw"], WIN[a["cat"]][0])
                continue
            if a["id"] == 0x28 and a["cat"] == 9 and a["params"] == 0xFFFF:
                gs = lambda x: GOHMA_STATE.get(struct.unpack_from(">I", bytes.fromhex(x["raw"]), 0x180 - 0x14C)[0] - x["upd"], "?")
                if gs(a) != gs(o):
                    say(f"GOHMA STATE {gs(o)} -> {gs(a)} (hp {a['hp']}, d {a['dist']})")
            if a["hp"] != o["hp"]:
                say(f"HP {tag} {o['hp']} -> {a['hp']}{'  *** DEFEATED ***' if a['hp'] == 0 else ''} (dist {a['dist']})")
            if ptr in watches:
                ev = watches[ptr].step(a["raw"])
                if ev and (a["cat"] == 9 or len(ev) <= 3):
                    say(f"STATE {tag} hp {a['hp']} d {a['dist']}: " + ", ".join(ev[:6]) + (f" (+{len(ev) - 6})" if len(ev) > 6 else ""))
        for ptr in set(pa) - set(ca):
            if pa[ptr]["id"] != 0x0016:
                say(f"GONE {name(pa[ptr]['id'])}@{ptr:#x} [{CATEGORY[pa[ptr]['cat']]}] last hp {pa[ptr]['hp']}")
            watches.pop(ptr, None)
        prev = cur
        time.sleep(0.05)


if __name__ == "__main__":
    main()
