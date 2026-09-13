"""Live-monitor a human playing in the signed Dolphin: log combat-relevant RAM events + a full 10 Hz trace.

    DOLPHIN_PID=<pid> .venv/bin/python-dbg -u scripts/monitor_play.py      # or it finds the signed Dolphin itself

Events (stdout): MONITORING signal, scene changes, Link damage/heal/death, enemy spawn / HP change / defeat,
Link state-flag changes (Z-target, shield, damaged, ...), equipment / C-item / ammo changes.
Trace: runs/manual/trace_<timestamp>.jsonl (for reward design: Link hp/pos/yaw/flags + every enemy each tick).
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.live import attach_n64_ram  # noqa: E402
from zelda_rl.obs import PLAY_STATE, SAVE_CTX, SF1_BITS, VI_COUNTER  # noqa: E402

CATS = ["SWITCH", "BG", "PLAYER", "EXPLOSIVE", "NPC", "ENEMY", "PROP", "ITEMACTION", "MISC", "BOSS", "DOOR", "CHEST"]
SF1_NAMES = {b: n for n, b in SF1_BITS} | {22: "shielding", 17: "parallel"}
u32 = lambda r, a: struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
s16 = lambda r, a: struct.unpack_from(">h", r, a & 0x3FFFFF)[0]
u16 = lambda r, a: struct.unpack_from(">H", r, a & 0x3FFFFF)[0]
vec3 = lambda r, a: struct.unpack_from(">fff", r, a & 0x3FFFFF)


def find_pid() -> int:
    if os.environ.get("DOLPHIN_PID"):
        return int(os.environ["DOLPHIN_PID"])
    out = subprocess.run(["pgrep", "-f", ".cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    if not pids:
        raise SystemExit("signed Dolphin (~/.cache/zelda-rl/Dolphin.app) is not running")
    return pids[0]


def snapshot(ram) -> dict:
    ps, sc = PLAY_STATE, SAVE_CTX
    player = u32(ram, ps + 0x1C44)
    ok = 0x80000000 <= player < 0x80400000
    lp = vec3(ram, player + 0x24) if ok else (math.nan,) * 3
    actors = []
    for cat in (5, 9):                                   # ENEMY, BOSS
        a, guard = u32(ram, ps + 0x1C24 + 0x10 + cat * 8), 0
        while 0x80000000 <= a < 0x80400000 and guard < 64:
            p = vec3(ram, a + 0x24)
            actors.append({"ptr": a, "id": u16(ram, a), "cat": CATS[cat], "params": u16(ram, a + 0x1C),
                           "hp": ram[(a + 0xAF) & 0x3FFFFF], "flags": u32(ram, a + 4),
                           "pos": [round(v, 1) for v in p], "dist": round(math.dist(p, lp), 1) if ok else None,
                           "yaw": u16(ram, a + 0x32)})
            a = u32(ram, a + 0x124)
            guard += 1
    return {
        "t": time.time(), "vi": u32(ram, VI_COUNTER), "scene": s16(ram, ps + 0xA4),
        "hp": s16(ram, sc + 0x30), "hp_cap": s16(ram, sc + 0x2E), "rupees": s16(ram, sc + 0x34),
        "link_ok": ok, "pos": [round(v, 1) for v in lp], "yaw": u16(ram, player + 0x32) if ok else None,
        "sf1": u32(ram, player + 0x66C) if ok else 0,
        "b_item": ram[(sc + 0x68) & 0x3FFFFF], "c_items": list(ram[(sc + 0x69) & 0x3FFFFF:(sc + 0x6C) & 0x3FFFFF]),
        "equipment": u16(ram, sc + 0x9C), "ammo": list(ram[(sc + 0x8C) & 0x3FFFFF:(sc + 0x9C) & 0x3FFFFF]),
        "enemies": actors,
    }


def main():
    pid = find_pid()
    ram = attach_n64_ram(pid, timeout=3600)          # waits until you start/load a game
    out_dir = Path("runs/manual")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    trace = open(trace_path, "w")
    prev = snapshot(ram)
    t0 = time.time()
    print(f"MONITORING dolphin pid {pid} | scene {prev['scene']} | hearts {prev['hp'] / 16:.2f}/{prev['hp_cap'] / 16:.0f} | "
          f"enemies {len(prev['enemies'])} | trace {trace_path}", flush=True)
    ts = lambda: f"[{time.time() - t0:6.1f}s]"
    last_trace = 0.0
    while True:
        cur = snapshot(ram)
        if time.time() - last_trace >= 0.1:
            trace.write(json.dumps(cur, default=int) + "\n")
            trace.flush()
            last_trace = time.time()
        if cur["scene"] != prev["scene"]:
            print(f"{ts()} SCENE {prev['scene']} -> {cur['scene']} | enemies {len(cur['enemies'])}", flush=True)
        if cur["hp"] != prev["hp"]:
            d = cur["hp"] - prev["hp"]
            kind = "DAMAGE" if d < 0 else "HEAL"
            print(f"{ts()} LINK {kind} {d / 16:+.2f} hearts -> {cur['hp'] / 16:.2f}/{cur['hp_cap'] / 16:.0f}", flush=True)
            if cur["hp"] <= 0:
                print(f"{ts()} LINK DIED (hp 0)", flush=True)
        if cur["sf1"] != prev["sf1"]:
            on = [SF1_NAMES.get(b, f"bit{b}") for b in range(32) if cur["sf1"] >> b & 1 and not prev["sf1"] >> b & 1]
            off = [SF1_NAMES.get(b, f"bit{b}") for b in range(32) if prev["sf1"] >> b & 1 and not cur["sf1"] >> b & 1]
            interesting = {"z_targeting", "friendly_focus", "shielding", "damaged", "dead", "jumping", "freefall",
                           "in_cutscene", "talking", "getting_item", "carrying"}
            on_i, off_i = [x for x in on if x in interesting], [x for x in off if x in interesting]
            if on_i or off_i:
                print(f"{ts()} LINK STATE +{on_i} -{off_i}", flush=True)
        pe = {a["ptr"]: a for a in prev["enemies"]}
        ce = {a["ptr"]: a for a in cur["enemies"]}
        for p, a in ce.items():
            if p not in pe:
                print(f"{ts()} ENEMY SPAWN id {a['id']:#06x} {a['cat']} params {a['params']:#06x} hp {a['hp']} at dist {a['dist']}", flush=True)
            elif a["hp"] != pe[p]["hp"]:
                print(f"{ts()} ENEMY HIT id {a['id']:#06x} hp {pe[p]['hp']} -> {a['hp']} (dist {a['dist']})", flush=True)
        for p, a in pe.items():
            if p not in ce:
                print(f"{ts()} ENEMY GONE (defeated/despawned) id {a['id']:#06x} last hp {a['hp']} last dist {a['dist']} | "
                      f"{len(ce)} enemies left", flush=True)
        for k, label in (("b_item", "B ITEM"), ("c_items", "C ITEMS"), ("equipment", "EQUIPMENT"), ("ammo", "AMMO")):
            if cur[k] != prev[k]:
                print(f"{ts()} {label} {prev[k]} -> {cur[k]}", flush=True)
        prev = cur
        time.sleep(0.02)


if __name__ == "__main__":
    main()
