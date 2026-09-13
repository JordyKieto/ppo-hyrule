"""Record Deku Baba internals + Link collider region while a human fights, to decode enemy state (lunge/stunned/dying),
enemy facing/velocity, and the real shield-block signal.  Live: prints state-candidate changes next to combat events.

    DOLPHIN_PID=<pid> .venv/bin/python-dbg -u scripts/monitor_fight.py
Trace: runs/manual/fight_<ts>.jsonl  (20 Hz; raw hex windows for offline analysis)
"""
import json, math, os, struct, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.live import attach_n64_ram
from zelda_rl.obs import PLAY_STATE as PS, SAVE_CTX as SC
u32 = lambda r, a: struct.unpack_from(">I", r, a & 0x3FFFFF)[0]
u16 = lambda r, a: struct.unpack_from(">H", r, a & 0x3FFFFF)[0]
s16 = lambda r, a: struct.unpack_from(">h", r, a & 0x3FFFFF)[0]
f32 = lambda r, a: struct.unpack_from(">f", r, a & 0x3FFFFF)[0]
v3 = lambda r, a: struct.unpack_from(">fff", r, a & 0x3FFFFF)
raw = lambda r, a, n: bytes(r[a & 0x3FFFFF:(a + n) & 0x3FFFFF]).hex()
BABA, WIN0, WIN1 = 0x0055, 0x14C, 0x3A0          # EnDekubaba-specific struct region (after the 0x14C Actor header)
PWIN0, PWIN1 = 0x3F0, 0x900                        # Player region incl. colliders (cylinder, weapon & shield quads)

def pid():
    if os.environ.get("DOLPHIN_PID"): return int(os.environ["DOLPHIN_PID"])
    out = subprocess.run(["pgrep", "-f", ".cache/zelda-rl/Dolphin.app/Contents/MacOS/Dolphin"], capture_output=True, text=True).stdout.split()
    if not out: sys.exit("signed Dolphin not running")
    return int(out[0])

def snap(r):
    pl = u32(r, PS + 0x1C44); ok = pl >> 24 == 0x80
    lp = v3(r, pl + 0x24) if ok else (math.nan,) * 3
    babas = []
    a = u32(r, PS + 0x1C24 + 0x10 + 5 * 8)
    while a >> 24 == 0x80:
        if u16(r, a) == BABA:
            p = v3(r, a + 0x24)
            babas.append({"ptr": a, "hp": int(r[(a + 0xAF) & 0x3FFFFF]), "pos": [round(x, 1) for x in p],
                          "dist": round(math.dist(p, lp), 1) if ok else None,
                          "rot_world": [s16(r, a + 0x30), s16(r, a + 0x32), s16(r, a + 0x34)],
                          "rot_shape": [s16(r, a + 0xB4), s16(r, a + 0xB6), s16(r, a + 0xB8)],
                          "vel": [round(x, 2) for x in v3(r, a + 0x5C)], "speed": round(f32(r, a + 0x68), 2),
                          "yaw_to_player": s16(r, a + 0x8A), "flags": u32(r, a + 4), "focus": [round(x, 1) for x in v3(r, a + 0x38)],
                          "fn": u32(r, a + WIN0), "raw": raw(r, a + WIN0, WIN1 - WIN0)})
        a = u32(r, a + 0x124)
    return {"t": time.time(), "hp": s16(r, SC + 0x30), "sf1": u32(r, pl + 0x66C) if ok else 0, "sf2": u32(r, pl + 0x670) if ok else 0,
            "pos": [round(x, 1) for x in lp], "yaw": u16(r, pl + 0x32) if ok else 0, "targeted": u32(r, PS + 0x1CCC),
            "player_raw": raw(r, pl + PWIN0, PWIN1 - PWIN0) if ok else "", "babas": babas}

def main():
    p = pid(); r = attach_n64_ram(p, timeout=3600)
    out = Path("runs/manual") / f"fight_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w"); prev = snap(r); t0 = time.time()
    print(f"MONITORING (fight recorder) pid {p} | hearts {prev['hp']/16:.1f} | babas {[(b['hp'], b['dist']) for b in prev['babas']]} | {out}", flush=True)
    ts = lambda: f"[{time.time()-t0:6.1f}s]"
    fn_seen = {}
    while True:
        cur = snap(r); f.write(json.dumps(cur) + "\n")
        pb = {b["ptr"]: b for b in prev["babas"]}
        for b in cur["babas"]:
            o = pb.get(b["ptr"])
            tag = f"baba@{b['ptr']:#x}"
            fn_seen.setdefault(b["fn"], len(fn_seen))
            if o is None:
                print(f"{ts()} {tag} appears hp {b['hp']} dist {b['dist']} fn#{fn_seen[b['fn']]}={b['fn']:#x}", flush=True); continue
            if b["fn"] != o["fn"]:
                print(f"{ts()} {tag} STATE fn#{fn_seen[o['fn']]}->fn#{fn_seen[b['fn']]} ({b['fn']:#x}) hp {b['hp']} dist {b['dist']} speed {b['speed']}", flush=True)
            if b["hp"] != o["hp"]:
                print(f"{ts()} {tag} HP {o['hp']}->{b['hp']}", flush=True)
        for ptr in set(pb) - {b["ptr"] for b in cur["babas"]}:
            print(f"{ts()} baba@{ptr:#x} gone", flush=True)
        if cur["hp"] != prev["hp"]:
            print(f"{ts()} LINK HP {prev['hp']/16:.2f}->{cur['hp']/16:.2f}", flush=True)
        for bit, name in ((22, "SHIELD"), (15, "ZTARGET"), (26, "DAMAGED"), (7, "DEAD")):
            if (cur["sf1"] >> bit & 1) != (prev["sf1"] >> bit & 1):
                print(f"{ts()} LINK {name} {'on' if cur['sf1'] >> bit & 1 else 'off'}", flush=True)
        prev = cur; time.sleep(0.05)

if __name__ == "__main__":
    main()
