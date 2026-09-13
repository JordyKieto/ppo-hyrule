"""Which port/device does the user's savestate listen on? Bind the pipe pad to all 4 GC ports and inspect the game's 4 input structs."""
import os, sys, time, shutil, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.dolphin import build_user_dir, DolphinInstance, HOTKEY_MAP
from zelda_rl.live import attach_n64_ram, be_s16, be_u32
SLOT = os.path.expanduser(os.environ.get("SLOT", "~/Library/Application Support/Dolphin/StateSaves/NACE01.s04"))
ud = Path("runs/instances/ports"); shutil.rmtree(ud, ignore_errors=True)
ud = build_user_dir(ud.resolve(), dump_frames=False)
ini = (ud / "Config/GCPadNew.ini").read_text()
pad_body = ini.split("\n", 1)[1]
(ud / "Config/GCPadNew.ini").write_text("".join(f"[GCPad{i}]\n{pad_body}" for i in range(1, 5)))
d = (ud / "Config/Dolphin.ini").read_text()
for i in (1, 2, 3): d = d.replace(f"SIDevice{i} = 0", f"SIDevice{i} = 6")
(ud / "Config/Dolphin.ini").write_text(d)
mode = os.environ.get("WIIMOTE", "none")
if mode == "none":
    (ud / "Config/WiimoteNew.ini").write_text("[Wiimote1]\nSource = 0\n[Wiimote2]\nSource = 0\n[Wiimote3]\nSource = 0\n[Wiimote4]\nSource = 0\n")
    d = (ud / "Config/Dolphin.ini").read_text().replace("[Core]\n", "[Core]\nWiimoteSource0 = 0\nWiimoteSource1 = 0\nWiimoteSource2 = 0\nWiimoteSource3 = 0\n")
    (ud / "Config/Dolphin.ini").write_text(d)
elif mode == "classic":
    cc = {"Classic/Buttons/A": "`Button A`", "Classic/Buttons/B": "`Button B`", "Classic/Buttons/X": "`Button X`", "Classic/Buttons/Y": "`Button Y`",
          "Classic/Buttons/+": "`Button START`", "Classic/Buttons/ZL": "`Button Z`", "Classic/Triggers/L": "`Button L`", "Classic/Triggers/R": "`Button R`",
          "Classic/Left Stick/Up": "`Axis MAIN Y +`", "Classic/Left Stick/Down": "`Axis MAIN Y -`", "Classic/Left Stick/Left": "`Axis MAIN X -`", "Classic/Left Stick/Right": "`Axis MAIN X +`",
          "Classic/Right Stick/Up": "`Axis C Y +`", "Classic/Right Stick/Down": "`Axis C Y -`", "Classic/Right Stick/Left": "`Axis C X -`", "Classic/Right Stick/Right": "`Axis C X +`"}
    (ud / "Config/WiimoteNew.ini").write_text("[Wiimote1]\nSource = 1\nDevice = Pipe/0/pad\nExtension = Classic\n" + "".join(f"{k} = {v}\n" for k, v in cc.items()))
print("wiimote mode:", mode)
shutil.copy(SLOT, ud / "StateSaves/NACE01.s01")
inst = DolphinInstance(ud)
ram = attach_n64_ram(inst.proc.pid)
k = HOTKEY_MAP["Load State/Load State Slot 1"]
for _ in range(40):
    if be_s16(ram, 0x801C8D60 + 0xA4) in (52, 85): break
    inst.hk.send(f"PRESS {k}"); time.sleep(0.15); inst.hk.send(f"RELEASE {k}"); time.sleep(0.35)
time.sleep(1)
def inputs(): return [ram[(0x801C8D60 + 0x14 + i*0x14) & 0x3FFFFF:(0x801C8D60 + 0x18 + i*0x14) & 0x3FFFFF].tobytes().hex() for i in range(4)]
print("idle     :", inputs())
inst.pad.send("SET MAIN 1.0 0.5", "PRESS A"); time.sleep(0.3)
print("right + A:", inputs())
inst.pad.send("SET MAIN 0.5 0.5", "RELEASE A")
p = be_u32(ram, 0x801C8D60 + 0x1C44); p0 = struct.unpack_from(">fff", ram, (p & 0x3FFFFF) + 0x24)
inst.pad.send("SET MAIN 0.5 1.0"); time.sleep(1.0); inst.pad.send("SET MAIN 0.5 0.5")
print("Link moved:", p0, "->", struct.unpack_from(">fff", ram, (p & 0x3FFFFF) + 0x24))
# Also dump the SI-related strings from the savestate for clues
inst.stop()
