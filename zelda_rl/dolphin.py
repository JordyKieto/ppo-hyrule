"""Launch isolated Dolphin instances and drive them via named pipes.

Each instance gets its own user dir (config, NAND copy, pipes, frame dumps) so
many can run side by side without touching the user's main Dolphin profile.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

DOLPHIN_BIN = os.environ.get("DOLPHIN_BIN", "/Applications/Dolphin.app/Contents/MacOS/Dolphin")
HOST_USER_DIR = Path(os.environ.get("DOLPHIN_HOST_USER_DIR", Path.home() / "Library/Application Support/Dolphin"))
OOT_TITLE_ID = "000100014e414345"  # NACE: Ocarina of Time (USA) Virtual Console

GC_BUTTONS = ["A", "B", "X", "Y", "Z", "START", "L", "R", "D_UP", "D_DOWN", "D_LEFT", "D_RIGHT"]

# Hotkeys are driven by a second pipe; each maps to a pipe "button" (12 available).
HOTKEY_MAP = {
    "Load State/Load State Slot 1": "A",
    "Load State/Load State Slot 2": "B",
    "Load State/Load State Slot 3": "X",
    "Load State/Load State Slot 4": "Y",
    "Load State/Load State Slot 5": "Z",
    "Load State/Load State Slot 6": "D_UP",
    "Load State/Load State Slot 7": "D_DOWN",
    "Load State/Load State Slot 8": "D_LEFT",
    "Load State/Load State Slot 9": "D_RIGHT",
    "Load State/Load State Slot 10": "R",
    "Save State/Save State Slot 1": "L",
    "General/Toggle Pause": "START",
}


def _ini(sections: dict[str, dict[str, str]]) -> str:
    out = []
    for sec, kv in sections.items():
        out.append(f"[{sec}]")
        out += [f"{k} = {v}" for k, v in kv.items()]
    return "\n".join(out) + "\n"


def build_user_dir(root: Path, *, dump_frames: bool = True, internal_res: int = 1,
                   unlimited_speed: bool = False, audio: bool = False,
                   fullscreen: bool = False, volume: int = 100) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = root / "Config"
    cfg.mkdir(exist_ok=True)
    if not (root / "Wii").exists():
        shutil.copytree(HOST_USER_DIR / "Wii", root / "Wii")
    for sub in ("Pipes", "Dump/Frames", "StateSaves", "ScreenShots"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in ("pad", "hk"):
        p = root / "Pipes" / name
        if not (p.exists() and stat.S_ISFIFO(p.stat().st_mode)):
            p.unlink(missing_ok=True)
            os.mkfifo(p)

    (cfg / "Dolphin.ini").write_text(_ini({
        "Core": {
            "SIDevice0": "6", "SIDevice1": "0", "SIDevice2": "0", "SIDevice3": "0",
            "EmulationSpeed": "0.0" if unlimited_speed else "1.0",
            "CPUThread": "True",
        },
        # Audio off for training (no audio device on pods, less CPU); Cubeb (the macOS default) for watching.
        "DSP": {"Backend": "Cubeb" if audio else "No Audio Output",
                "Volume": str(max(0, min(100, int(volume)))), "EnableJIT": "True"},
        "Input": {"BackgroundInput": "True"},
        "Interface": {"ConfirmStop": "False", "HotkeysRequireFocus": "False", "UsePanicHandlers": "False",
                      "OnScreenDisplayMessages": "False"},
        "Movie": {"DumpFrames": str(dump_frames), "DumpFramesSilent": "True"},
        # HotkeysRequireFocus lives in [General] in 2603a (kept in [Interface] too); without it hotkeys only reach
        # the focused Dolphin window, which breaks savestate loads for every other instance.
        "General": {"HotkeysRequireFocus": "False"},
        "Analytics": {"PermissionAsked": "True", "Enabled": "False"},
        "AutoUpdate": {"UpdateTrack": ""},
        "Display": {"RenderWindowAutoSize": "False", "RenderWindowWidth": "640",
                    "RenderWindowHeight": "480", "Fullscreen": str(fullscreen)},
    }))
    (cfg / "GFX.ini").write_text(_ini({
        "Settings": {
            "InternalResolution": str(internal_res),
            "DumpFramesAsImages": "True",
            "PNGCompressionLevel": "0",
            "ShowFPS": "False",
        },
        "Hardware": {"VSync": "False"},
    }))
    pad = {"Device": "Pipe/0/pad"}
    for b, key in [("A", "A"), ("B", "B"), ("X", "X"), ("Y", "Y"), ("Z", "Z"), ("Start", "START")]:
        pad[f"Buttons/{b}"] = f"`Button {key}`"
    for d, t in [("Up", "Y +"), ("Down", "Y -"), ("Left", "X -"), ("Right", "X +")]:
        pad[f"Main Stick/{d}"] = f"`Axis MAIN {t}`"
        pad[f"C-Stick/{d}"] = f"`Axis C {t}`"
    pad["Triggers/L"] = "`Button L`"
    pad["Triggers/R"] = "`Button R`"
    for d in ("Up", "Down", "Left", "Right"):
        pad[f"D-Pad/{d}"] = f"`Button D_{d.upper()}`"
    (cfg / "GCPadNew.ini").write_text(_ini({"GCPad1": pad}))
    hk = {"Device": "Pipe/0/hk"}
    hk.update({k: f"`Button {v}`" for k, v in HOTKEY_MAP.items()})
    (cfg / "Hotkeys.ini").write_text(_ini({"Hotkeys": hk}))
    return root


class Pipe:
    """Writer for a Dolphin pipe input device (PRESS/RELEASE/SET commands)."""

    def __init__(self, path: Path):
        # O_RDWR on a FIFO never blocks on open and survives the reader restarting.
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)

    def send(self, *cmds: str) -> None:
        os.write(self.fd, ("\n".join(cmds) + "\n").encode())

    def close(self):
        os.close(self.fd)


class DolphinInstance:
    def __init__(self, user_dir: Path, *, title_id: str = OOT_TITLE_ID, savestate: Path | None = None,
                 video_backend: str = "Metal", extra: list[str] | None = None):
        self.user_dir = user_dir
        cmd = [DOLPHIN_BIN, "-u", str(user_dir), "-b", "-n", title_id, "-v", video_backend]
        if savestate:
            cmd += ["-s", str(savestate)]
        cmd += extra or []
        self.log = open(user_dir / "dolphin_stdout.log", "w")
        env = dict(os.environ)
        if sys.platform.startswith("linux") and not env.get("DISPLAY"):
            # Verified on RunPod: Qt offscreen starts DolphinQt but never boots the game. Run under xvfb-run instead.
            raise RuntimeError("Linux Dolphin needs an X display: run the trainer under `xvfb-run -a`")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.pad = Pipe(user_dir / "Pipes/pad")
        self.hk = Pipe(user_dir / "Pipes/hk")

    def tap(self, button: str, hold: float = 0.1, pipe: Pipe | None = None):
        pipe = pipe or self.pad
        pipe.send(f"PRESS {button}")
        time.sleep(hold)
        pipe.send(f"RELEASE {button}")

    def hotkey(self, name: str, hold: float = 0.1):
        self.tap(HOTKEY_MAP[name], hold, self.hk)

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
