"""Frame source: Dolphin's PNG frame dump dir (dedups identical frames, so ~20 fps in OoT gameplay)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

_NUM = re.compile(r"framedump_(\d+)\.png$")


class DumpDirFrames:
    def __init__(self, user_dir: Path, keep: int = 4):
        self.dir = user_dir / "Dump/Frames"
        self.keep = keep
        self.last_idx = -1

    def _indices(self) -> list[int]:
        out = []
        with __import__("os").scandir(self.dir) as it:
            for e in it:
                m = _NUM.search(e.name)
                if m:
                    out.append(int(m.group(1)))
        out.sort()
        return out

    def latest(self, gc: bool = True) -> tuple[int, np.ndarray] | None:
        """Return (index, HxWx3 uint8) for the newest fully-written frame; prunes old files."""
        idx = self._indices()
        if len(idx) < 2:
            return None
        # The newest file may still be mid-write; take the second newest.
        i = idx[-2]
        try:
            img = np.asarray(Image.open(self.dir / f"framedump_{i}.png").convert("RGB"))
        except (OSError, SyntaxError):
            return None
        if gc:
            for j in idx[: -self.keep]:
                (self.dir / f"framedump_{j}.png").unlink(missing_ok=True)
        self.last_idx = i
        return i, img
