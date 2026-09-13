"""HUD readers for OoT (644x480 frame dump coordinates, 1x internal res)."""
from __future__ import annotations

import numpy as np

# Rupee icon (lime-green gem), bottom-left of HUD: measured bbox y418-429, x59-77.
RUPEE_ICON = (slice(414, 434), slice(55, 81))


def in_gameplay(img: np.ndarray) -> bool:
    roi = img[RUPEE_ICON].astype(int)
    r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
    green = (g > 120) & (g > r + 25) & (g > b + 60)
    return int(green.sum()) > 40
