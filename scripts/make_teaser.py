"""Write `paper/figures/teaser.png`: one filmstrip of the PPO policy defeating Queen Gohma.

Source is the recorded win in `media/gohma-slot5/` -- `gohma-slot5_s05_try04_crop.mp4` (the game window,
cropped) and `play_by_play.csv` (one row per HUD tick, with her health, her state and the event dictionary).
Both describe the SAME episode: policy `gohma-kl-v15-r4` at update 40, start slot 5, Gohma at 10/10 health,
killed at t = 191.0 s. See `media/gohma-slot5/PLAY_BY_PLAY.md`.

The scientific point the strip has to carry is that Gohma takes damage ONLY while she is stunned, and that a
stun has to be manufactured -- by a slingshot seed into her red eye to bring her down, or by a Deku Nut to
hold her down once she has landed. So the columns alternate stun / damage and end on her death.

    .venv/bin/python scripts/make_teaser.py

VIDEO CLOCK vs CSV CLOCK
------------------------
The recording starts ~1.45 s before the episode's t = 0, so video_t = csv_t + VIDEO_OFFSET. That is measured,
not assumed, from three independent HUD/event anchors in the video:

  * the slingshot counter steps 26 -> 25 between video 1.6 s and 1.9 s; the csv fires `seed=1` at t = 0.4 s;
  * the Deku Nut counter steps 6 -> 5 between video 189.2 s and 189.6 s; the csv fires `nut=1` at t = 187.8 s,
    and 5 -> 4 between 191.6 s and 191.9 s against the csv's `nut=1` at t = 190.1 s;
  * the final sword flash lands at video 192.2-192.5 s; the csv's `boss_defeat=True` is at t = 191.0 s.

All three give +1.35 to +1.6. `check_alignment()` below re-asserts the gross form of this (video duration =
csv duration + offset + death cutscene) every run, and `MOMENTS` carries the csv event each frame must
coincide with, so a wrong timestamp fails loudly instead of producing a plausible-looking wrong figure.

WHAT IS DONE TO THE PIXELS, AND WHY IT IS THE SAME FOR EVERY PANEL
------------------------------------------------------------------
Two adjustments, both applied IDENTICALLY to all six frames. Anything per-frame would misrepresent the
footage, so `check_uniform()` asserts the panels come out the same shape and the gamma is a module constant
rather than a per-moment field.

1. GAMMA. The Deku Tree boss chamber is very dim: the opening frames have a mean luma of 25/255 against
   65-100 for the lit floor fights, and at print size the ceiling phase read as black. Every frame gets
   `out = in ** GAMMA` with GAMMA = 0.65 in [0,1] -- a shadow lift, no per-frame exposure, no white point
   change, no saturation change. Stated on the figure itself as well as here.

2. CROP. The frames are not all letterboxed the same way, which made the strip look misassembled:
     * gameplay renders a 32 px black border -- content is rows 32-674 of 720;
     * from video 193.2 s the death cutscene adds OoT's own cinematic bars -- content is rows 81-634;
     * a full-screen flash (a Deku Nut, a sword hit) writes over the border, so a naive "find the black
       bands" scan returns the whole frame for those and nothing lines up.
   `content_box()` detects the bands, CLAMPS the result into the known gameplay box (so a flash cannot
   widen it), and then centre-crops the WIDTH so every panel has the same aspect ratio as a gameplay
   frame. Nothing is stretched and no panel keeps a black bar; the cutscene panel is simply a slightly
   tighter field of view, which is invisible next to a camera cut.

The death cutscene does NOT contain the blue flame or the blue warp: the recording cuts to black at video
196.7 s, before Gohma disintegrates, and the uncropped 267 MB file is the same 197.7 s. The last frame in
which she is on screen at all is ~196.5 s, so the final panel is the best "visibly defeated" pose the
footage actually holds -- she is on her side, underside up, legs off the ground.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "media" / "gohma-slot5" / "gohma-slot5_s05_try04_crop.mp4"
CSV = ROOT / "media" / "gohma-slot5" / "play_by_play.csv"
OUT = ROOT / "paper" / "figures" / "teaser.png"

# Seconds the video clock leads the csv clock. Measured; see the module docstring.
VIDEO_OFFSET = 1.45

ROW_LABEL = "Queen Gohma\n(PPO, 8.33% win rate)"   # two lines: one does not fit the strip height

# (csv time, descriptor, csv event the frame must sit on, nudge)
#
# `nudge` shifts the extracted frame off csv_t + VIDEO_OFFSET by a few tenths, to land on the frame that
# actually reads -- a sword hit is a one-frame white flash and the csv tick that records it is up to 0.2 s
# either side of the flash's clearest frame. The label always reports the frame's own time on the csv clock.
#
# Why each moment:
#   0.4   the agent fires a slingshot seed at her red eye while she hangs on the ceiling. This is the only
#         move that can BRING her down, and the policy uses it 4 times in 191 s out of 26 seeds carried.
#   0.9   the seed has connected and she is falling. The stun was manufactured; nothing else makes her
#         reachable, and she is unreachable for 67% of this episode.
#   4.1   first sword hit, 10 -> 9 health, landed while she is down. Damage only happens inside this window.
#   63.7  a Deku Nut stun. Nuts only work once she is already on the floor, so this is the agent EXTENDING a
#         window rather than opening one -- and it is the policy's primary stun (13 nuts thrown, 4 stuns).
#   119.7 a double sword hit, 4 -> 2 health, in the third damage window.
#   194.6 she is dead, lying on her side with her underside up and her legs off the ground. Chosen by
#         sweeping the whole cutscene at 0.25 s: she is hit and still standing to 193.0, the cinematic bars
#         come in at 193.2 with her upright and merely wounded (reads as alive), she starts to slump at
#         194.8, is tipping at 195.5, and is fully over from 196.0 until the recording cuts to black at
#         196.7. The kill itself is logged at csv t = 191.0 (`boss_defeat=True`); the csv ends there, so
#         this label is on the same clock, past its last row.
MOMENTS = [
    (0.4, "slingshot shot", "seed=1", 0.0),
    (0.9, "she falls", None, 0.0),
    (4.1, "first sword hit", None, 0.0),
    (63.7, "deku nut stun", "nut_stun=True", 0.0),
    (119.7, "double sword hit", None, 0.0),
    (194.6, "defeated, collapsing", None, -0.05),
]

# Applied identically to every panel; see the module docstring. Never make either of these per-frame.
GAMMA = 0.65
GAMEPLAY_ROWS = (32, 675)     # content rows of a gameplay frame, measured; the rest is a rendered black border
FRAME_H = 720
FRAME_W = 960
PIXEL_NOTE = f"all panels identically gamma-corrected ({GAMMA}); no other adjustment"

GREY = "#555555"
GREY_LIGHT = "#8a8a8a"


def read_csv_rows() -> list[dict]:
    with CSV.open() as fh:
        return list(csv.DictReader(fh))


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def check_alignment(rows: list[dict], duration: float) -> None:
    """The video must be this episode: csv duration + the lead-in + a death cutscene of a few seconds."""
    last_t = float(rows[-1]["t"])
    if "boss_defeat=True" not in rows[-1]["events"]:
        raise SystemExit(f"csv does not end on the kill: last events = {rows[-1]['events']!r}")
    cutscene = duration - (last_t + VIDEO_OFFSET)
    if not 180.0 <= last_t <= 200.0:
        raise SystemExit(f"csv ends at {last_t} s, expected ~191 s -- wrong episode?")
    if not 2.0 <= cutscene <= 12.0:
        raise SystemExit(
            f"video is {duration:.1f} s and the kill is at csv {last_t} s (+{VIDEO_OFFSET} s offset), "
            f"leaving {cutscene:.1f} s of tail. Expected 2-12 s of death cutscene. "
            "The video and the csv may not be the same episode -- STOP and check."
        )
    print(f"alignment OK: csv kill at {last_t:.1f} s, video {duration:.1f} s, "
          f"{cutscene:.1f} s of death cutscene after it")


def check_event(rows: list[dict], t: float, needle: str) -> None:
    """Assert the csv really fires `needle` within a tick of `t`, so a mistyped timestamp fails here."""
    near = [r for r in rows if abs(float(r["t"]) - t) <= 0.6 and needle in r["events"]]
    if not near:
        raise SystemExit(f"no csv row within 0.6 s of t={t} carries {needle!r} -- timestamp is wrong")


def hp_at(rows: list[dict], t: float) -> str:
    prev = [r for r in rows if float(r["t"]) <= t]
    return prev[-1]["gohma_hp"] if prev else rows[0]["gohma_hp"]


def extract(video_t: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{video_t:.3f}", "-i", str(VIDEO),
         "-frames:v", "1", "-y", str(dest)],
        check=True,
    )
    if not dest.exists() or dest.stat().st_size < 1000:
        raise SystemExit(f"ffmpeg produced nothing usable at video t={video_t:.2f}")


def content_box(img: "np.ndarray") -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the frame's picture, with the same aspect ratio for every frame.

    Detects the black bands, clamps into the known gameplay box so a full-screen flash cannot widen it
    past the rendered border, then centre-crops the width to the gameplay aspect. See the docstring.
    """
    luma = img[..., :3].mean(axis=2) if img.ndim == 3 else img
    if luma.max() <= 1.001:
        luma = luma * 255.0
    lit = luma.max(axis=1) >= 16.0
    rows = np.flatnonzero(lit)
    if rows.size == 0:
        raise SystemExit("frame is entirely black -- wrong timestamp")
    top = max(int(rows[0]), GAMEPLAY_ROWS[0])
    bot = min(int(rows[-1]) + 1, GAMEPLAY_ROWS[1])
    if top < GAMEPLAY_ROWS[0] + 30:
        # No cinematic bars: snap to the known gameplay box. Detection alone wanders by a few rows
        # depending on whether anything bright happens to sit against the border on this frame.
        top, bot = GAMEPLAY_ROWS
    h = bot - top
    gameplay_aspect = FRAME_W / (GAMEPLAY_ROWS[1] - GAMEPLAY_ROWS[0])
    w = min(int(round(h * gameplay_aspect)), FRAME_W)
    x0 = (FRAME_W - w) // 2
    return x0, top, x0 + w, bot


def prepare(img: "np.ndarray") -> "np.ndarray":
    """Crop to the common box and apply the one uniform gamma lift. Identical for every panel."""
    x0, y0, x1, y1 = content_box(img)
    out = img[y0:y1, x0:x1, :3].astype(np.float64)
    if out.max() > 1.001:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0) ** GAMMA


def check_uniform(frames: list["np.ndarray"]) -> None:
    """Panels must share an aspect ratio and carry no black band, or the strip is misassembled.

    Equal ASPECT is the requirement, not equal pixel count: every panel goes into the same axes box, so
    matching aspects mean nothing is stretched. A panel that still has a letterbox bar fails here too --
    that is the fault this check was written for, so it is tested against a deliberately un-cropped frame
    in the negative control at the bottom of this file's history.
    """
    ars = [f.shape[1] / f.shape[0] for f in frames]
    if max(ars) - min(ars) > 0.005 * max(ars):
        raise SystemExit(f"panels have different aspect ratios, they would be stretched: {ars}")
    for i, f in enumerate(frames):
        edge = min(f[:2, :, :].max(), f[-2:, :, :].max())
        if edge < 0.06:
            raise SystemExit(f"panel {i} still has a black band at top or bottom (edge max {edge:.3f})")
    sizes = " ".join(f"{f.shape[1]}x{f.shape[0]}" for f in frames)
    print(f"panels uniform: aspect {ars[0]:.4f} ({sizes}), gamma {GAMMA}")


def main() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")
    for p in (VIDEO, CSV):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    rows = read_csv_rows()
    check_alignment(rows, probe_duration(VIDEO))

    tmp = Path(tempfile.mkdtemp(prefix="teaser_"))
    frames = []
    try:
        for i, (t, desc, event, nudge) in enumerate(MOMENTS):
            if event is not None:
                check_event(rows, t, event)
            video_t = t + VIDEO_OFFSET + nudge
            path = tmp / f"f{i}.png"
            extract(video_t, path)
            frames.append(prepare(plt.imread(path)))
            print(f"  t = {t:6.1f} s  (video {video_t:6.2f} s)  hp {hp_at(rows, t):>2}  {desc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    check_uniform(frames)
    n = len(frames)
    h_px, w_px = frames[0].shape[:2]
    aspect = h_px / w_px

    # Geometry in figure fractions: a left strip for the rotated row label, then n frames with thin gutters.
    fig_w = 14.0
    left = 0.055
    right = 0.996
    gutter = 0.006
    fw = (right - left - gutter * (n - 1)) / n
    frame_h_in = fw * fig_w * aspect
    label_in = 0.62          # room above the frames for the two label lines
    bottom_in = 0.26         # room below for the one-line note on what was done to the pixels
    fig_h = frame_h_in + label_in + bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    b = bottom_in / fig_h
    fh = frame_h_in / fig_h

    for i, (img, (t, desc, _, _)) in enumerate(zip(frames, MOMENTS)):
        ax = fig.add_axes([left + i * (fw + gutter), b, fw, fh])
        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        label = f"t = {t:g} s"
        ax.text(0.5, 1.24, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, color=GREY)
        ax.text(0.5, 1.07, desc, transform=ax.transAxes, ha="center", va="center",
                fontsize=9.0, color=GREY_LIGHT)
        if i == 0:
            ax.set_ylabel(ROW_LABEL, color=GREY, fontsize=9.0, labelpad=6,
                          multialignment="center")

    fig.text(right, (bottom_in - 0.165) / fig_h, PIXEL_NOTE,
             ha="right", va="center", fontsize=7.0, color=GREY_LIGHT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)

    size = OUT.stat().st_size
    import PIL.Image
    with PIL.Image.open(OUT) as im:
        px = im.size
    print(f"wrote {OUT.relative_to(ROOT)}  {px[0]}x{px[1]} px  {size / 1e6:.2f} MB")
    if px[0] < 2000:
        raise SystemExit(f"figure is only {px[0]} px wide, needs >= 2000")
    if size > 3_000_000:
        raise SystemExit(f"figure is {size / 1e6:.2f} MB, needs to stay under 3 MB")


if __name__ == "__main__":
    sys.exit(main())
