#!/usr/bin/env python3
"""Generate every figure in paper/figures/ from the artefacts they claim to show.

Run:  .venv/bin/python scripts/make_paper_figures.py [--only NAME]

Not to be confused with results/make_figures.py, which predates this and regenerates the
melee collection dataset's own figures and tables. Those describe the DATASET -- success
rate pooled across every checkpoint tier that contributed episodes -- not the policy the
paper reports, which is why none of them appears in the paper.

Why this exists, and why it is a script rather than four committed PNGs. A figure with no
generator cannot be re-derived: nobody can tell later which rows produced it, whether it
drifted from the table beside it, or what a changed number would do to it. Each function
below prints its source files and the numbers it drew BEFORE writing the image, so a run of
this script is itself the provenance record (AGENTS.md section 27).

Figures:
  fig2_profile   per-start profile of the best boss policy, from the 456 matched episodes
  fig3_versions  boss damage across training versions, the measurement argument of section 9
  fig4_arch      the policy architecture, drawn from the real layer widths
  fig5_anatomy   the anatomy of one recorded win, from its play-by-play

fig1 (the teaser filmstrip) comes from scripts/make_teaser.py, which reads a video rather
than a table.

RECONCILIATION, read before trusting fig3. Section 9's version table was computed while the
pods were alive, from their own training logs, pooled across FedAvg ranks. Those pods are
gone. What survives is the W&B export in results/<run_id>/history.csv, which is the
aggregator rank. This script uses the export, states the deltas, and asserts that every
conclusion the table draws still holds on it: same sign on all seven versions, same two
negative versions, same ordering.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

import torch  # noqa: F401  (imported only so --only arch can read real layer widths)

torch.set_num_threads(1)  # AGENTS.md section 28: anything that may open an env pins threads

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "figures")

# One palette for every figure, so the paper reads as one document. Chosen to stay legible
# in greyscale print and to be distinguishable under the common forms of colour blindness:
# the pairs that must separate (win/loss, damage/rate) differ in lightness, not only in hue.
INK = "#1b1b1b"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
BLUE = "#2b6ca3"
ORANGE = "#d1702f"
GREEN = "#3f7d5a"
RED = "#a33b36"
FAINT = "#e8eef3"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    mb = os.path.getsize(path) / 1e6
    print(f"  wrote {os.path.relpath(path, REPO)}  ({mb:.2f} MB)")
    return path


def _ols(ys):
    """Slope and its standard error, plus the fitted endpoints. n-2 degrees of freedom."""
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
    return a, b, se, n


def _wilson(k, n, z=1.96):
    """Wilson score interval. The normal approximation is wrong at these rates: start 8 is
    0 of 76, where it gives a zero-width interval."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------------------
# Figure 2: the per-start profile
# --------------------------------------------------------------------------------------

# Start conditions. These are properties of the savestates, not of any run, and they are the
# ordering variable the figure is about. They match the table in paper section 5.1.
STARTS = {
    7: ("stunned", 5, 38),
    5: ("ceiling, eye red", 10, 347),
    3: ("stunned", 10, 189),
    4: ("ceiling, spawning", 8, 436),
    6: ("floor, eye red", 6, 357),
    8: ("stunned", 10, 448),
}


def fig2_profile():
    src = sorted(glob.glob(os.path.join(REPO, "results/mateval300/mateval_v15_r*.jsonl")))
    rows = []
    for f in src:
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  source: {len(src)} files, {len(rows)} episodes")
    assert len(rows) == 456, f"expected the 456 matched episodes, got {len(rows)}"

    by = collections.defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)

    stats = []
    for s, v in by.items():
        n = len(v)
        wins = sum(1 for x in v if x.get("boss_defeated"))
        dmg = [x.get("boss_damage", 0.0) for x in v]
        mean = sum(dmg) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in dmg) / (n - 1))
        state, hp0, dist = STARTS[s]
        stats.append(
            dict(slot=s, n=n, wins=wins, rate=wins / n, ci=_wilson(wins, n),
                 dmg=mean, se=sd / math.sqrt(n), hp0=hp0, share=mean / hp0,
                 state=state, dist=dist)
        )
    stats.sort(key=lambda d: -d["share"])

    print("  slot  n  wins   rate     share of her starting HP")
    for d in stats:
        print(f"   {d['slot']}   {d['n']}   {d['wins']:>2}  {100*d['rate']:6.2f}%   {d['share']:.3f}")
    pooled_w = sum(d["wins"] for d in stats)
    pooled_n = sum(d["n"] for d in stats)
    bal = sum(d["dmg"] for d in stats) / len(stats)
    print(f"  pooled {pooled_w}/{pooled_n} = {100*pooled_w/pooled_n:.2f}%, balanced damage {bal:.2f}")
    assert pooled_w == 38 and pooled_n == 456

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.1), width_ratios=[1, 1])
    ys = list(range(len(stats)))[::-1]
    labels = [f"start {d['slot']}" for d in stats]
    sub = [f"{d['state']}, {d['hp0']} HP, {d['dist']} u" for d in stats]

    # left: the binary, with intervals that overlap almost everywhere. Labels sit beyond the
    # upper bound rather than above the point, or the whisker is drawn through the text.
    for y, d in zip(ys, stats):
        lo, hi = d["ci"]
        axL.plot([100 * lo, 100 * hi], [y, y], color=MUTED, lw=1.1, solid_capstyle="round", zorder=2)
        axL.plot([100 * d["rate"]], [y], "o", color=BLUE, ms=5.5, zorder=3)
        axL.text(100 * hi + 1.4, y, f"{100*d['rate']:.2f}%  ({d['wins']}/{d['n']})",
                 ha="left", va="center", fontsize=7.2, color=INK)
    axL.axvline(100 * pooled_w / pooled_n, color=RED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axL.text(100 * pooled_w / pooled_n, len(stats) - 0.55,
             f"pooled {100*pooled_w/pooled_n:.2f}%", color=RED, fontsize=7.2, va="bottom",
             ha="center")
    axL.set_xlim(-1.5, 52)
    axL.set_title("Defeat rate, with 95% Wilson intervals")
    axL.set_xlabel("episodes won (%)")

    # right: the graded quantity, which orders all six
    for y, d in zip(ys, stats):
        axR.barh(y, d["share"], height=0.52, color=FAINT, edgecolor=BLUE, linewidth=1.0, zorder=2)
        axR.plot([d["share"] - d["se"] / d["hp0"], d["share"] + d["se"] / d["hp0"]], [y, y],
                 color=INK, lw=1.0, zorder=3)
        axR.text(d["share"] + 0.025, y, f"{d['share']:.3f}", va="center", fontsize=7.4, color=INK)
    axR.set_xlim(0, 0.93)
    axR.set_title("Health removed, as a share of her starting HP")
    axR.set_xlabel("mean damage / starting HP")

    for ax in (axL, axR):
        ax.set_yticks(ys)
        ax.set_ylim(-0.8, len(stats) - 0.1)
        ax.grid(axis="y", visible=False)
        _despine(ax)
    axL.set_yticklabels([f"{a}\n{b}" for a, b in zip(labels, sub)], fontsize=7.4, color=INK)
    axR.set_yticklabels([])

    fig.tight_layout()
    return _save(fig, "profile.png")


# --------------------------------------------------------------------------------------
# Figure 3: boss damage across versions
# --------------------------------------------------------------------------------------

VERSIONS = [
    ("gohma-band-v11", "v11", ""),
    ("gohma-band-v12", "v12", ""),
    ("gohma-band-v13", "v13", ""),
    ("gohma-kl-v14", "v14", "no approach term"),
    ("gohma-kl-v15", "v15", "approach 0.5"),
    ("gohma-kl-v16", "v16", "approach 1.5"),
    ("gohma-kl-v17", "v17", "0.5 + deficit"),
]
# Section 9's table, computed at the time from the pods' own logs. Kept here so the
# reconciliation is mechanical rather than a claim in prose.
PAPER_TABLE = {"v11": (1.72, 2.10), "v12": (2.06, 2.22), "v13": (2.41, 2.75),
               "v14": (2.86, 3.22), "v15": (3.17, 3.32), "v16": (3.12, 2.85),
               "v17": (3.19, 2.86)}


def _run_dirs():
    out = {}
    for rj in glob.glob(os.path.join(REPO, "results/*/run.json")):
        try:
            g = json.load(open(rj)).get("group")
        except Exception:
            continue
        if g:
            out[g] = os.path.dirname(rj)
    return out


def fig3_versions():
    dirs = _run_dirs()
    series = []
    print("  ver  n   first3rd last3rd  slope +- SE      t     section 9 table")
    for group, label, note in VERSIONS:
        d = dirs[group]
        rows = list(csv.DictReader(open(os.path.join(d, "history.csv"))))
        dmg = [float(r["gohma/boss_damage"]) for r in rows
               if r.get("gohma/boss_damage") not in ("", "NaN", None)]
        suc = [float(r["combat/success_rate"]) for r in rows
               if r.get("combat/success_rate") not in ("", "NaN", None)]
        k = max(1, len(dmg) // 3)
        t1, t3 = sum(dmg[:k]) / k, sum(dmg[-k:]) / k
        ks = max(1, len(suc) // 3)
        s1, s3 = sum(suc[:ks]) / ks, sum(suc[-ks:]) / ks
        _, b, se, n = _ols(dmg)
        ps, pe = PAPER_TABLE[label]
        print(f"  {label}  {n:>3}   {t1:6.2f} {t3:6.2f}   {b:+.4f} +- {se:.4f}  {b/se:+5.2f}   "
              f"{ps:.2f} -> {pe:.2f}")
        # the conclusions the table draws, asserted on the surviving export
        assert (t3 - t1 > 0) == (pe - ps > 0), f"{label}: export and table disagree on direction"
        series.append(dict(label=label, note=note, dmg=dmg, suc=suc, t1=t1, t3=t3,
                           s1=s1, s3=s3, slope=b, se=se, run=os.path.basename(d)))
    neg = [s["label"] for s in series if s["t3"] < s["t1"]]
    assert neg == ["v16", "v17"], f"expected v16 and v17 to be the regressions, got {neg}"
    print(f"  regressions: {neg}  (matches section 9)")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.8), height_ratios=[2.2, 1],
                                  sharex=True)
    x0 = 0
    ticks, ticklabels = [], []
    for s in series:
        n = len(s["dmg"])
        xs = [x0 + i for i in range(n)]
        falling = s["t3"] < s["t1"]
        col = RED if falling else BLUE
        ax.plot(xs, s["dmg"], color=col, lw=0.7, alpha=0.30, zorder=2)
        # the thirds the table quotes, drawn as the table computes them
        k = max(1, n // 3)
        ax.plot([xs[0], xs[k - 1]], [s["t1"]] * 2, color=col, lw=2.6, solid_capstyle="butt", zorder=4)
        ax.plot([xs[-k], xs[-1]], [s["t3"]] * 2, color=col, lw=2.6, solid_capstyle="butt", zorder=4)
        ax.annotate("", xy=(xs[-k], s["t3"]), xytext=(xs[k - 1], s["t1"]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, alpha=0.85,
                                    shrinkA=0, shrinkB=0), zorder=3)
        sig = "*" if abs(s["slope"] / s["se"]) > 2 else ""
        top = max(s["t1"], s["t3"])
        ax.text((xs[0] + xs[-1]) / 2, top + 0.78,
                f"{s['label']}{sig}", ha="center", fontsize=8.2, color=col, fontweight="bold")
        if s["note"]:
            ax.text((xs[0] + xs[-1]) / 2, top + 0.44, s["note"],
                    ha="center", fontsize=6.4, color=MUTED)
        # the SAME estimator on the binary, so the two panels are comparable rather than
        # one being a trend line and the other a cloud
        if s["suc"]:
            sxs = xs[: len(s["suc"])]
            ax2.plot(sxs, [100 * v for v in s["suc"]], color=col, lw=0.6, alpha=0.25, zorder=2)
            kk = max(1, len(sxs) // 3)
            ax2.plot([sxs[0], sxs[kk - 1]], [100 * s["s1"]] * 2, color=col, lw=2.6,
                     solid_capstyle="butt", zorder=4)
            ax2.plot([sxs[-kk], sxs[-1]], [100 * s["s3"]] * 2, color=col, lw=2.6,
                     solid_capstyle="butt", zorder=4)
        ticks.append((xs[0] + xs[-1]) / 2)
        ticklabels.append(s["label"])
        x0 += n + 5

    ax.set_ylabel("boss damage (of 10 HP)")
    ax.set_title("The graded signal rose; the binary it was read through did not")
    ax.set_ylim(0.6, 5.3)
    ax.grid(axis="x", visible=False)
    _despine(ax)
    ax.text(0.012, 0.99, "thick bars: first-third and last-third means, the estimator section 9 quotes\n"
                         "* slope over all updates distinguishable from zero on this export",
            transform=ax.transAxes, va="top", fontsize=6.6, color=MUTED)

    ax2.set_ylabel("defeat rate (%)")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticklabels, fontsize=8)
    ax2.set_xlabel("training updates, seven versions laid end to end (gaps are version boundaries)")
    ax2.grid(axis="x", visible=False)
    _despine(ax2)

    fig.tight_layout()
    return _save(fig, "versions.png")


# --------------------------------------------------------------------------------------
# Figure 4: the architecture
# --------------------------------------------------------------------------------------

def fig4_arch():
    """Every width in this diagram is read out of the reported checkpoint's tensors.

    A diagram drawn from memory is the one artefact in a paper nobody checks, and this one
    was wrong on its first draft: it omitted the global feature block entirely, which is
    297 of the 425 numbers entering the trunk.
    """
    ck_path = os.path.join(REPO, "checkpoints/kept/gohma-kl-v15-r4_latest.pt")
    assert os.path.exists(ck_path), f"missing {ck_path}: the diagram must not be drawn from memory"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    d = dict(
        obs=cfg["obs_dim"], glob=cfg["global_dim"], n_actors=cfg["n_actors"],
        actor_dim=cfg["actor_dim"], nact=cfg["n_actions"],
        enc_in=tuple(sd["enc.actor_enc.0.weight"].shape)[1],
        enc_h=tuple(sd["enc.actor_enc.0.weight"].shape)[0],
        enc_out=tuple(sd["enc.actor_enc.2.weight"].shape)[0],
        trunk_in=tuple(sd["proj.0.weight"].shape)[1],
        gru=tuple(sd["gru.weight_hh"].shape)[1],
        head=tuple(sd["pi.2.weight"].shape)[0],
    )
    print(f"  widths read from {os.path.relpath(ck_path, REPO)}:")
    for k, v in d.items():
        print(f"    {k:10s} {v}")
    # the diagram's arithmetic, asserted rather than drawn and hoped for
    assert d["trunk_in"] == d["glob"] + 2 * d["enc_out"], "trunk input is not global + mean||max"
    assert d["head"] == d["nact"]

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(-1, 103)
    ax.set_ylim(7, 42)
    ax.axis("off")

    def box(x, y, w, h, title, sub, ec, fc=FAINT, fs=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                    facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=3))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), title, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="bold", zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.1, sub, ha="center", va="center",
                    fontsize=6.3, color=MUTED, zorder=4)

    def arrow(x1, y1, x2, y2, color=MUTED, lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=color, linewidth=lw, zorder=2))

    # ---- inputs: one global block, and a ragged table of actor rows
    ax.text(12, 40.4, f"observation v7, {d['obs']} numbers from emulator memory, never pixels",
            ha="center", fontsize=7.0, color=INK, fontweight="bold")

    box(1, 32.0, 22, 6.0, f"global block, {d['glob']}", "room, timers, Link's own state", ORANGE,
        fc="white", fs=7.2)

    ax.text(12, 29.0, f"actor table, {d['n_actors']} rows × {d['actor_dim']}", ha="center",
            fontsize=7.2, color=INK, fontweight="bold")
    rows = ["Link", "Gohma", "larva 1", "…", "empty (masked)"]
    for i, lab in enumerate(rows):
        y = 24.0 - i * 4.0
        empty = lab.startswith("empty")
        ax.add_patch(FancyBboxPatch((4, y), 16, 3.0,
                                    boxstyle="round,pad=0.2,rounding_size=0.7",
                                    facecolor="#f5f5f5" if empty else "white",
                                    edgecolor=MUTED if empty else BLUE, linewidth=0.8,
                                    linestyle=":" if empty else "-", zorder=3))
        ax.text(12, y + 1.5, lab, ha="center", va="center", fontsize=6.4,
                color=MUTED if empty else INK, zorder=4)
        if not empty:
            arrow(20.3, y + 1.5, 27.5, 18.5)

    # ---- shared encoder and pool
    box(28, 14.0, 16, 9.0, "per-actor MLP",
        f"{d['enc_in']} → {d['enc_h']} → {d['enc_out']}, shared", BLUE)
    arrow(44.3, 18.5, 47.5, 18.5)
    box(48, 14.0, 14, 9.0, "masked pool", f"mean ‖ max, {2*d['enc_out']}", BLUE)

    # ---- concatenation with the global block, which the first draft of this figure missed
    arrow(62.3, 18.5, 65.3, 21.2)
    arrow(23.3, 35.0, 65.0, 24.6, color=ORANGE)
    ax.add_patch(plt.Circle((66.8, 22.6), 1.6, facecolor="white", edgecolor=INK, lw=0.9, zorder=4))
    ax.text(66.8, 22.5, "‖", ha="center", va="center", fontsize=8, color=INK, zorder=5)
    ax.text(66.8, 26.1, f"{d['glob']} + {2*d['enc_out']} = {d['trunk_in']}", ha="center",
            fontsize=6.3, color=MUTED)
    arrow(68.6, 22.6, 72.5, 22.6)

    box(73, 18.0, 13, 9.0, f"GRU {d['gru']}", None, GREEN)
    arrow(86.3, 24.5, 89.0, 29.5)
    arrow(86.3, 20.5, 89.0, 15.0)
    box(89, 28.0, 11, 7.0, "policy", f"{d['nact']} actions", ORANGE, fs=7.4)
    box(89, 10.5, 11, 7.0, "value", "one scalar", ORANGE, fs=7.4)

    # the recurrent edge: the only reason a GRU is here rather than a linear layer
    for seg in [((79.5, 18.0), (79.5, 13.8)), ((79.5, 13.8), (73.8, 13.8))]:
        ax.add_patch(FancyArrowPatch(*seg, arrowstyle="-", color=GREEN, lw=0.8, zorder=2))
    arrow(73.8, 13.8, 73.8, 17.8, color=GREEN, lw=0.8)
    ax.text(79.5, 10.4, "hidden state, carried\nacross the episode", fontsize=6.0,
            color=GREEN, ha="center", va="top", linespacing=1.3)

    return _save(fig, "architecture.png")


# --------------------------------------------------------------------------------------
# Figure 5: the anatomy of one win
# --------------------------------------------------------------------------------------

STUNNED = {"stunned", "stunned_hit", "fall"}


def fig5_anatomy():
    src = os.path.join(REPO, "media/gohma-slot5/play_by_play.csv")
    rows = list(csv.DictReader(open(src)))
    print(f"  source: {os.path.relpath(src, REPO)}, {len(rows)} logged steps")
    t = [float(r["t"]) for r in rows]
    hp = [float(r["gohma_hp"]) for r in rows]
    hearts = [float(r["link_hearts"]) for r in rows]
    state = [r["gohma_state"] for r in rows]
    tmax = t[-1]

    hits = [(float(r["t"]), r["events"]) for r in rows if "boss_hit" in r["events"]]
    nuts = [float(r["t"]) for r in rows if "nut_stun" in r["events"]]
    seeds = [float(r["t"]) for r in rows if "seed_stun" in r["events"]]
    own = [float(r["t"]) for r in rows
           if "boss_stunned=True" in r["events"] and "nut_stun" not in r["events"]
           and "seed_stun" not in r["events"]]
    larva = [float(r["t"]) for r in rows if "larva_kill" in r["events"]]
    print(f"  {len(hits)} damage events, stuns: {len(nuts)} nut, {len(seeds)} seed, "
          f"{len(own)} her own descent; {len(larva)} larvae killed")
    print(f"  her health {hp[0]:.0f} -> {hp[-1]:.0f}, Link {hearts[0]} -> {min(hearts)} hearts at worst")

    fig, ax = plt.subplots(figsize=(7.2, 2.9))

    # the stun windows, which are the only times damage is possible
    i = 0
    first = True
    while i < len(rows):
        if state[i] in STUNNED:
            j = i
            while j + 1 < len(rows) and state[j + 1] in STUNNED:
                j += 1
            ax.axvspan(t[i], t[min(j + 1, len(t) - 1)], color="#fdf0d5", zorder=1,
                       label="she is vulnerable" if first else None)
            first = False
            i = j + 1
        else:
            i += 1

    ax.step(t, hp, where="post", color=RED, lw=1.7, zorder=4, label="Gohma health (of 10)")
    ax.step(t, hearts, where="post", color=BLUE, lw=1.2, zorder=4, label="Link hearts")

    for k, (x, _) in enumerate(hits):
        ax.plot([x], [10.6], marker="v", color=RED, ms=4.2, zorder=5,
                label="sword hit" if k == 0 else None)
    for k, x in enumerate(nuts):
        ax.plot([x], [11.5], marker="o", color=GREEN, ms=4.0, zorder=5,
                label="stun: Deku Nut" if k == 0 else None)
    for k, x in enumerate(seeds):
        ax.plot([x], [11.5], marker="s", color=ORANGE, ms=4.0, zorder=5,
                label="stun: slingshot" if k == 0 else None)
    for k, x in enumerate(own):
        ax.plot([x], [11.5], marker="x", color=MUTED, ms=4.4, mew=1.1, zorder=5,
                label="stun: her own descent" if k == 0 else None)
    for k, x in enumerate(larva):
        ax.plot([x], [-0.7], marker="|", color=INK, ms=5, mew=1.0, zorder=5,
                label="larva killed" if k == 0 else None)

    ax.annotate("Link at half a heart", xy=(166, 0.5), xytext=(126, 3.2), fontsize=6.6,
                color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.7))
    # the longest stretch with no damage dealt, which is most of the episode
    gaps = [(hits[i][0], hits[i + 1][0]) for i in range(len(hits) - 1)]
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    ax.annotate("", xy=(a, 4.6), xytext=(b, 4.6),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7))
    ax.text((a + b) / 2, 4.9, f"{b-a:.0f} s with no damage dealt", ha="center", fontsize=6.4,
            color=MUTED)
    ax.plot([tmax], [0], marker="*", color=RED, ms=9, zorder=6, clip_on=False)
    ax.text(tmax - 2, 1.0, "defeated", ha="right", fontsize=6.8, color=RED, fontweight="bold")
    ax.set_xlim(-3, tmax + 4)
    ax.set_ylim(-1.6, 12.4)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("game time (s)")
    ax.set_ylabel("health")
    ax.set_title("One win, 191 s: health falls only inside the windows the agent manufactures")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, fontsize=6.8,
              handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    return _save(fig, "anatomy.png")


FIGURES = {"fig2_profile": fig2_profile, "fig3_versions": fig3_versions,
           "fig4_arch": fig4_arch, "fig5_anatomy": fig5_anatomy}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FIGURES), help="generate one figure")
    a = ap.parse_args()
    todo = [a.only] if a.only else list(FIGURES)
    total = 0
    for name in todo:
        print(f"\n{name}")
        p = FIGURES[name]()
        total += os.path.getsize(p)
    print(f"\ntotal {total/1e6:.2f} MB in {os.path.relpath(OUT, REPO)}/")
    if total > 6e6:
        print("WARNING: over 6 MB. arXiv accepts more, but the siblings' bundles are ~5 MB.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
