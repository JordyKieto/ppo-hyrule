"""Top-down (x/z) trajectory plot: where episodes went relative to the goal and the rupees."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLOT_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#8c564b", 5: "#d62728"}


def plot_trajectories(episodes: list[dict], goal_pos, rupee_items, success_radius: float, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    for v, p in rupee_items:
        ax.scatter(p[0], p[2], marker="D", s=18 + 4 * min(v, 20), c="#17becf", edgecolors="k", linewidths=0.4, zorder=2)
    for ep in episodes:
        xs, zs = ep["traj_x"], ep["traj_z"]
        c = SLOT_COLORS.get(ep["slot"], "gray")
        ax.plot(xs, zs, c=c, alpha=0.55, lw=1.2, zorder=3)
        ax.scatter(xs[0], zs[0], c=c, marker="o", s=30, edgecolors="k", zorder=4)
        ax.scatter(xs[-1], zs[-1], c=c, marker="x", s=40, zorder=4)
    ax.add_patch(plt.Circle((goal_pos[0], goal_pos[2]), success_radius, color="#d62728", alpha=0.15, zorder=1))
    ax.scatter(goal_pos[0], goal_pos[2], marker="*", s=260, c="#d62728", edgecolors="k", zorder=5, label="goal (slot 5)")
    for s, c in SLOT_COLORS.items():
        if any(ep["slot"] == s for ep in episodes):
            ax.plot([], [], c=c, label=f"start slot {s}")
    ax.scatter([], [], marker="D", c="#17becf", edgecolors="k", label="rupee")
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
