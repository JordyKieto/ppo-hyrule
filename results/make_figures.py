"""Regenerate every figure and table in paper/paper.md from committed artifacts.

    .venv/bin/python results/make_figures.py

Writes results/figures/*.png and results/tables/*.md. No number in the paper is typed by hand: if a table
here disagrees with the paper, the paper is stale. Pass --dataset to point at a different collection run.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DS = ROOT / "datasets/transitions/deku-baba/20260918-134229"
FIGURES = Path(__file__).parent / "figures"
TABLES = Path(__file__).parent / "tables"
# One colour per competence tier, dark -> light with training progress.
COLORS = {5: "#7f2704", 30: "#a63603", 90: "#e6550d", 110: "#fd8d3c", 130: "#fdbe85", 150: "#2171b5"}
ENDS = ["success", "death", "left_room", "timeout"]


def load(ds: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((ds / "dataset.json").read_text())
    rows = [json.loads(l) for l in (ds / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return meta, rows


def by_checkpoint(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["global_update"]].append(r)
    return dict(sorted(out.items()))


def table_dataset(meta: dict, rows: list[dict]) -> str:
    """paper.md Table 2: dataset composition per competence tier."""
    g = by_checkpoint(rows)
    out = ["| Checkpoint | Episodes | Transitions | Success | Death | Timeout | Mean episode steps |",
           "|---|---|---|---|---|---|---|"]
    for gu, eps in g.items():
        n = len(eps)
        steps = sum(e["steps"] for e in eps)
        frac = {k: sum(e["end"] == k for e in eps) / n for k in ENDS}
        out.append(f"| G{gu} | {n} | {steps:,} | {frac['success']:.0%} | {frac['death']:.0%} | "
                   f"{frac['timeout']:.0%} | {steps / n:.0f} |")
    n = len(rows)
    steps = sum(e["steps"] for e in rows)
    frac = {k: sum(e["end"] == k for e in rows) / n for k in ENDS}
    out.append(f"| **Total** | **{n}** | **{steps:,}** | **{frac['success']:.0%}** | **{frac['death']:.0%}** | "
               f"**{frac['timeout']:.0%}** | {steps / n:.0f} |")
    return "\n".join(out)


def table_schema(meta: dict) -> str:
    """paper.md §3.1: the state encoding, straight from the dataset's own schema."""
    return "\n".join([
        "| Block | Dims | Fields |", "|---|---|---|",
        f"| Link | {meta['link_dim']} | {', '.join(meta['link_fields'])} |",
        f"| Actors | {meta['max_enemies']} x {meta['enemy_dim']} | {', '.join(meta['enemy_fields'])} |",
        f"| **Total** | **{meta['state_dim']}** | |",
    ])


def table_provenance(meta: dict, rows: list[dict], ds: Path) -> str:
    """Appendix: what produced this dataset."""
    g = by_checkpoint(rows)
    lines = ["| Property | Value |", "|---|---|",
             f"| Dataset | `{ds.relative_to(ROOT)}` |",
             f"| Episodes / transitions | {meta['episodes']} / {meta['transitions']:,} |",
             f"| Wall clock | {meta['wall_seconds'] / 60:.1f} min |",
             f"| Throughput | {meta['transitions'] / meta['wall_seconds']:.0f} steps/s |",
             f"| Decision rate | {60 // meta['vi_per_step']} Hz ({meta['vi_per_step']} VI/step) |",
             f"| Episode cap | {meta['episode_seconds']:.0f} s |",
             f"| Action space | {meta['n_actions']} discrete |",
             f"| Policy obs | v{meta['obs_version']}, {meta['obs_dim']} dims |",
             f"| Start slots | {meta['slots']} |",
             f"| Sampling | {'argmax' if meta['deterministic'] else 'sampled from the policy'} |",
             f"| Actors dropped | {sum(r['actors_dropped'] for r in rows)} |"]
    for ck in meta["checkpoints"]:
        lines.append(f"| G{ck['global_update']} | {ck['episodes']} eps, {ck['transitions']:,} trans — "
                     f"`{ck['path']}` |")
    lines.append(f"| Tiers | {len(g)} |")
    return "\n".join(lines)


def figures(meta: dict, rows: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -> tables only; pip install matplotlib for figures)")
        return []
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    FIGURES.mkdir(exist_ok=True)
    g = by_checkpoint(rows)
    gus = list(g)
    made = []

    # 1. Outcome mix and mean episode length across competence tiers.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    bottom = [0.0] * len(gus)
    for end, col in zip(ENDS, ["#2171b5", "#cb181d", "#756bb1", "#bdbdbd"]):
        vals = [sum(e["end"] == end for e in g[k]) / len(g[k]) for k in gus]
        ax1.bar(range(len(gus)), vals, bottom=bottom, label=end, color=col, width=0.72)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_xticks(range(len(gus)), [f"G{k}" for k in gus])
    ax1.set_ylabel("fraction of episodes")
    ax1.set_title("Outcome mix by competence tier")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    ax2.plot(range(len(gus)), [sum(e["steps"] for e in g[k]) / len(g[k]) for k in gus], "o-", color="#238b45")
    ax2.set_xticks(range(len(gus)), [f"G{k}" for k in gus])
    ax2.set_ylabel("mean episode steps")
    ax2.set_title("Episode length by competence tier")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_composition.png", bbox_inches="tight", dpi=200)
    made.append("dataset_composition.png")
    plt.close(fig)

    # 2. Where the transitions actually come from: episode-length distribution per tier.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k in gus:
        ax.hist([e["steps"] for e in g[k]], bins=24, histtype="step", linewidth=1.6,
                color=COLORS.get(k, "#666"), label=f"G{k}")
    ax.set_xlabel("episode length (steps)")
    ax.set_ylabel("episodes")
    ax.set_title("Episode-length distribution (the 1800-step wall is the timeout)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "episode_lengths.png", bbox_inches="tight", dpi=200)
    made.append("episode_lengths.png")
    plt.close(fig)

    # 3. Per-slot difficulty: which start states the dataset covers well.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    slots = sorted({r["slot"] for r in rows})
    ax.bar([str(s) for s in slots],
           [sum(r["end"] == "success" for r in rows if r["slot"] == s) / max(sum(r["slot"] == s for r in rows), 1)
            for s in slots], color="#2171b5", width=0.7)
    ax.set_xlabel("start slot")
    ax.set_ylabel("success rate (all tiers pooled)")
    ax.set_title("Start-state difficulty")
    fig.tight_layout()
    fig.savefig(FIGURES / "slot_difficulty.png", bbox_inches="tight", dpi=200)
    made.append("slot_difficulty.png")
    plt.close(fig)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DS))
    args = ap.parse_args()
    ds = Path(args.dataset)
    meta, rows = load(ds)
    TABLES.mkdir(exist_ok=True)
    for name, body in [("dataset_composition", table_dataset(meta, rows)),
                       ("state_schema", table_schema(meta)),
                       ("provenance", table_provenance(meta, rows, ds))]:
        (TABLES / f"{name}.md").write_text(body + "\n")
        print(f"  tables/{name}.md")
    for f in figures(meta, rows):
        print(f"  figures/{f}")
    print(f"\n{meta['episodes']} episodes, {meta['transitions']:,} transitions from {ds.name}")
    print(table_dataset(meta, rows))


if __name__ == "__main__":
    main()
