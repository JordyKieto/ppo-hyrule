"""What is actually IN the human corpus: the event chain, per episode, mined from the recorded observations.

The reward function is a guess about which events matter. The corpus is 100 wins by someone who does matter,
so the events that precede her HP dropping are evidence about the reward rather than an assumption. This
script extracts them and reports which ones correlate with damage dealt.

Everything is read from the recorded obs by FIELD NAME out of each file's own `obs_fields` (AGENTS 3: never
hand-count an offset), so it stays correct if the layout changes.

    .venv/bin/python scripts/mine_corpus.py                 # aggregate + correlations
    .venv/bin/python scripts/mine_corpus.py --per-episode   # one row per episode too
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def offsets(z) -> dict:
    f = json.loads(str(z["obs_fields"]))
    out, off = {}, 0
    for name, w in f["global"]:
        out[name] = (off, w)
        off += w
    return out


def col(obs, idx, name, k=0):
    o, w = idx[name]
    return obs[:, o + k]


def resets(x, lo=0.05):
    """Count an EVENT from a recency field: `since_last_*` is normalised 0..1 and snaps to ~0 when the thing
    happens, so a downward step into ~0 is one occurrence.

    This replaces counting `last_impact_present`, which is STICKY -- it is true for 99.6% of an episode after
    the first seed lands, so anything conditioned on it reads ~100% and means nothing. That mistake was in
    the first version of this script and produced a "98.4% of hits follow a seed impact" figure that was
    true by construction (AGENTS 1: check whether the signal is free before believing it).
    """
    b = (x <= lo).astype(np.int8)
    return int(((b[1:] == 1) & (b[:-1] == 0)).sum())


def rising(x, thresh=0.5):
    """Count 0->1 transitions: an EVENT, not a state. A state that is true for 40 frames is one event."""
    b = (x > thresh).astype(np.int8)
    return int(((b[1:] == 1) & (b[:-1] == 0)).sum())


def drops(x, eps=1e-6):
    """Indices where a monotone-ish quantity falls: HP lost = damage dealt."""
    d = x[:-1] - x[1:]
    return np.where(d > eps)[0] + 1


def episode_events(f: Path) -> dict:
    z = np.load(f, allow_pickle=False)
    obs, act = z["obs"], z["action"]
    idx = offsets(z)
    g_hp = col(obs, idx, "gohma_hp_frac")
    l_hp = col(obs, idx, "link_hp_frac")
    dist = col(obs, idx, "gohma_eye_dist")
    hit_i = drops(g_hp)                                   # her HP fell: a damage event
    hurt_i = drops(l_hp)                                  # Link's HP fell
    stun = col(obs, idx, "gohma_stunned")
    vuln = col(obs, idx, "gohma_vulnerable")
    imp = col(obs, idx, "last_impact_present")
    e = {
        "file": f.name, "slot": int(z["slot"]), "steps": len(act),
        "damage": int(z["boss_damage"]), "win": bool(z["boss_defeated"]),
        "start_state": str(z["start_gohma_state"].item() if z["start_gohma_state"].shape == () else ""),
        "boss_hits": len(hit_i),
        "stuns": rising(stun), "vulnerable_onsets": rising(vuln),
        "seed_impacts": resets(col(obs, idx, "since_last_impact")) if "since_last_impact" in idx else -1,
        "seeds_fired": resets(col(obs, idx, "since_last_seed")) if "since_last_seed" in idx else -1,
        "nuts_thrown": resets(col(obs, idx, "since_last_nut")) if "since_last_nut" in idx else -1,
        "hurt_events": len(hurt_i), "hp_lost": float(l_hp[0] - l_hp[-1]),
        "ztarget_frac": float((col(obs, idx, "locked_on_gohma") > 0.5).mean()),
        "shield_frac": float((col(obs, idx, "shielding") > 0.5).mean()),
        "slingshot_frac": float((col(obs, idx, "held_slingshot") > 0.5).mean()),
        "dist_mean": float(dist.mean() * 1000.0),          # the field is normalised by 1000 units
        "dist_at_hit": float(dist[hit_i].mean() * 1000.0) if len(hit_i) else np.nan,
        "ceiling_frac": float((col(obs, idx, "gohma_on_ceiling") > 0.5).mean()),
        "first_hit_step": int(hit_i[0]) if len(hit_i) else -1,
    }
    # larva HP drops: did the human bother with the larvae at all?
    lk = 0
    for n in ("larva0_hp", "larva1_hp", "larva2_hp"):
        if n in idx:
            lk += len(drops(col(obs, idx, n)))
    e["larva_hits"] = lk
    # THE CHAIN: was she stunned or vulnerable in the 20 steps (2 s) before each of her HP drops?
    win = 20
    after_stun = sum(1 for i in hit_i if (stun[max(0, i - win):i] > 0.5).any())
    after_vuln = sum(1 for i in hit_i if (vuln[max(0, i - win):i] > 0.5).any())
    rec = col(obs, idx, "since_last_impact") if "since_last_impact" in idx else None
    after_imp = sum(1 for i in hit_i if rec is not None and (rec[max(0, i - win):i] <= 0.05).any())
    e["hits_after_stun"] = after_stun
    e["hits_after_vuln"] = after_vuln
    e["hits_after_impact"] = after_imp
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/human/gohma")
    ap.add_argument("--per-episode", action="store_true")
    ap.add_argument("--out", default="results/corpus_events.json")
    a = ap.parse_args()
    files = sorted(Path(a.data).glob("ep*.npz"))
    if not files:
        print("no episodes")
        return 1
    rows = [episode_events(f) for f in files]
    n = len(rows)
    print(f"{n} episodes, {sum(r['steps'] for r in rows):,} steps\n")

    if a.per_episode:
        print(f"{'file':<26}{'slot':>5}{'dmg':>5}{'hits':>5}{'stun':>5}{'imp':>5}{'larv':>5}"
              f"{'hurt':>5}{'Z%':>5}{'sl%':>5}{'dist':>7}")
        for r in rows:
            print(f"{r['file']:<26}{r['slot']:>5}{r['damage']:>5}{r['boss_hits']:>5}{r['stuns']:>5}"
                  f"{r['seed_impacts']:>5}{r['larva_hits']:>5}{r['hurt_events']:>5}"
                  f"{100*r['ztarget_frac']:>5.0f}{100*r['slingshot_frac']:>5.0f}{r['dist_mean']:>7.0f}")
        print()

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")
    print("PER-EPISODE MEANS")
    for k in ("damage", "boss_hits", "stuns", "vulnerable_onsets", "seed_impacts", "larva_hits",
              "hurt_events", "hp_lost", "steps", "first_hit_step", "seeds_fired", "nuts_thrown"):
        print(f"  {k:<20}{m(k):>9.2f}")
    print("  time fraction:  "
          f"Z-target {100*m('ztarget_frac'):.0f}%  slingshot held {100*m('slingshot_frac'):.0f}%  "
          f"shield {100*m('shield_frac'):.0f}%  she is on the ceiling {100*m('ceiling_frac'):.0f}%")
    print(f"  mean distance to her eye {m('dist_mean'):.0f}u | at the moment she takes damage "
          f"{m('dist_at_hit'):.0f}u")

    print("\nTHE CHAIN -- what precedes her losing HP (within 2 s)")
    H = sum(r["boss_hits"] for r in rows)
    for k, label in (("hits_after_stun", "she was STUNNED"), ("hits_after_vuln", "she was VULNERABLE"),
                     ("hits_after_impact", "a seed IMPACT landed")):
        v = sum(r[k] for r in rows)
        print(f"  {label:<26}{v:>6} of {H} hits  ({100*v/max(H,1):.1f}%)")

    print("\nWHAT CORRELATES WITH DAMAGE DEALT (Pearson r over episodes)")
    dmg = np.array([r["damage"] for r in rows], float)
    for k in ("boss_hits", "stuns", "vulnerable_onsets", "seed_impacts", "seeds_fired", "nuts_thrown",
              "larva_hits", "hurt_events", "ztarget_frac", "slingshot_frac", "shield_frac", "dist_mean",
              "steps", "ceiling_frac"):
        v = np.array([r[k] if r[k] == r[k] else 0.0 for r in rows], float)
        r_ = float(np.corrcoef(dmg, v)[0, 1]) if v.std() > 0 else float("nan")
        print(f"  {k:<20}{r_:+.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"\nper-episode events -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
