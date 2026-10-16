"""The slingshot mechanic, mined from 100 human wins: when does a fired seed LAND, and what does it buy?

Damage requires a stun (100% of 678 recorded hits) and a stun comes from a seed in the eye, so this is the
part of the fight the policy has to learn. The question a reward can act on is not "did it fire" but "what
distinguishes a fire that lands from one that misses" -- aim alignment, her state, range, first-person mode.

Events come from the recency fields resetting (`since_last_seed`, `since_last_impact`), never from the
sticky `*_present` flags, which are true for most of an episode once set.

    .venv/bin/python scripts/mine_slingshot.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.gohma import GOHMA_STATES  # noqa: E402

HIT_WINDOW = 12          # steps (1.2 s) for a fired seed to land
STUN_WINDOW = 15         # steps for a stun to follow an impact


def idxmap(z):
    f = json.loads(str(z["obs_fields"]))
    out, off = {}, 0
    for n, w in f["global"]:
        out[n] = (off, w)
        off += w
    return out


def ev(x, lo=0.05):
    b = (x <= lo).astype(np.int8)
    return np.where((b[1:] == 1) & (b[:-1] == 0))[0] + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/human/gohma")
    a = ap.parse_args()
    fires, hits, misses = [], [], []
    impacts = []
    for f in sorted(Path(a.data).glob("ep*.npz")):
        z = np.load(f, allow_pickle=False)
        o, I = z["obs"], idxmap(z)
        fire_i = ev(o[:, I["since_last_seed"][0]])
        imp_i = set(ev(o[:, I["since_last_impact"][0]]).tolist())
        st = o[:, I["gohma_state_onehot"][0]:I["gohma_state_onehot"][0] + I["gohma_state_onehot"][1]]
        state = [GOHMA_STATES[int(k)] if k < len(GOHMA_STATES) else "?" for k in st.argmax(1)]
        aim = o[:, I["link_aim_to_eye_cos_sin_elev"][0]:I["link_aim_to_eye_cos_sin_elev"][0] + 3]
        dist = o[:, I["gohma_eye_dist"][0]] * 1000.0
        fp = o[:, I["first_person_aim"][0]]
        ready = o[:, I["ready_to_fire"][0]] if "ready_to_fire" in I else np.zeros(len(o))
        stun = o[:, I["gohma_stunned"][0]]
        ghp = o[:, I["gohma_hp_frac"][0]]
        for i in fire_i:
            landed = any((i < j <= i + HIT_WINDOW) for j in imp_i)
            row = {"state": state[i], "dist": float(dist[i]), "aim_cos": float(aim[i, 0]),
                   "aim_sin": float(aim[i, 1]), "elev": float(aim[i, 2]), "fp": float(fp[i]),
                   "ready": float(ready[i]), "landed": landed}
            fires.append(row)
            (hits if landed else misses).append(row)
        for j in sorted(imp_i):
            stunned_after = bool((stun[j:j + STUN_WINDOW] > 0.5).any())
            hp_drop = float(ghp[j] - ghp[min(len(ghp) - 1, j + STUN_WINDOW + 20)])
            impacts.append({"state": state[j], "dist": float(dist[j]),
                            "stun_followed": stunned_after, "hp_taken_after": hp_drop})

    n, h = len(fires), len(hits)
    print(f"{n} seeds fired across the corpus, {h} landed within {HIT_WINDOW/10:.1f}s "
          f"-> HIT RATE {100*h/max(n,1):.1f}%\n")

    print("HER STATE WHEN THE SEED WAS FIRED  (hit rate by state)")
    byst = {}
    for r in fires:
        byst.setdefault(r["state"], []).append(r["landed"])
    for s_, v in sorted(byst.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s_:<20}{len(v):>5} fires   {100*sum(v)/len(v):>5.1f}% land")

    print("\nWHAT SEPARATES A HIT FROM A MISS (means)")
    def mm(rows, k):
        v = [r[k] for r in rows]
        return float(np.mean(v)) if v else float("nan")
    print(f"{'':<14}{'hits':>9}{'misses':>9}")
    for k, lab in (("dist", "range (u)"), ("aim_cos", "aim.cos to eye"), ("aim_sin", "aim.sin"),
                   ("elev", "elevation"), ("fp", "first-person"), ("ready", "ready_to_fire")):
        print(f"  {lab:<12}{mm(hits,k):>9.3f}{mm(misses,k):>9.3f}")

    print("\nRANGE BANDS")
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 450), (450, 10000)):
        sel = [r for r in fires if lo <= r["dist"] < hi]
        if sel:
            print(f"  {lo:>4}-{hi if hi<10000 else '+':<5}{len(sel):>5} fires   "
                  f"{100*sum(r['landed'] for r in sel)/len(sel):>5.1f}% land")

    print(f"\nWHAT AN IMPACT BUYS ({len(impacts)} impacts)")
    sf = sum(1 for r in impacts if r["stun_followed"])
    print(f"  a stun followed within {STUN_WINDOW/10:.1f}s: {sf}/{len(impacts)} ({100*sf/max(len(impacts),1):.1f}%)")
    print(f"  mean boss HP taken in the {(STUN_WINDOW+20)/10:.1f}s after an impact: "
          f"{np.mean([r['hp_taken_after'] for r in impacts])*10:.2f} of 10")
    ist = {}
    for r in impacts:
        ist.setdefault(r["state"], []).append(r["stun_followed"])
    print("  impact landed while she was in:")
    for s_, v in sorted(ist.items(), key=lambda kv: -len(kv[1])):
        print(f"    {s_:<20}{len(v):>5}   stun followed {100*sum(v)/len(v):>5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
