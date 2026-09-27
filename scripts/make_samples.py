"""Write `samples/` excerpts and `artifacts/MANIFEST.json`, so the repo describes its own data and models.

The full datasets are 28-53 GB in S3 and the checkpoints are up to 138 MB; neither belongs in git. What belongs
is enough to see the shape of everything without a download, plus a checksum for every artefact so a file can
be identified later. Format follows ../neural-simulators.

    .venv/bin/python scripts/make_samples.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS = [
    ("rl/deku_baba_G150", "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt",
     "Deku Baba SOLVED policy, global update 150. obs v4 483d, 63 actions, GRU 256."),
    ("rl/gohma_v10_r13_u50", "checkpoints/inbox/gohma-r13_u00050.pt",
     "Gohma v10 rank 13 update 50; best of 18 ranks by late defeat rate (8.43%). The v11 starting point."),
    ("rl/gohma_v13_r12_u70", "runs/gohma_v13/v13_r12_latest.pt",
     "BEST Gohma policy to date: v13 rank 12 update 70, 7.4% defeat, 0.64% at full HP. Optimizer state intact."),
    ("wm/token_v2_ck0080", "runs/wm_v2/ck_0080.pt",
     "Token WM, teacher-forced. Best one-step skill +0.188."),
    ("wm/token_v3_ck0085", "runs/wm_v3/ck_0085.pt",
     "Token WM after autoregressive fine-tune; one-step skill went negative."),
    ("wm/jepa_v2_ck0045", "runs/jepa_v2/ck_0045.pt", "Latent JEPA stage 1, teacher-forced."),
    ("wm/jepa_v3_ck0070", "runs/jepa_v3/ck_0070.pt", "Latent JEPA stage 2, autoregressive."),
]


def md5(p: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def sample_episode():
    """One episode, with its own field names, so the state layout is legible without the loader."""
    import glob
    from zelda_rl.state_layout import ENEMY_FIELDS, N_ROWS, ROW_DIM, link_fields
    fs = sorted(glob.glob(str(ROOT / "datasets/transitions/v5/episodes/*.npz")))
    if not fs:
        return None
    src = Path(fs[len(fs) // 2])
    d = np.load(src)
    st, ac = d["state"], d["action"]
    out = ROOT / "samples/rl"
    out.mkdir(parents=True, exist_ok=True)
    n = min(120, len(ac))
    np.savez_compressed(out / "episode_excerpt.npz", state=st[:n + 1].astype(np.float32),
                        action=ac[:n].astype(np.int64), reward=d["reward"][:n].astype(np.float32))
    lf = link_fields(st.shape[1])
    (out / "episode_excerpt.json").write_text(json.dumps({
        "source_episode": src.name, "source_dataset": "s3://zelda-rl/zelda/transitions/v5/",
        "frames_in_source": int(len(ac)), "frames_in_excerpt": n,
        "layout": "state[T+1, D], action[T], reward[T]; transition t = (state[t], action[t], state[t+1])",
        "state_dim": int(st.shape[1]), "link_dim": len(lf), "n_rows": N_ROWS, "row_dim": ROW_DIM,
        "link_fields": lf, "enemy_fields": list(ENEMY_FIELDS),
        "note": "Full v5 is 91,522,473 transitions over 103,716 episodes. This is one episode's opening.",
    }, indent=2))
    return src.name, n


def sample_counterfactual():
    p = ROOT / "datasets/cf/pairs.npz"
    if not p.exists():
        return None
    d = np.load(p)
    out = ROOT / "samples/wm"
    out.mkdir(parents=True, exist_ok=True)
    k = min(20, len(d["s0"]))
    np.savez_compressed(out / "counterfactual_excerpt.npz",
                        **{n: d[n][:k] for n in ("s0", "s1", "s2", "a1", "a2", "slot")})
    (out / "counterfactual_excerpt.json").write_text(json.dumps({
        "source": "datasets/cf/pairs.npz", "pairs_in_source": int(len(d["s0"])), "pairs_in_excerpt": k,
        "lag_steps": int(d["lag"]) if "lag" in d else None,
        "layout": "s0 = state at the branch point; a1/a2 = the two move actions held `lag` steps; "
                  "s1/s2 = the two real outcomes. Built by scripts/collect_counterfactual.py.",
        "what_it_measures": "On 43% of pairs the two actions produce outcomes <0.5 world units apart -- the "
                            "game ignores the command. On the other 57% they separate by a mean of 21.4 units. "
                            "Any action-conditioning metric must be scored on the sensitive subset.",
    }, indent=2))
    return k


def main():
    art = []
    for name, rel, what in ARTIFACTS:
        p = ROOT / rel
        if not p.exists():
            art.append({"name": name, "path": rel, "present": False, "description": what})
            print(f"  MISSING  {rel}")
            continue
        art.append({"name": name, "path": rel, "present": True, "bytes": p.stat().st_size,
                    "md5": md5(p), "description": what})
        print(f"  {md5(p)[:12]}  {p.stat().st_size:>11,}  {rel}")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts/MANIFEST.json").write_text(json.dumps({
        "note": "Checkpoints are NOT committed (gitignored; up to 138 MB). This manifest identifies them by "
                "md5 so a file found later can be matched to a run. Provenance: PROVENANCE.md.",
        "artifacts": art}, indent=2))
    e = sample_episode()
    print(f"  samples/rl/episode_excerpt.npz  from {e[0]}, {e[1]} frames" if e else "  no local v5 episodes")
    c = sample_counterfactual()
    print(f"  samples/wm/counterfactual_excerpt.npz  {c} pairs" if c else "  no counterfactual set")


if __name__ == "__main__":
    main()
