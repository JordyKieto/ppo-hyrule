"""Reward-critical event detection from a decoded latent, with an honestly-chosen operating point.

    .venv/bin/python scripts/eval_decode_events.py checkpoints/bc/bc-decode-final.pt

The reward is computed by CombatTracker.step() from raw state, so what matters is whether the events it keys
on -- a Baba losing HP, a Baba dying, Link taking a hit -- can be recovered from a frozen BC latent.

argmax on decoded HP is an arbitrary operating point. Each detector instead produces a calibrated score, the
threshold is chosen on the TUNE slots, and everything reported is from the TEST slots at that fixed
threshold. Average precision is also reported: it is threshold-free and is the number to trust.

Detectors are compared rather than assumed, since two plausible ideas turned out not to help:
  HP-drop probability     sum over class pairs of P_t(a) P_{t+1}(b) for b < a
  hit-state entry         the Baba behaviour state decodes more accurately than HP, and 77% of post-damage
                          frames are `hit` against a 1.67% base rate
  despawn corroboration   every true kill despawns within 40 steps (164/164)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, StateDecoder, decode_targets,  # noqa: E402
                         featurise, hp_class)
from scripts.train_bc_decode import DEFAULT_DATA, load  # noqa: E402

HIT, DYING, DESPAWN_WINDOW = 6, 8, 40


def best_threshold(score, truth):
    """F1-maximising threshold. Called ONLY on the tune split."""
    if truth.sum() == 0 or not np.isfinite(score).any():
        return None
    best, bth = -1.0, None
    for th in np.unique(np.sort(score[np.isfinite(score)])[::-1][:max(50 * int(truth.sum()), 2000)]):
        p = score >= th
        tp, fp, fn = int((p & truth).sum()), int((p & ~truth).sum()), int((~p & truth).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best:
            best, bth = f1, float(th)
    return bth


def at_threshold(score, truth, th):
    """Metrics at a fixed threshold, plus threshold-free average precision."""
    p = score >= th
    tp, fp, fn = int((p & truth).sum()), int((p & ~truth).sum()), int((~p & truth).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    order = np.argsort(-score)
    t = truth[order]
    tp_c, fp_c = np.cumsum(t), np.cumsum(~t)
    prec, rec = tp_c / np.maximum(tp_c + fp_c, 1), tp_c / max(truth.sum(), 1)
    return {"f1": 2 * pr * rc / max(pr + rc, 1e-9), "precision": pr, "recall": rc, "tp": tp, "fp": fp,
            "fn": fn, "ap": float(np.sum(np.diff(np.r_[0, rec]) * prec)), "threshold": th,
            "true_events": int(truth.sum())}


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("checkpoint")
    ap_.add_argument("--data", default=DEFAULT_DATA)
    ap_.add_argument("--out", default="results/event_detection.json")
    a = ap_.parse_args()
    torch.set_num_threads(4)

    ck = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    meta, state, action, slot, epid = load(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)

    net = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=ck["n_actions"],
                latent=ck["latent"], gru=0)
    net.load_state_dict(ck["model"]); net.eval()
    dec = StateDecoder(latent=ck["latent"]); dec.load_state_dict(ck["decoder"]); dec.eval()
    norm = Norm.load(ck["norm"])
    test_slots, tune_slots = set(ck["split"]["val_slots"]), set(ck["split"].get("tune_slots", []))
    if not tune_slots:
        raise SystemExit("checkpoint has no tune_slots: retrain with --tune-slots so the operating point is "
                         "not fitted on the split being reported")

    def scores(idx):
        """-> per-event dict of detector scores plus the ground-truth event mask, for this frame subset."""
        Fn = norm.apply(featurise(state[idx]))
        P = {}
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                sl = slice(s, s + 8192)
                z = net.encode(torch.from_numpy(Fn["link"][sl]), torch.from_numpy(Fn["rows"][sl]),
                               torch.from_numpy(Fn["ids"][sl]), torch.from_numpy(Fn["mask"][sl]))
                for k, v in dec(z).items():
                    P.setdefault(k, []).append(v)
        P = {k: torch.cat(v) for k, v in P.items()}
        y, e = {k: v[idx] for k, v in Y.items()}, epid[idx]
        sm = e[1:] == e[:-1]
        p_rhp, p_lhp = P["row_hp"].softmax(-1).numpy(), P["link_hp"].softmax(-1).numpy()
        p_state, p_pres = P["row_state"].softmax(-1).numpy(), P["row_present"].sigmoid().numpy()
        baba = (y["row_is_baba"] > 0.5)[:-1][sm]

        def drop(p):
            cum = np.cumsum(p[1:], -1)
            lower = np.concatenate([np.zeros_like(cum[..., :1]), cum[..., :-1]], -1)
            return (p[:-1] * lower).sum(-1)[sm]

        n = len(idx)
        gone = np.zeros_like(p_pres)
        for t in range(n - 1, -1, -1):
            nxt = gone[t + 1] if t + 1 < n and e[t + 1] == e[t] else 0.0
            gone[t] = np.maximum(1 - p_pres[t], nxt * 0.995)
        w = np.stack([gone[min(t + DESPAWN_WINDOW, n - 1)] if e[min(t + DESPAWN_WINDOW, n - 1)] == e[t]
                      else gone[t] for t in range(n)])[1:][sm]
        s_hp_r, s_hit = drop(p_rhp), (p_state[1:, :, HIT] * (1 - p_state[:-1, :, HIT]))[sm]
        s_kill = ((1 - p_rhp[:-1, :, 0]) * p_rhp[1:, :, 0])[sm]
        tru = np.where(y["row_present"] > 0.5, y["row_hp_c"], 0)
        return {
            "damage": {"truth": (((tru[1:] - tru[:-1])[sm] < 0) & baba)[baba],
                       "HP-drop probability": s_hp_r[baba],
                       "hit-state entry": s_hit[baba],
                       "hit-entry x HP-drop": (s_hit * s_hp_r)[baba],
                       "hit-entry + HP-drop": (s_hit + s_hp_r)[baba]},
            "kill": {"truth": (((tru[:-1][sm] > 0) & (tru[1:][sm] == 0)) & baba)[baba],
                     "HP->0 probability": s_kill[baba],
                     "HP->0 x despawn": (s_kill * w)[baba],
                     "HP->0 x despawn x dying": (s_kill * w * p_state[1:, :, DYING][sm])[baba]},
            "link_damage": {"truth": (y["link_hp_c"][1:] - y["link_hp_c"][:-1])[sm] < 0,
                            "HP-drop probability": drop(p_lhp),
                            "HP-drop x sf1_damaged": drop(p_lhp) * P["link_bits"][1:, 2].sigmoid().numpy()[sm]},
        }

    S_tune = scores(np.where(np.isin(slot, list(tune_slots)))[0])
    S_test = scores(np.where(np.isin(slot, list(test_slots)))[0])

    print(f"checkpoint {a.checkpoint}")
    print(f"train {ck['split'].get('train_slots')} | tune (threshold only) {sorted(tune_slots)} | "
          f"TEST (reported) {sorted(test_slots)}\n")
    out = {"checkpoint": a.checkpoint, "split": ck["split"], "events": {}}
    for event in ("damage", "kill", "link_damage"):
        tt, te = S_tune[event], S_test[event]
        print(f"{event.upper()}   tune events {int(tt['truth'].sum())} | test events {int(te['truth'].sum())}")
        print(f"  {'detector':<26} {'F1':>6} {'prec':>6} {'rec':>6} {'AP':>6}  {'TP/FP/FN':>11}  threshold")
        print("  " + "-" * 78)
        out["events"][event] = {}
        for det in [k for k in tt if k != "truth"]:
            th = best_threshold(tt[det], tt["truth"])
            if th is None:
                continue
            r = at_threshold(te[det], te["truth"], th)
            out["events"][event][det] = r
            print(f"  {det:<26} {r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['ap']:>6.3f}  "
                  f"{r['tp']:>3}/{r['fp']:>3}/{r['fn']:<3}  {th:.4g}")
        print()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"-> {a.out}\nAP is threshold-free and is the number to trust; F1/precision/recall are at a "
          f"threshold fitted on the tune slots only.")


if __name__ == "__main__":
    main()
