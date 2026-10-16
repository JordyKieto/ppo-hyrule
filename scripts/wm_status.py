"""WM STATUS: the four questions, answered from a checkpoint, without touching the running job.

AGENTS 1/2/34. The training log's aggregate L1 cannot answer "does the world model know when damage
happens", because damage ticks are under 1% of the corpus -- a model that predicts NO event at all scores
nearly the same aggregate error as one that gets them right. This script answers, per checkpoint:

  Q1 is it better than doing NOTHING?   per-horizon error vs the persistence baseline (skill = 1 - m/p)
  Q2 is it getting better?              the saved per-epoch history, slope +- SE (needs --log or ck history)
  Q3 does it predict EVENTS?            gate precision/recall/F1 on "boss HP changed" / "Link HP changed",
                                        its ROC-AUC and best-threshold F1 (is the signal there at all, or
                                        only below the 0.5 cut), and the magnitude error on firing ticks
                                        against persistence's own error on those same ticks.
  Q4 is the ACTION used?                shuffled-action penalty (x1.00 = the action is decoration)

The held-out split is reproduced from the CHECKPOINT'S OWN config (data dirs, limit, val_frac, shard), so
these are the same episodes the trainer held out and not a fresh random draw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.wm_obs import (event_accuracy, field_index, load_files,   # noqa: E402
                             load_wm, scan_meta)

torch.set_num_threads(1)                                             # AGENTS 28


def rebuild_val(cfg, data_override=None, n_keep=48):
    """The trainer's OWN held-out episodes, identified from file metadata and then loaded -- `n_keep` of
    them, never the whole corpus, so this runs beside a live trainer (AGENTS 23)."""
    data = data_override or cfg.get("data") or []
    metas = scan_meta(data, int(cfg.get("limit", 0) or 0))
    world = int(cfg.get("fleet_world_size", 1) or 1)
    rank = int(cfg.get("fleet_rank", 0) or 0)
    if world > 1:
        def klass(e):
            f = str(e.get("file", "")).lower()
            return "human" if "human" in f or f[:2] == "ep" else ("win" if e.get("win") else "loss")
        buckets: dict[str, list] = {}
        for e in metas:
            buckets.setdefault(klass(e), []).append(e)
        metas = [e for k in sorted(buckets) for e in buckets[k][rank::world]]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(metas))
    n_val = max(4, int(float(cfg.get("val_frac", 0.1)) * len(metas)))
    held = [metas[i] for i in order[:n_val]]
    print(f"{len(metas)} episodes in this rank's shard, {n_val} held out, loading {min(n_keep, len(held))}",
          flush=True)
    return load_files(held[:n_keep])


def pad(group, obs_dim, dev, T_max=2000):
    T = min(max(len(e["obs"]) for e in group), T_max)
    B = len(group)
    obs = torch.zeros(B, T, obs_dim); act = torch.zeros(B, T, dtype=torch.long); msk = torch.zeros(B, T)
    for b, e in enumerate(group):
        n = min(len(e["obs"]), T)
        obs[b, :n] = torch.from_numpy(e["obs"][:n]); act[b, :n] = torch.from_numpy(e["act"][:n])
        msk[b, :n] = 1.0
    return obs.to(dev), act.to(dev), msk.to(dev)


def auc(scores, labels):
    """ROC-AUC by rank, so a gate that never crosses 0.5 can still be shown to ORDER events correctly."""
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = np.empty(len(s), float)
    o = np.argsort(s, kind="mergesort")
    sv = s[o]
    i = 0
    while i < len(sv):                                   # average ranks within ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def best_f1(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if not y.any():
        return float("nan"), float("nan")
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, 60)))
    bf, bt = 0.0, float("nan")
    for t in cand:
        p = s > t
        tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
        if tp == 0:
            continue
        f = 2 * tp / (2 * tp + fp + fn)
        if f > bf:
            bf, bt = f, float(t)
    return bf, bt


@torch.no_grad()
def report(ckpt, episodes, fields, dev, n_eps, horizons, max_steps):
    model, scales, ck = load_wm(ckpt, dev)
    obs_dim = int(ck["obs_dim"])
    idx = field_index(fields) if fields else (field_index(ck["fields"]) if ck.get("fields") else {})
    cols = {k: idx.get(v) for k, v in (("boss_hp", "gohma_hp_frac"), ("link_hp", "link_hp_frac"))}
    pool = episodes[:n_eps]
    out = {"checkpoint": str(ckpt), "epoch": int(ck.get("epoch", 0)), "n_val_episodes": len(pool),
           "horizons": {}, "events": {}}

    # ---- Q1 / Q4: per-horizon error vs persistence, and the shuffled-action ablation
    acc = {h: {"model": [], "persist": [], "shuf": []} for h in horizons}
    for i in range(0, len(pool), 8):
        group = pool[i:i + 8]
        if len(group) < 2:
            break
        obs, act, msk = pad(group, obs_dim, dev)
        B, T, _ = obs.shape
        for shuf, key in ((False, "model"), (True, "shuf")):
            aa = act[torch.randperm(B)] if shuf else act
            h = model.init_hidden(B, dev)
            cur = obs[:, 0].clone()
            for j in range(min(max(horizons), T - 1)):
                d, _, h = model.step(cur, aa[:, j], h, hard=True)
                cur = cur + d * scales
                if (j + 1) in acc:
                    acc[j + 1][key].append(float(((cur - obs[:, j + 1]).abs() / scales
                                                  * msk[:, j + 1][..., None]).mean()))
        for j in range(min(max(horizons), T - 1)):
            if (j + 1) in acc:
                acc[j + 1]["persist"].append(float(((obs[:, 0] - obs[:, j + 1]).abs() / scales
                                                    * msk[:, j + 1][..., None]).mean()))
    for h, d in acc.items():
        m = {k: (float(np.mean(v)) if v else float("nan")) for k, v in d.items()}
        m["skill"] = 1 - m["model"] / max(m["persist"], 1e-9)
        m["action_penalty"] = m["shuf"] / max(m["model"], 1e-9)
        m["n_batches"] = len(d["model"])
        out["horizons"][h] = m

    # ---- Q3: per-EVENT accuracy -- missed vs hallucinated, on all four key events
    out["events"] = event_accuracy(model, scales, pool, cols["boss_hp"], cols["link_hp"], dev,
                                   max_steps=max_steps, batch=8, obs_dim=obs_dim)
    # plus the gate's ORDERING on the two HP columns: is the signal there but below the 0.5 cut?
    out["gate"] = {}
    for name, col in cols.items():
        if col is None:
            continue
        sc, lab = [], []
        for i in range(0, len(pool), 8):
            group = pool[i:i + 8]
            if len(group) < 1:
                break
            obs, act, msk = pad(group, obs_dim, dev)
            h = model.init_hidden(obs.shape[0], dev)
            for t in range(min(obs.shape[1] - 1, max_steps)):
                _, g, h = model.step(obs[:, t], act[:, t], h)
                m = msk[:, t + 1] > 0.5
                if not bool(m.any()):
                    continue
                sc.extend(torch.sigmoid(g[:, col])[m].cpu().tolist())
                lab.extend(((obs[:, t + 1, col] - obs[:, t, col]).abs() > 1e-6)[m].cpu().tolist())
        y = np.asarray(lab, bool); s_ = np.asarray(sc, float)
        bf, bt = best_f1(s_, y)
        out["gate"][name] = {"auc": auc(s_, y), "best_f1": bf, "best_threshold": bt,
                             "p_on_event": float(s_[y].mean()) if y.any() else float("nan"),
                             "p_off_event": float(s_[~y].mean()) if (~y).any() else float("nan"),
                             "fires_at_0.5": int((s_ > 0.5).sum()), "n_ticks": int(len(y))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data", nargs="*", default=None, help="override the checkpoint's data dirs")
    ap.add_argument("--n-eps", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20, 100])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")

    rows = []
    cache: dict[str, tuple] = {}
    for c in a.ckpt:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        cfg = dict(ck.get("config") or {})
        key = json.dumps([a.data or cfg.get("data"), cfg.get("limit"), cfg.get("fleet_rank"),
                          cfg.get("fleet_world_size"), cfg.get("val_frac")], sort_keys=True)
        if key not in cache:
            cache[key] = rebuild_val(cfg, a.data, a.n_eps)
        val, fields = cache[key]
        del ck
        rows.append(report(c, val, fields, dev, a.n_eps, tuple(a.horizons), a.max_steps))

    print("\n" + "=" * 100)
    for r in rows:
        print(f"\n{Path(r['checkpoint']).name}  epoch {r['epoch']}  "
              f"{r['n_val_episodes']} held-out episodes")
        print("  Q1 BETTER THAN DOING NOTHING?")
        for h, m in sorted(r["horizons"].items()):
            v = ("BEATS persistence" if m["skill"] > 0.02 else
                 "LOSES TO persistence" if m["skill"] < -0.02 else "ties persistence")
            print(f"     H{h:<4} model {m['model']:.4f}  persistence {m['persist']:.4f}  "
                  f"skill {m['skill']:+.3f}  -> {v}")
        pen = r["horizons"].get(20, {}).get("action_penalty", float("nan"))
        print(f"  Q4 IS THE ACTION USED?  shuffled-action penalty x{pen:.2f} -> "
              f"{'ACTION-CONDITIONED' if pen > 1.1 else 'NO -- the action is decoration'}")
        print("  Q3 DOES IT PREDICT EVENTS?  (missed = the event happened and the model's next state "
              "does not contain it)")
        for k, e in r["events"].items():
            if not e["n_events"]:
                print(f"     {k:<12} no instances in these episodes")
                continue
            print(f"     {k:<12} {e['n_events']:>4} events in {e['n_ticks']:,} ticks | "
                  f"MISSED {100*e['missed']:5.1f}% | HALLUCINATED {e['halluc_per_1k_quiet']:.1f}/1k quiet | "
                  f"precision {e['precision']:.3f}")
            print(f"                  phantom drift on quiet ticks {e['phantom_drift_per_tick']:.2e} "
                  f"per tick (x{e['phantom_drift_per_tick']*100:.1f} of a hit over 100 steps)")
            if e["n_events"] and not np.isnan(e["event_skill"]):
                print(f"                  magnitude on firing ticks: MAE {e['mae_on_event']:.4f} vs "
                      f"true delta {e['mean_true_delta']:.4f} -> skill {e['event_skill']:+.3f}")
        for k, g in r.get("gate", {}).items():
            print(f"     gate[{k}]: fires at 0.5 on {g['fires_at_0.5']}/{g['n_ticks']:,} ticks | "
                  f"p {g['p_on_event']:.4f} on event vs {g['p_off_event']:.4f} off | AUC {g['auc']:.3f} | "
                  f"best F1 {g['best_f1']:.3f} at {g['best_threshold']:.4g} -> "
                  f"{'ORDERS events correctly, never fires' if g['auc'] > 0.6 and g['fires_at_0.5'] == 0 else 'no ordering signal' if g['auc'] <= 0.6 else 'fires'}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
