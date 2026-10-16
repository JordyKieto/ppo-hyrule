#!/usr/bin/env python3
"""Gohma world model over the POLICY'S OWN observation space, trained on FULL episodes.

Why this and not scripts/train_wm_tokens.py: that one consumes the structured transition rows written by
collect_transitions (link_dim 36 / actor_dim 14 / 63 actions, a Deku Baba nav grid), and the only Gohma data
that exists is 21,559 full episodes of the 1033-d policy observation. This model predicts the next OBS, which
is also exactly what a dream needs: the policy that plays in the dream is the same RecurrentEntityActorCritic
that plays the real game, so a Gohma policy can be initialised into it without conversion.

The regime is the concluded one in docs/WM_EXPERIMENT_REGIME.md plus the MOBA lessons in docs/WM_MOTION.md:

  * FULL EPISODES, not windows: truncated BPTT carries the recurrent state across the whole episode, and
    stage 2 rolls the model on its OWN output to the end of the episode.
  * DRIFT-CORRECTING TARGETS in rollout: the target at step j+1 is (true state at j+1) - (ROLLED state at j),
    never the true delta from the true state, or the model faithfully reproduces the right increment from the
    wrong place and perpetuates its own error.
  * STEP DECAY as a half-life, renormalised, so deep steps are not weighted to zero.
  * PER-FIELD SCALE = std of the NON-ZERO deltas (MOBA: "divided by a per-(class, field) scale"), so a field
    that moves rarely is not drowned by one that moves every tick.
  * EVENT-GATED LOSS on the sparse fields: the value loss for a field is averaged only over the ticks where
    it actually fires, and KEY-EVENT ticks (boss HP or Link HP changing) carry extra weight. Damage is what
    the reward pays for (AGENTS 34), so damage is what the WM is held to.
  * THE ACTION CARRIES ITS GEOMETRY: the 9 world directions enter as (sin, cos) beside the index embedding,
    because the model otherwise has to memorise 108 rows to learn that an action is a displacement.

THE CONTROLS, which decide whether any of it means anything (AGENTS 1):

  * PERSISTENCE: predict zero delta. The floor. A WM that does not beat it has learned nothing.
  * SHUFFLED ACTIONS: the same model, actions permuted within the batch. This MUST get worse. If it does
    not, the model is extrapolating momentum and the action is decoration -- the measurement that said
    "0% solved" for the token WM (docs/WM_MOTION.md).

    python3 -u scripts/train_wm_obs.py --data datasets/ppo/wm --epochs 30 --device cuda --out runs/wm_obs/wm.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# python puts the SCRIPT's directory on sys.path, not the repo root, so `python3 scripts/x.py` from the
# repo cannot import zelda_rl without this -- which is how lifting ObsWM into the package broke the run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.wm_obs import N_DIRS, ObsWM, field_index, load_episodes  # noqa: E402

def fit_scales(eps, obs_dim: int) -> np.ndarray:
    """Per-field std of the NON-ZERO deltas, the MOBA normalisation."""
    acc = np.zeros(obs_dim, np.float64)
    cnt = np.zeros(obs_dim, np.int64)
    for e in eps[: min(len(eps), 400)]:
        d = np.diff(e["obs"], axis=0)
        nz = np.abs(d) > 1e-8
        acc += (d * d * nz).sum(0)
        cnt += nz.sum(0)
    sc = np.sqrt(np.maximum(acc, 1e-12) / np.maximum(cnt, 1))
    sc[cnt == 0] = 1.0
    return np.maximum(sc, 1e-3).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16, help="episodes per batch")
    ap.add_argument("--chunk", type=int, default=64, help="truncated-BPTT window; state carries across them")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--gru", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--event-weight", type=float, default=5.0,
                    help="extra weight on ticks where boss HP or Link HP changes")
    ap.add_argument("--stage2-epoch", type=int, default=10,
                    help="epoch at which autoregressive (own-output) rollout training starts")
    # The reference (puffermoba-dyna/checkpoints/wm_multistep_k10/manifest.json) rolls k=10 with gamma 0.9.
    # 600 with a 200-epoch half-life is ~60x deeper than it ever rolled, and WM_EXPERIMENT_REGIME.md item 4
    # says not to extend the horizon mixture. Measured here: stage 2 at depth 600 drove H20 skill from
    # -0.05 to -1.28 with the trend at t=-2.43.
    ap.add_argument("--roll-half-life", type=float, default=10.0)
    ap.add_argument("--max-roll", type=int, default=10, help="deepest rollout in stage 2 (reference: 10)")
    ap.add_argument("--out", default="runs/wm_obs/wm.pt")
    ap.add_argument("--resume", action="store_true")
    # SCALE ACROSS GPU PODS, the way the reference does it (mobawm/fleet_link.py: "used by mobawm.wm.train
    # on each pod (rank) of a fleet experiment"). Episodes are sharded by rank so an epoch over the full
    # corpus takes 1/N the wall clock, and weights are averaged every --fleet-sync-every epochs. The 100
    # HUMAN WINS are kept on every rank: they are the only clean kills in the corpus and a WM that has not
    # seen one cannot dream one.
    ap.add_argument("--fleet-experiment", default=None)
    ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world-size", type=int, default=1)
    ap.add_argument("--fleet-aggregator", type=int, default=0)
    ap.add_argument("--fleet-launch-id", default="0")
    ap.add_argument("--fleet-sync-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"device {dev}", flush=True)

    eps, fields = load_episodes(a.data, a.limit)
    if len(eps) < 20:
        raise SystemExit(f"only {len(eps)} episodes -- refusing to train a WM on that")
    obs_dim = eps[0]["obs"].shape[1]
    n_actions = 108
    idx = field_index(fields) if fields else {}
    g_hp = idx.get("gohma_hp_frac"); l_hp = idx.get("link_hp_frac")
    print(f"obs {obs_dim}, actions {n_actions}, gohma_hp col {g_hp}, link_hp col {l_hp}", flush=True)

    if a.fleet_world_size > 1:
        n0 = len(eps)
        wins = [e for e in eps if e.get("win")]
        losses = [e for e in eps if not e.get("win")]
        mine = losses[a.fleet_rank::a.fleet_world_size]
        eps = wins + mine
        print(f"shard {a.fleet_rank}/{a.fleet_world_size}: every win ({len(wins)}) + "
              f"{len(mine)}/{len(losses)} losses -> {len(eps)} of {n0} episodes", flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    n_val = max(4, int(a.val_frac * len(eps)))
    val = [eps[i] for i in order[:n_val]]
    tr = [eps[i] for i in order[n_val:]]
    print(f"train {len(tr)} / val {len(val)} episodes", flush=True)

    scales = torch.from_numpy(fit_scales(tr, obs_dim)).to(dev)
    model = ObsWM(obs_dim, n_actions, a.hidden, a.gru).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    start = 1
    if a.resume and out.is_file():
        ck = torch.load(out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("epoch", 0)) + 1
        print(f"RESUMED at epoch {start}", flush=True)

    def batches(pool, bs):
        o = np.random.permutation(len(pool))
        for i in range(0, len(o) - bs + 1, bs):
            yield [pool[j] for j in o[i:i + bs]]

    def pad(group):
        T = min(max(len(e["obs"]) for e in group), 2000)
        B = len(group)
        obs = torch.zeros(B, T, obs_dim)
        act = torch.zeros(B, T, dtype=torch.long)
        msk = torch.zeros(B, T)
        for b, e in enumerate(group):
            n = min(len(e["obs"]), T)
            obs[b, :n] = torch.from_numpy(e["obs"][:n])
            act[b, :n] = torch.from_numpy(e["act"][:n])
            msk[b, :n] = 1.0
        return obs.to(dev), act.to(dev), msk.to(dev)

    def ev_weight(obs, msk):
        """Extra weight on KEY EVENT ticks: boss HP or Link HP changing (AGENTS 34)."""
        w = torch.ones_like(msk)
        if g_hp is not None and l_hp is not None:
            dg = (obs[:, 1:, g_hp] - obs[:, :-1, g_hp]).abs() > 1e-6
            dl = (obs[:, 1:, l_hp] - obs[:, :-1, l_hp]).abs() > 1e-6
            ev = (dg | dl).float()
            w[:, :-1] = 1.0 + (a.event_weight - 1.0) * ev
        return w * msk

    def teacher_forced(obs, act, msk, shuffle_actions=False):
        B, T, _ = obs.shape
        h = model.init_hidden(B, dev)
        aa = act[torch.randperm(B)] if shuffle_actions else act
        loss_num = loss_den = 0.0
        w = ev_weight(obs, msk)
        for t0 in range(0, T - 1, a.chunk):
            t1 = min(t0 + a.chunk, T - 1)
            h = h.detach()
            preds, tgts, ws, gates = [], [], [], []
            for t in range(t0, t1):
                d, g, h = model.step(obs[:, t], aa[:, t], h)
                preds.append(d); gates.append(g)
                tgts.append((obs[:, t + 1] - obs[:, t]) / scales)
                ws.append(w[:, t])
            P = torch.stack(preds, 1); Tg = torch.stack(tgts, 1); W = torch.stack(ws, 1)[..., None]
            G = torch.stack(gates, 1)
            # EVENT-GATED: a field's error counts on the ticks where it MOVES, plus a floor so a field
            # that never moves is still held to predicting no movement
            fired = (Tg.abs() > 1e-6).float()
            gate = 0.1 + 0.9 * fired
            # the GATE learns which fields change; the VALUE head only has to be right where they do
            bce = F.binary_cross_entropy_with_logits(G, fired, reduction="none")
            l = ((F.smooth_l1_loss(P, Tg, reduction="none") * gate + bce) * W).sum()
            den = (gate * W).sum().clamp(min=1.0)
            loss_num = loss_num + l; loss_den = loss_den + den
            if model.training:
                (l / den).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                opt.step(); opt.zero_grad(set_to_none=True)
        return float((loss_num / loss_den).detach() if isinstance(loss_num, torch.Tensor)
                     else loss_num / loss_den)

    def rollout_loss(obs, act, msk, depth):
        """Stage 2: the model's OWN output, drift-correcting targets, half-life step decay."""
        B, T, _ = obs.shape
        depth = min(depth, T - 1)
        h = model.init_hidden(B, dev)
        cur = obs[:, 0].clone()
        num = den = 0.0
        rep_n = rep_d = 0.0                       # reported loss, kept off the autograd graph
        w = ev_weight(obs, msk)
        for j in range(depth):
            d, g, h = model.step(cur, act[:, j], h)
            nxt = cur + d * scales
            tgt = (obs[:, j + 1] - cur) / scales                 # drift-correcting
            dec = 0.5 ** (j / a.roll_half_life)
            fired = (tgt.abs() > 1e-6).float()
            gate = 0.1 + 0.9 * fired
            ww = (w[:, j][..., None] * dec)
            bce = F.binary_cross_entropy_with_logits(g, fired, reduction="none")
            step_l = ((F.smooth_l1_loss(d, tgt, reduction="none") * gate + bce) * ww).sum()
            step_d = (gate * ww).sum()
            num = num + step_l
            den = den + step_d
            rep_n += float(step_l.detach()); rep_d += float(step_d.detach())
            cur = nxt.detach() if (j + 1) % a.chunk == 0 else nxt
            if (j + 1) % a.chunk == 0 and model.training:
                l = num / den.clamp(min=1.0)
                l.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                num = den = 0.0
                h = h.detach()
        if model.training and isinstance(den, torch.Tensor) and float(den.detach()) > 0:
            (num / den.clamp(min=1.0)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        return rep_n / max(rep_d, 1e-9)

    @torch.no_grad()
    def evaluate(pool, horizons=(1, 5, 20, 100)):
        """Per-horizon error against PERSISTENCE, plus the shuffled-action ablation (AGENTS 1)."""
        model.eval()
        res = {h: {"model": [], "persist": [], "shuf": []} for h in horizons}
        for group in batches(pool, min(8, len(pool))):
            obs, act, msk = pad(group)
            B, T, _ = obs.shape
            for shuf, key in ((False, "model"), (True, "shuf")):
                aa = act[torch.randperm(B)] if shuf else act
                h = model.init_hidden(B, dev)
                cur = obs[:, 0].clone()
                for j in range(min(max(horizons), T - 1)):
                    d, _, h = model.step(cur, aa[:, j], h, hard=True)
                    cur = cur + d * scales
                    if (j + 1) in res:
                        err = ((cur - obs[:, j + 1]).abs() / scales * msk[:, j + 1][..., None]).mean()
                        res[j + 1][key].append(float(err))
            for j in range(min(max(horizons), T - 1)):
                if (j + 1) in res:
                    err = ((obs[:, 0] - obs[:, j + 1]).abs() / scales * msk[:, j + 1][..., None]).mean()
                    res[j + 1]["persist"].append(float(err))
            break                                              # one batch is enough per epoch
        model.train()
        return {h: {k: float(np.mean(v)) if v else float("nan") for k, v in d.items()} for h, d in res.items()}

    link = None
    if a.fleet_experiment and a.fleet_world_size > 1:
        try:
            from zelda_rl.fleet_link import FleetLink
            link = FleetLink(a.fleet_experiment, a.fleet_rank, a.fleet_world_size, a.fleet_launch_id)
            print(f"fleet sync: {a.fleet_experiment} rank {a.fleet_rank}/{a.fleet_world_size}, "
                  f"averaging every {a.fleet_sync_every} epoch(s)", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"fleet sync unavailable ({str(e)[:70]}): training alone", flush=True)

    print(f"\n{'epoch':>5}{'stage':>7}{'train':>10}{'val':>10}   H1 model/persist/shuf   H20   H100", flush=True)
    for ep in range(start, a.epochs + 1):
        t0 = time.time()
        model.train()
        stage = 2 if ep >= a.stage2_epoch else 1
        tl, n = 0.0, 0
        for group in batches(tr, a.batch):
            obs, act, msk = pad(group)
            if stage == 1:
                tl += teacher_forced(obs, act, msk); n += 1
            else:
                # 60% one-step, the rest geometric in depth to the full episode (the concluded mixture)
                depth = (1 if np.random.rand() < 0.6
                         else int(min(a.max_roll, 2 ** np.random.uniform(1, np.log2(max(a.max_roll, 2))))))
                tl += (teacher_forced(obs, act, msk) if depth == 1
                       else rollout_loss(obs, act, msk, depth)); n += 1
        with torch.no_grad():
            model.eval()
            vl = np.mean([teacher_forced(*pad(g)) for g in batches(val, min(8, len(val)))])
            model.train()
        m = evaluate(val)
        if link is not None and ep % a.fleet_sync_every == 0:
            try:
                sync = link.average_weights(model, ep, aggregator=a.fleet_aggregator)
                if sync:
                    print(f"      fleet sync @epoch {ep}: {sync}", flush=True)
                _h20 = m[20]
                link.heartbeat({"epoch": ep,
                                "h20_skill": 1 - _h20["model"] / max(_h20["persist"], 1e-9),
                                "shuf_penalty": _h20["shuf"] / max(_h20["model"], 1e-9)})
            except Exception as e:                               # noqa: BLE001
                print(f"      fleet sync FAILED @epoch {ep}: {str(e)[:90]} -- training alone", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep,
                    "scales": scales.cpu().numpy(), "obs_dim": obs_dim, "n_actions": n_actions,
                    "hidden": a.hidden, "gru": a.gru, "fields": fields, "config": vars(a),
                    "metrics": m}, out)
        h1, h20, h100 = m[1], m[20], m[100]
        print(f"{ep:>5}{stage:>7}{tl / max(n,1):>10.4f}{vl:>10.4f}   "
              f"{h1['model']:.3f}/{h1['persist']:.3f}/{h1['shuf']:.3f}   "
              f"{h20['model']:.2f}/{h20['persist']:.2f}/{h20['shuf']:.2f}   "
              f"{h100['model']:.2f}/{h100['persist']:.2f}/{h100['shuf']:.2f}"
              f"   {time.time() - t0:.0f}s", flush=True)
        # THE VERDICT, every epoch, so a model that has learned nothing cannot hide behind a falling loss
        sk = 1 - h20["model"] / max(h20["persist"], 1e-9)
        ab = h20["shuf"] / max(h20["model"], 1e-9)
        print(f"      H20 skill vs persistence {sk:+.3f} | shuffled-action penalty x{ab:.2f} "
              f"({'ACTION-CONDITIONED' if ab > 1.1 else 'NOT action-conditioned -- the action is decoration'})",
              flush=True)
    print(f"\nsaved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
