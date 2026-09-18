"""Behaviour cloning on the Deku Baba structured-state transition dataset -> a frozen encoder E_BC(s) -> z.

Step one of the latent world model (paper §5-6): learn z = E_BC(s) from competent behaviour, freeze it, and
learn the dynamics in z later. This script only trains the BC model and writes the encoder checkpoint.

The demonstrators were recurrent (`RecurrentEntityActorCritic`) and saw episode-memory observation features
the 335-d structured state does not carry, so a_t = pi(s_t) is not exactly realisable. The GRU therefore sits
*after* the encoder, in the action head only: z_t = E(s_t), h_t = GRU(z_t, h_{t-1}), logits = W h_t. The
auxiliary heads stay on z_t. `--gru 0` gives the feedforward ablation.

Split is BY START SLOT (consecutive frames are near-duplicates, so a random split leaks); a few train-slot
episodes are held back as a dev set for checkpoint selection so the val slots stay untouched.

    .venv/bin/python scripts/train_bc.py --compare      # feedforward vs GRU x all/G110+/success-only
    .venv/bin/python scripts/train_bc.py --min-tier 110 --gru 128 --run-name bc-competent
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import BABA_ID, LINK_DIM, N_ROWS, RI, ROW_DIM, BCNet, Norm, featurise  # noqa: E402

DEFAULT_DATA = "datasets/transitions/deku-baba/20260918-134229"
# (tag suffix, episode filter) x (feedforward, GRU head)
TIERS = {"all": {}, "competent": {"min_tier": 110}, "success": {"success_only": True}}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default="checkpoints/bc")
    ap.add_argument("--run-name", default="bc")
    ap.add_argument("--val-slots", default="7,9")
    ap.add_argument("--min-tier", type=int, default=0, help="keep episodes with manifest global_update >= this")
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--compare", action="store_true", help="train every competence variant x {feedforward, GRU}")
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--row-hidden", type=int, default=96)
    ap.add_argument("--gru", type=int, default=128, help="GRUCell width in the action head; 0 = feedforward")
    ap.add_argument("--no-layernorm", dest="layernorm", action="store_false", default=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=48, help="BPTT sequence length (steps; 10 Hz)")
    ap.add_argument("--chunks-per-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--dev-episodes", type=int, default=20, help="train-slot episodes held back for selection")
    ap.add_argument("--w-reward", type=float, default=0.1)
    ap.add_argument("--w-terminal", type=float, default=0.1)
    ap.add_argument("--w-hp", type=float, default=0.2)
    ap.add_argument("--w-targets", type=float, default=0.2)
    ap.add_argument("--soft", type=float, default=1.0, help="weight on KL(pi_demo || pi_BC) soft targets vs the "
                    "sampled-action cross-entropy; needs the manifest's policy checkpoints (cached once)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]), help="root for checkpoint paths")
    ap.add_argument("--demo-cache", default=None, help="default <out-dir>/demo_targets.npz")
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--probe-lr", type=float, default=1e-2)
    ap.add_argument("--no-probe", dest="probe", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=4)
    return ap.parse_args()


# ---------------------------------------------------------------- data

def load(data_dir: Path):
    meta = json.loads((data_dir / "dataset.json").read_text())
    rows = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    S, NS, act, rew, term, ep = [], [], [], [], [], []
    for i, r in enumerate(rows):
        d = np.load(data_dir / r["file"])
        s, a = d["state"], d["action"]
        t = len(a)
        S.append(s[:t]); NS.append(s[1:t + 1]); act.append(a); rew.append(d["reward"])
        z = np.zeros(t, np.float32)
        if r["end"] != "timeout":
            z[-1] = 1.0                       # success / death; a timeout is truncation, not termination
        term.append(z)
        ep.append(np.full(t, i, np.int64))
        r["steps_actual"] = t
    lengths = np.array([r["steps_actual"] for r in rows])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    cat = np.concatenate
    D = {"state": cat(S), "next_state": cat(NS), "action": cat(act), "reward": cat(rew), "terminal": cat(term),
         "episode": cat(ep)}
    E = {"slot": np.array([r["slot"] for r in rows]), "tier": np.array([r["global_update"] for r in rows]),
         "success": np.array([int(r["success"]) for r in rows]), "offset": offsets, "length": lengths}
    return meta, rows, D, E


def derived(state: np.ndarray):
    """Labels read off the raw state: Link HP, live target count, nearest-target distance, its behaviour state."""
    n = len(state)
    rows = state[:, LINK_DIM:].reshape(n, N_ROWS, ROW_DIM)
    live = (rows[:, :, RI["present"]] > 0.5) & (rows[:, :, RI["id"]] == BABA_ID) & (rows[:, :, RI["hp"]] > 0)
    rpos = rows[:, :, [RI["pos_x"], RI["pos_y"], RI["pos_z"]]]
    dist = np.linalg.norm(rpos - state[:, None, [5, 6, 7]], axis=2)
    dist = np.where(live, dist, np.inf)
    j = dist.argmin(1)
    any_live = live.any(1)
    near = np.where(any_live, dist[np.arange(n), j], 0.0).astype(np.float32)
    sidx = np.clip(rows[np.arange(n), j, RI["state_idx"]].astype(np.int64), 0, 9)
    return {"hp": state[:, 2].astype(np.float32), "targets": live.sum(1).astype(np.float32),
            "near_dist": near, "near_state": np.where(any_live, sidx, 9), "has_target": any_live}


def pad_index(eps, E, chunk=None, rng=None):
    """Index matrix (B, L) into the flat arrays plus a validity mask. Whole episodes if chunk is None,
    otherwise contiguous length-`chunk` windows tiled over each episode from a random offset."""
    if chunk is None:
        L = int(E["length"][eps].max())
        idx = E["offset"][eps][:, None] + np.arange(L)[None, :]
        valid = (np.arange(L)[None, :] < E["length"][eps][:, None])
        return np.minimum(idx, (E["offset"][eps] + E["length"][eps] - 1)[:, None]), valid.astype(np.float32)
    starts, lens = [], []
    for e in eps:
        n, o = int(E["length"][e]), int(E["offset"][e])
        s0 = int(rng.integers(0, min(chunk, n))) if rng is not None else 0
        if s0:
            starts.append(o); lens.append(min(s0, n))
        for s in range(s0, n, chunk):
            starts.append(o + s); lens.append(min(chunk, n - s))
    starts, lens = np.array(starts), np.array(lens)
    idx = starts[:, None] + np.arange(chunk)[None, :]
    valid = (np.arange(chunk)[None, :] < lens[:, None])
    return np.minimum(idx, (starts + lens - 1)[:, None]), valid.astype(np.float32)


def demo_targets(a, data_dir, manifest, E, meta, path: Path):
    """pi_demo(.|s) for every frame, by replaying each manifest checkpoint over its own episodes' stored `obs`.

    The demonstrators are recurrent, so the hidden state is rolled from the episode start; obs alone then
    reproduces the exact distribution the stored action was sampled from. This buys three things: soft targets
    (far better posed than a single sample), KL(pi_demo || pi_BC), and the Bayes-optimal top-1 / CE ceiling.
    """
    if path.exists():
        d = np.load(path)
        if len(d["logp"]) == int(E["length"].sum()):
            print(f"demo targets: cached {path}", flush=True)
            return d["logp"]
    from zelda_rl.ppo import build_model
    out = np.zeros((int(E["length"].sum()), meta["n_actions"]), np.float16)
    t0 = time.time()
    for ck in sorted({r["checkpoint"] for r in manifest}):
        c = torch.load(Path(a.repo) / ck, map_location="cpu", weights_only=False)
        m = build_model(c["config"]); m.load_state_dict(c["model"]); m.eval()
        eps = np.array([i for i, r in enumerate(manifest) if r["checkpoint"] == ck])
        for s in range(0, len(eps), 8):
            sel = eps[s:s + 8]
            L = int(E["length"][sel].max())
            ob = np.zeros((len(sel), L, meta["obs_dim"]), np.float32)
            for j, e in enumerate(sel):
                o = np.load(data_dir / manifest[e]["file"])["obs"]
                ob[j, :len(o)] = o
            ob = torch.from_numpy(ob)
            h = m.init_hidden(len(sel))
            with torch.no_grad():
                for t in range(L):
                    logits, _, h = m.step(ob[:, t], h, torch.full((len(sel),), 1.0 if t == 0 else 0.0))
                    lp = torch.log_softmax(logits, -1).numpy().astype(np.float16)
                    for j, e in enumerate(sel):
                        if t < E["length"][e]:
                            out[E["offset"][e] + t] = lp[j]
        print(f"demo targets: {ck} ({len(eps)} eps, {time.time() - t0:.0f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logp=out)
    return out


# ---------------------------------------------------------------- forward passes

def roll(net, T, idx, valid, grad=False):
    """Encode every position (batched, no recurrence), then roll the GRU head along the time axis."""
    B, L = idx.shape
    flat = torch.from_numpy(idx.reshape(-1))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        z = net.encode(T["link"][flat], T["rows"][flat], T["ids"][flat], T["mask"][flat]).reshape(B, L, -1)
        if net.gru is None:
            logits = net.action(z)
        else:
            h = net.init_hidden(B)
            outs = []
            for k in range(L):
                lg, h = net.step(z[:, k], h, None)
                outs.append(lg)
            logits = torch.stack(outs, 1)
        return z, logits, net.heads(z.reshape(B * L, -1))


@torch.no_grad()
def predict(net, T, eps, E, bs=24):
    """Whole-episode rollout (hidden state carried from the episode start). -> flat idx + per-frame outputs."""
    net.eval()
    idx_out, z_out, lg_out, aux_out = [], [], [], {k: [] for k in BCNet.AUX}
    for s in range(0, len(eps), bs):
        sel = eps[s:s + bs]
        idx, valid = pad_index(sel, E)
        z, logits, aux = roll(net, T, idx, valid)
        m = valid.reshape(-1) > 0
        idx_out.append(idx.reshape(-1)[m])
        z_out.append(z.reshape(-1, z.shape[-1])[m])
        lg_out.append(logits.reshape(-1, logits.shape[-1])[m])
        for k, v in aux.items():
            aux_out[k].append(v[m])
    return (np.concatenate(idx_out), torch.cat(z_out), torch.cat(lg_out),
            {k: torch.cat(v) for k, v in aux_out.items()})


def metrics(pred, lab, rmu, rsd, sub=None):
    """Pure aggregation over a frame subset of a prediction, so breakdowns need no extra forward pass."""
    idx, _z, logits, aux = pred
    if sub is not None:
        idx, logits, aux = idx[sub], logits[sub], {k: v[sub] for k, v in aux.items()}
    n = len(idx)
    if n == 0:
        return {}
    y = torch.from_numpy(lab["action"][idx])
    k3 = logits.topk(3, -1).indices
    g = logits.reshape(n, 9, 7)                      # 9 world-frame stick directions x 7 button combos
    rew = torch.from_numpy(((lab["reward"][idx] - rmu) / rsd).astype(np.float32))
    hp = torch.from_numpy(lab["next_hp"][idx] / 48.0)
    tg = torch.from_numpy(lab["next_targets"][idx] / 3.0)
    t = lab["terminal"][idx].astype(np.int64)
    p = (aux["terminal"].numpy() > 0).astype(np.int64)
    tp, fp, fn = int(((p == 1) & (t == 1)).sum()), int(((p == 1) & (t == 0)).sum()), int(((p == 0) & (t == 1)).sum())
    extra = {}
    if lab.get("demo_logp") is not None:
        dlp = torch.from_numpy(lab["demo_logp"][idx].astype(np.float32))
        dp = dlp.exp()
        blp = torch.log_softmax(logits, -1)
        extra = {"kl": float((dp * (dlp - blp)).sum(-1).mean()), "demo_ce": float(-(dp * dlp).sum(-1).mean()),
                 "demo_top1": float((dlp.argmax(-1) == y).float().mean()),
                 "mode_agree": float((blp.argmax(-1) == dlp.argmax(-1)).float().mean())}
    return {"n": n, "top1": float((k3[:, 0] == y).float().mean()), "top3": float((k3 == y[:, None]).any(1).float().mean()),
            **extra,
            "ce": float(F.cross_entropy(logits, y)), "dir_top1": float((g.logsumexp(2).argmax(1) == y // 7).float().mean()),
            "btn_top1": float((g.logsumexp(1).argmax(1) == y % 7).float().mean()),
            "reward_mae": float((aux["reward"] * rsd + rmu - torch.from_numpy(lab["reward"][idx])).abs().mean()),
            "hp_mae": float((aux["hp"] * 48.0 - torch.from_numpy(lab["next_hp"][idx])).abs().mean()),
            "targets_mae": float((aux["targets"] * 3.0 - torch.from_numpy(lab["next_targets"][idx])).abs().mean()),
            "l_reward": float(F.smooth_l1_loss(aux["reward"], rew)), "l_hp": float(F.mse_loss(aux["hp"], hp)),
            "l_targets": float(F.mse_loss(aux["targets"], tg)), "term_pos": tp + fn,
            "term_prec": tp / max(tp + fp, 1), "term_recall": tp / max(tp + fn, 1)}


# ---------------------------------------------------------------- training

def run_variant(a, Fa, lab, E, split, meta, tag, gru, min_tier=0, success_only=False, out_dir=None):
    # the dev set gets the same competence filter: selecting against behaviour the variant deliberately
    # excluded picks the untrained epoch (measured: G110+ collapsed to the majority class without this).
    keep = lambda eps: eps[(E["tier"][eps] >= min_tier) & ((E["success"][eps] == 1) | (not success_only))]
    tr_eps, dev_eps = keep(split["train"]), keep(split["dev"])
    if len(dev_eps) == 0:
        dev_eps = split["dev"]
    tr_frames = np.concatenate([E["offset"][e] + np.arange(E["length"][e]) for e in tr_eps])
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    norm = Norm.fit({k: v[tr_frames] for k, v in Fa.items()})
    T = {k: torch.from_numpy(v) for k, v in norm.apply(Fa).items()}
    rmu, rsd = float(lab["reward"][tr_frames].mean()), float(lab["reward"][tr_frames].std() + 1e-6)
    A = torch.from_numpy(lab["action"])
    Y = {"reward": torch.from_numpy(((lab["reward"] - rmu) / rsd).astype(np.float32)),
         "terminal": torch.from_numpy(lab["terminal"]),
         "hp": torch.from_numpy(lab["next_hp"] / 48.0), "targets": torch.from_numpy(lab["next_targets"] / 3.0)}
    pos = max(float(lab["terminal"][tr_frames].sum()), 1.0)
    pw = torch.tensor((len(tr_frames) - pos) / pos, dtype=torch.float32)
    DLP = torch.from_numpy(lab["demo_logp"].astype(np.float32)) if a.soft > 0 else None
    w = {"reward": a.w_reward, "terminal": a.w_terminal, "hp": a.w_hp, "targets": a.w_targets}

    net = BCNet(link_dim=Fa["link"].shape[1], row_dim=Fa["rows"].shape[2], n_actions=meta["n_actions"],
                latent=a.latent, hidden=a.hidden, row_hidden=a.row_hidden, layernorm=a.layernorm,
                dropout=a.dropout, gru=gru)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    nch = len(pad_index(tr_eps, E, a.chunk)[0])
    steps = a.epochs * max(nch // a.chunks_per_batch, 1) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps + 1, pct_start=0.2)

    t0 = time.time()
    best = (float("inf"), -1, None)
    for ep in range(a.epochs):
        net.train()
        idx, valid = pad_index(tr_eps, E, a.chunk, rng)
        order = rng.permutation(len(idx))
        tot = nb = 0.0
        for s in range(0, len(order), a.chunks_per_batch):
            b = order[s:s + a.chunks_per_batch]
            bi, bv = idx[b], torch.from_numpy(valid[b])
            _, logits, aux = roll(net, T, bi, bv, grad=True)
            fi = torch.from_numpy(bi.reshape(-1))
            v = bv.reshape(-1)
            nv = v.sum().clamp(min=1.0)
            wl = lambda x: (x * v).sum() / nv
            lg = logits.reshape(-1, logits.shape[-1])
            loss = (1.0 - a.soft) * wl(F.cross_entropy(lg, A[fi], reduction="none",
                                                       label_smoothing=a.label_smoothing))
            if a.soft > 0:  # KL(pi_demo || pi_BC): match the distribution, not the one sampled index
                dlp = DLP[fi]
                loss = loss + a.soft * wl((dlp.exp() * (dlp - torch.log_softmax(lg, -1))).sum(-1))
            loss = loss + w["reward"] * wl(F.smooth_l1_loss(aux["reward"], Y["reward"][fi], reduction="none"))
            loss = loss + w["terminal"] * wl(F.binary_cross_entropy_with_logits(
                aux["terminal"], Y["terminal"][fi], pos_weight=pw, reduction="none"))
            loss = loss + w["hp"] * wl((aux["hp"] - Y["hp"][fi]) ** 2)
            loss = loss + w["targets"] * wl((aux["targets"] - Y["targets"][fi]) ** 2)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps:
                sched.step()
            tot += loss.item(); nb += 1
        pv = predict(net, T, split["val"], E)
        pd_ = predict(net, T, dev_eps, E)
        d, m = metrics(pd_, lab, rmu, rsd), metrics(pv, lab, rmu, rsd)
        # select on the full dev objective: the encoder is the artefact, and the aux heads are what keep
        # reward / HP / target count readable from z.
        obj = (d["kl"] if a.soft > 0 else d["ce"]) + w["reward"] * d["l_reward"] + w["hp"] * d["l_hp"] \
            + w["targets"] * d["l_targets"]
        if obj < best[0]:
            best = (obj, ep, {k: v.detach().clone() for k, v in net.state_dict().items()})
        print(f"[{tag}] ep {ep + 1}/{a.epochs} train {tot / nb:.4f} dev_obj {obj:.4f} dev_ce {d['ce']:.4f} "
              f"dev_top1 {d['top1']:.4f} | val top1 {m['top1']:.4f} top3 {m['top3']:.4f} dir {m['dir_top1']:.4f} "
              f"hp_mae {m['hp_mae']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best[2])
    print(f"[{tag}] selected epoch {best[1] + 1} (dev objective {best[0]:.4f})", flush=True)

    pv = predict(net, T, split["val"], E)
    out = {"tag": tag, "gru": gru, "train_episodes": len(tr_eps), "train_n": len(tr_frames),
           "seconds": time.time() - t0, "best_epoch": best[1] + 1, "dev_objective": best[0]}
    out.update(metrics(pv, lab, rmu, rsd))
    ts = tr_eps[np.random.default_rng(0).permutation(len(tr_eps))[:24]]     # train-split fit, for the CE gap
    out["train_metrics"] = metrics(predict(net, T, ts, E), lab, rmu, rsd)
    slot, tier = E["slot"][lab["episode"][pv[0]]], E["tier"][lab["episode"][pv[0]]]
    out["per_slot"] = {int(s): metrics(pv, lab, rmu, rsd, sub=slot == s) for s in sorted(set(slot.tolist()))}
    out["per_tier"] = {int(t): metrics(pv, lab, rmu, rsd, sub=tier == t) for t in sorted(set(tier.tolist()))}
    if out_dir is not None:
        ck = {"model": net.state_dict(), "norm": norm.state_dict(),
              "config": vars(a) | {"tag": tag, "gru": gru, "min_tier": min_tier, "success_only": success_only},
              "latent": a.latent, "link_dim": int(Fa["link"].shape[1]), "row_dim": int(Fa["rows"].shape[2]),
              "n_actions": meta["n_actions"], "reward_norm": [rmu, rsd],
              "split": {"val_slots": sorted(set(E["slot"][split["val"]].tolist())),
                        "train_slots": sorted(set(E["slot"][split["train"]].tolist())),
                        "dev_episodes": [int(x) for x in dev_eps],
                        "train_episodes": [int(x) for x in tr_eps]},
              "dataset": str(a.data), "metrics": out}
        p = Path(out_dir) / f"{tag}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, p)
        out["checkpoint"] = str(p)
    return net, T, out


# ---------------------------------------------------------------- probes

def linear_probe(Xtr, ytr, Xva, yva, classes=None, epochs=40, bs=1024, lr=1e-2, seed=0):
    """Identical optimiser, schedule and budget for every feature set; inputs must already be standardised."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lin = nn.Linear(Xtr.shape[1], classes or 1)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, pct_start=0.15,
                                                total_steps=epochs * (len(Xtr) // bs + 1) + 1)
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), bs):
            i = torch.from_numpy(perm[s:s + bs])
            p = lin(Xtr[i])
            loss = F.cross_entropy(p, ytr[i]) if classes else F.mse_loss(p.squeeze(-1), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        q = lin(Xtr)
        fit = (float((q.argmax(-1) == ytr).float().mean()) if classes
               else float(1 - ((q.squeeze(-1) - ytr) ** 2).mean() / ytr.var()))
        p = lin(Xva)
        if classes:
            return {"val": float((p.argmax(-1) == yva).float().mean()), "train": fit}
        p = p.squeeze(-1)
        return {"val": float(1 - ((p - yva) ** 2).mean() / yva.var()), "train": fit,
                "mae": float((p - yva).abs().mean())}


def main():
    a = parse_args()
    torch.set_num_threads(a.torch_threads)
    meta, manifest, D, E = load(Path(a.data))
    print(f"loaded {len(manifest)} episodes, {len(D['action'])} transitions", flush=True)

    Fa = featurise(D["state"])
    dv, dvn = derived(D["state"]), derived(D["next_state"])
    lab = {"action": D["action"], "reward": D["reward"], "terminal": D["terminal"], "episode": D["episode"],
           "next_hp": dvn["hp"], "next_targets": dvn["targets"], "near_state": dv["near_state"],
           "near_dist": dv["near_dist"], "demo_logp": None}
    if a.soft > 0:
        cache = Path(a.demo_cache or Path(a.out_dir) / "demo_targets.npz")
        lab["demo_logp"] = demo_targets(a, Path(a.data), manifest, E, meta, cache)

    val_slots = [int(s) for s in a.val_slots.split(",") if s]
    all_eps = np.arange(len(manifest))
    val_eps = all_eps[np.isin(E["slot"], val_slots)]
    pool = all_eps[~np.isin(E["slot"], val_slots)]
    perm = np.random.default_rng(a.seed).permutation(pool)
    dev_eps, tr_eps = np.sort(perm[:a.dev_episodes]), np.sort(perm[a.dev_episodes:])
    split = {"train": tr_eps, "dev": dev_eps, "val": val_eps}
    fr = lambda eps: int(E["length"][eps].sum())
    print(f"split: train slots {sorted(set(E['slot'][tr_eps].tolist()))} {len(tr_eps)} eps / {fr(tr_eps)} frames | "
          f"dev {len(dev_eps)} eps / {fr(dev_eps)} frames | val slots {val_slots} {len(val_eps)} eps / "
          f"{fr(val_eps)} frames", flush=True)

    tr_frames = np.concatenate([E["offset"][e] + np.arange(E["length"][e]) for e in tr_eps])
    va_frames = np.concatenate([E["offset"][e] + np.arange(E["length"][e]) for e in val_eps])
    st = D["state"][tr_frames]
    sd = st.std(0)
    print(f"raw state: {int((sd < 1e-9).sum())}/{st.shape[1]} dims constant on the train split", flush=True)

    ya, yv = D["action"][tr_frames], D["action"][va_frames]
    maj = Counter(ya.tolist()).most_common(1)[0]
    p = np.bincount(ya, minlength=meta["n_actions"]).astype(np.float64); p /= p.sum()
    base = {"top1": float((yv == maj[0]).mean()),
            "top3": float(np.isin(yv, [k for k, _ in Counter(ya.tolist()).most_common(3)]).mean()),
            "dir_top1": float((yv // 7 == Counter((ya // 7).tolist()).most_common(1)[0][0]).mean()),
            "btn_top1": float((yv % 7 == Counter((ya % 7).tolist()).most_common(1)[0][0]).mean()),
            "ce": float(-np.log(np.maximum(p[yv], 1e-12)).mean())}
    print(f"majority baseline: action {maj[0]} ({meta['action_names'][maj[0]]}) train freq "
          f"{maj[1] / len(ya):.4f} -> val " + " ".join(f"{k} {v:.4f}" for k, v in base.items()), flush=True)
    ceil = {}
    if lab["demo_logp"] is not None:
        dlp = lab["demo_logp"][va_frames].astype(np.float32)
        dp = np.exp(dlp)
        ceil = {"top1": float((dlp.argmax(1) == yv).mean()), "ce": float(-(dp * dlp).sum(1).mean()),
                "top3": float((np.argsort(-dlp, 1)[:, :3] == yv[:, None]).any(1).mean()),
                "dir_top1": float((dp.reshape(-1, 9, 7).sum(2).argmax(1) == yv // 7).mean()),
                "btn_top1": float((dp.reshape(-1, 9, 7).sum(1).argmax(1) == yv % 7).mean()), "kl": 0.0}
        print(f"demonstrator ceiling on val slots: top1 {ceil['top1']:.4f} top3 {ceil['top3']:.4f} "
              f"dir {ceil['dir_top1']:.4f} btn {ceil['btn_top1']:.4f} entropy/ce {ceil['ce']:.4f} "
              f"(effective {np.exp(ceil['ce']):.1f} of 63 actions)", flush=True)

    if a.compare:
        variants = {f"{a.run_name}-{t}-{'gru' if g else 'ff'}": dict(kw, gru=g)
                    for t, kw in TIERS.items() for g in (0, a.gru)}
    else:
        variants = {a.run_name: {"gru": a.gru, "min_tier": a.min_tier, "success_only": a.success_only}}
    probe_sub = tr_frames[np.random.default_rng(0).permutation(len(tr_frames))[:40000]]
    results, zs = {}, {}
    for tag, kw in variants.items():
        net, T, out = run_variant(a, Fa, lab, E, split, meta, tag, out_dir=a.out_dir, **kw)
        results[tag] = out
        with torch.no_grad():                       # keep only the frozen latents, not the per-variant tensors
            net.eval()
            enc = lambda i: torch.cat([net.encode(T["link"][j], T["rows"][j], T["ids"][j], T["mask"][j])
                                       for j in (torch.from_numpy(i[s:s + 8192]) for s in range(0, len(i), 8192))])
            zs[tag] = (enc(probe_sub), enc(va_frames))
        del net, T
        print("  per slot:", {s: round(m["top1"], 4) for s, m in out["per_slot"].items()},
              "\n  per tier:", {t: round(m["top1"], 4) for t, m in out["per_tier"].items()}, flush=True)

    print(f"\n=== action prediction, held-out slots {val_slots} ({len(va_frames)} frames) ===")
    cols = ("top1", "top3", "dir_top1", "btn_top1", "ce", "kl", "reward_mae", "hp_mae", "targets_mae",
            "term_prec", "term_recall")
    nanf = float("nan")
    row = lambda d: "".join(f"{d.get(c, nanf):>12.4f}" for c in cols)
    print(f"{'variant':<26}{'eps':>5}{'frames':>8}" + "".join(f"{c:>12}" for c in cols))
    print(f"{'majority baseline':<26}{'-':>5}{'-':>8}" + row(base))
    if ceil:
        print(f"{'demonstrator (ceiling)':<26}{'-':>5}{'-':>8}" + row(ceil))
    for tag, o in results.items():
        print(f"{tag:<26}{o['train_episodes']:>5}{o['train_n']:>8}" + row(o))

    print("\n=== overfit check: train split vs held-out slots ===")
    k = "kl" if ceil else "ce"
    print(f"{'variant':<26}{'train top1':>12}{'val top1':>12}{'train ' + k:>12}{'val ' + k:>12}{'gap':>8}")
    for tag, o in results.items():
        t = o["train_metrics"]
        print(f"{tag:<26}{t['top1']:>12.4f}{o['top1']:>12.4f}{t[k]:>12.4f}{o[k]:>12.4f}{o[k] - t[k]:>8.3f}")

    print("\n=== per held-out slot / per competence tier (top1) ===")
    for tag, o in results.items():
        print(f"{tag:<26} slots " + " ".join(f"s{s}={m['top1']:.4f}" for s, m in o["per_slot"].items())
              + "  tiers " + " ".join(f"G{t}={m['top1']:.4f}" for t, m in o["per_tier"].items()))

    if a.probe:
        print("\n=== linear probes: frozen features -> held-out slots ===")
        sub = probe_sub
        ns_tr, ns_va = torch.from_numpy(lab["near_state"][sub]), torch.from_numpy(lab["near_state"][va_frames])
        mu, sdev = lab["near_dist"][sub].mean(), lab["near_dist"][sub].std() + 1e-6
        nd_tr = torch.from_numpy((lab["near_dist"][sub] - mu) / sdev)
        nd_va = torch.from_numpy((lab["near_dist"][va_frames] - mu) / sdev)
        m0, s0 = st.mean(0), np.where(sd < 1e-9, 1.0, sd)
        raw = torch.from_numpy(((D["state"] - m0) / s0).astype(np.float32))
        # every feature set is standardised with train-split mean/std and gets the same probe budget
        nrm = Norm.fit({k: v[tr_frames] for k, v in Fa.items()}).apply(Fa)
        feat = np.concatenate([nrm["link"], nrm["rows"].reshape(len(raw), -1)], 1)
        rows = [("raw state, standardised", raw[sub], raw[va_frames]),
                ("model input features", torch.from_numpy(feat[sub]), torch.from_numpy(feat[va_frames]))]
        rows += [(f"z:{tag}", ztr, zva) for tag, (ztr, zva) in zs.items()]
        mc = Counter(lab["near_state"][sub].tolist()).most_common(1)[0][0]
        print(f"{'features':<34}{'dim':>6}{'state fit':>10}{'state val':>10}{'dist fit':>10}{'dist val':>10}")
        print(f"{'majority / predict-the-mean':<34}{'-':>6}"
              f"{float((lab['near_state'][sub] == mc).mean()):>10.4f}"
              f"{float((lab['near_state'][va_frames] == mc).mean()):>10.4f}{0.0:>10.4f}"
              f"{float(1 - ((nd_va - nd_tr.mean()) ** 2).mean() / nd_va.var()):>10.4f}")
        for name, Xtr, Xva in rows:
            c = linear_probe(Xtr, ns_tr, Xva, ns_va, classes=10, epochs=a.probe_epochs, lr=a.probe_lr)
            r = linear_probe(Xtr, nd_tr, Xva, nd_va, epochs=a.probe_epochs, lr=a.probe_lr)
            print(f"{name:<34}{Xtr.shape[1]:>6}{c['train']:>10.4f}{c['val']:>10.4f}"
                  f"{r['train']:>10.4f}{r['val']:>10.4f}", flush=True)
        print(f"near_dist val std {lab['near_dist'][va_frames].std():.1f} game units; "
              f"'fit' columns are the same metric on the 40k train frames the probe was fitted on")
        mv = Counter(lab["near_state"][va_frames].tolist()).most_common(1)[0]
        print(f"oracle val-majority class for near_state: {mv[1] / len(va_frames):.4f} (class {mv[0]}); the "
              f"train-majority class scores only {float((lab['near_state'][va_frames] == mc).mean()):.4f} on val "
              f"-- the two splits have different modal behaviour states")


if __name__ == "__main__":
    main()
