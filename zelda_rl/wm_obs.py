"""The Gohma observation-space world model, shared by its trainer and the dream.

Defined here rather than in the training script so `scripts/train_dream_obs_ppo.py` reconstructs the SAME
architecture the checkpoint was saved from -- a dream that silently builds a different model would load the
state dict and roll garbage.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


N_DIRS = 9                      # 8 compass directions + neutral; the action index is dir * n_btn + btn


def load_episodes(dirs, limit=0, min_steps=120):
    eps, bad = [], 0
    # RECURSIVE: the collectors write <out>/episodes/*.npz and the Dyna loop passes the parent of its
    # per-iteration shards, so a one-level glob found 0 episodes and the loop died at its WM stage with
    # "refusing to train a WM on that" while 60 freshly collected episodes sat two directories down.
    files = [f for d in dirs for f in sorted(Path(d).rglob("*.npz"))]
    if limit:
        files = files[:limit]
    fields = None
    import time as _t
    _t0 = _t.time()
    for i, f in enumerate(files, 1):
        if i % 500 == 0:
            el = _t.time() - _t0
            print(f"  loading {i}/{len(files)}  {i/max(el,1e-9):.0f} files/s  "
                  f"eta {(len(files)-i)/max(i/max(el,1e-9),1e-9)/60:.1f} min", flush=True)
        try:
            z = np.load(f, allow_pickle=False)
            if "obs" not in z.files or "action" not in z.files:
                continue
            obs = z["obs"].astype(np.float32)
            act = z["action"].astype(np.int64)
            if len(obs) < min_steps:
                continue
            if fields is None and "obs_fields" in z.files:
                fields = json.loads(str(z["obs_fields"]))
            eps.append({"obs": obs, "act": act, "file": f.name,
                        "win": bool(z["boss_defeated"]) if "boss_defeated" in z.files else False})
        except Exception:                                    # noqa: BLE001
            bad += 1
    print(f"loaded {len(eps)} episodes ({bad} unreadable), "
          f"{sum(len(e['obs']) for e in eps):,} steps", flush=True)
    return eps, fields


def scan_meta(dirs, limit=0, min_steps=120):
    """File metadata ONLY -- no `obs` array is decompressed.

    `load_episodes` has to read every file to decide which episodes it keeps, which costs ~25 min and tens
    of GB on the full corpus. A held-out split only needs the ACCEPT/REJECT decision and the order, both of
    which are decidable from the npz directory plus the `obs` npy header, so an eval can rebuild the
    trainer's exact split alongside a live trainer without a second copy of the corpus in RAM (AGENTS 23).
    Returns the accepted files in the same order `load_episodes` would.
    """
    files = [f for d in dirs for f in sorted(Path(d).rglob("*.npz"))]   # see load_episodes
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        try:
            with np.load(f, allow_pickle=False) as z:
                if "obs" not in z.files or "action" not in z.files:
                    continue
                with z.zip.open("obs.npy") as fh:
                    ver = np.lib.format.read_magic(fh)
                    if ver == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(fh)
                    elif ver == (2, 0):
                        shape, _, _ = np.lib.format.read_array_header_2_0(fh)
                    else:
                        continue
                if int(shape[0]) < min_steps:
                    continue
                win = bool(z["boss_defeated"]) if "boss_defeated" in z.files else False
            out.append({"path": f, "file": f.name, "win": win, "steps": int(shape[0])})
        except Exception:                                    # noqa: BLE001
            continue
    return out


def load_files(metas):
    """Load just the episodes named by `scan_meta` rows, in their given order."""
    eps, fields = [], None
    for m in metas:
        z = np.load(m["path"], allow_pickle=False)
        if fields is None and "obs_fields" in z.files:
            fields = json.loads(str(z["obs_fields"]))
        eps.append({"obs": z["obs"].astype(np.float32), "act": z["action"].astype(np.int64),
                    "file": m["file"], "win": m["win"]})
    return eps, fields


def field_index(fields) -> dict:
    """name -> column. Accepts BOTH layout shapes in this repo and must, or a lookup silently returns None.

    The episode npz written by `train_combat.py` stores `{"global": [(name, width), ...], ...}`; a policy
    checkpoint's `config["obs_fields"]` stores the FLAT list `[(name, width), ...]` of the whole 1033-d
    vector. The global block comes first in both, so the offsets of the fields that matter agree -- but a
    `fields["global"]` lookup raises on the flat form, and a loop over the dict yields its keys, so a
    careless reader gets a wrong index rather than an error.
    """
    rows = fields["global"] if isinstance(fields, dict) else fields
    idx, off = {}, 0
    for name, w in rows:
        idx[name] = off
        off += int(w)
    return idx


def layout_width(fields) -> int:
    """Total width the layout describes -- compare it to obs_dim before trusting any column (AGENTS 3)."""
    rows = fields["global"] if isinstance(fields, dict) else fields
    return sum(int(w) for _, w in rows)


def fields_from_checkpoint(path) -> list:
    """`config["obs_fields"]` out of a policy checkpoint, for corpora whose npz do not carry the layout."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    of = (ck.get("config") or {}).get("obs_fields")
    if of is None:
        raise SystemExit(f"{path} has no config['obs_fields']; cannot recover the observation layout")
    return json.loads(of) if isinstance(of, str) else of


# ---------------------------------------------------------------------------------------------------
# THE FOUR EVENTS THAT MATTER. Everything else in the 1033-d observation is context; these are what the
# reward pays for (AGENTS 34) and what a dream must get right for dream-PPO to transfer:
#
#   boss_damage   gohma_hp_frac goes DOWN        link_damage   link_hp_frac goes DOWN
#   boss_death    gohma_hp_frac reaches 0        link_death    link_hp_frac reaches 0
#
# Missing one and hallucinating one are both fatal and in opposite directions: a model that never fires
# damage gives the dream a boss that cannot be hurt, and one that fires constantly gives a dream where
# flailing kills her. They are therefore weighted separately -- `miss` on the ticks where the event really
# happened, `halluc` on the ticks where it did not -- on top of a per-column weight that lifts these four
# columns above the other 1029.
# ---------------------------------------------------------------------------------------------------
KEY_EVENTS = ("boss_damage", "link_damage", "boss_death", "link_death")


def key_event_masks(obs: "torch.Tensor", g_hp: int | None, l_hp: int | None, eps: float = 1e-6) -> dict:
    """Per-step boolean masks for the four key events, from a (B,T,D) observation tensor.

    Returns masks shaped (B, T-1): index t is "this event happens between t and t+1". `eps` is the drop
    that counts as damage: 1e-6 for the TRAINING weights (every real change is an event) and a real
    magnitude for the EVENT MEASUREMENT, where a predicted drift of 1e-4 on every tick must not be scored
    as a hallucinated hit -- a boss hit is 0.1 of the bar, so 0.01 separates a hit from numerical drift
    while staying an order of magnitude below the smallest real event.
    """
    out = {}
    for name, col in (("boss", g_hp), ("link", l_hp)):
        if col is None:
            continue
        cur, nxt = obs[:, :-1, col], obs[:, 1:, col]
        out[f"{name}_damage"] = (nxt - cur) < -eps
        out[f"{name}_death"] = (cur > eps) & (nxt <= eps)
    return out


def column_weight(obs_dim: int, key_cols: list[int], key_w: float, device=None) -> "torch.Tensor":
    """1.0 on every column, `key_w` on the HP columns, so their error is not averaged into 1033 fields."""
    import torch
    w = torch.ones(obs_dim, device=device)
    for c in key_cols:
        if c is not None:
            w[c] = key_w
    return w


def gate_pos_weight(obs_dim: int, key_cols: list[int], pos_w: float, device=None) -> "torch.Tensor":
    """`pos_weight` for the gate's BCE, raised on the key columns only.

    DEVIATION, and a requested one: nothing in the reference Dyna implementation oversamples rare events.
    It is here because the key events occur on under 1% of ticks, where plain BCE is minimised by a gate
    that never fires -- measured on the live run: boss-HP gate precision 0.000, recall 0.000, AUC 0.495.
    The user asked for missing boss/Link damage and death to be penalised heavily; this is that penalty on
    the firing decision, and `column_weight` is the same on the magnitude.
    """
    import torch
    w = torch.ones(obs_dim, device=device)
    for c in key_cols:
        if c is not None:
            w[c] = pos_w
    return w


def step_weight(msk: "torch.Tensor", masks: dict, event_w: float, death_w: float) -> "torch.Tensor":
    """Per-tick weight: `event_w` on any damage tick, `death_w` on a death tick, 1 elsewhere, 0 on padding."""
    import torch
    w = torch.ones_like(msk)
    dmg = None
    dth = None
    for k, m in masks.items():
        t = m.float()
        if k.endswith("_damage"):
            dmg = t if dmg is None else torch.maximum(dmg, t)
        else:
            dth = t if dth is None else torch.maximum(dth, t)
    if dmg is not None:
        w[:, :-1] = torch.maximum(w[:, :-1], 1.0 + (event_w - 1.0) * dmg)
    if dth is not None:
        w[:, :-1] = torch.maximum(w[:, :-1], 1.0 + (death_w - 1.0) * dth)
    return w * msk


def pad_episodes(group, obs_dim, device, T_max=2000):
    """(B,T,D) obs, (B,T) actions, (B,T) mask -- the one padding used by the trainer and every eval."""
    import torch
    T = min(max(len(e["obs"]) for e in group), T_max)
    B = len(group)
    obs = torch.zeros(B, T, obs_dim); act = torch.zeros(B, T, dtype=torch.long); msk = torch.zeros(B, T)
    for b, e in enumerate(group):
        n = min(len(e["obs"]), T)
        obs[b, :n] = torch.from_numpy(e["obs"][:n]); act[b, :n] = torch.from_numpy(e["act"][:n])
        msk[b, :n] = 1.0
    return obs.to(device), act.to(device), msk.to(device)


def event_accuracy(model, scales, episodes, g_hp, l_hp, device, max_steps=600, batch=8,
                   obs_dim=None, event_eps=0.01):
    """MISSED vs HALLUCINATED, per key event, one step ahead and teacher-forced.

    The model's predicted next observation is run through the SAME predicate as the ground truth
    (`key_event_masks`), so "the model says the boss took damage" means exactly what it means of the real
    transition, and the two failure directions separate:

        missed       the event happened and the model's own next-state does not contain it
        hallucinated the model's next-state contains an event that did not happen

    Reported per event: recall (1 - missed rate), precision, and hallucinations per 1,000 quiet ticks --
    because at a sub-1% event rate a model can hold 100% recall and still make the dream unusable by firing
    on everything.
    """
    import torch
    obs_dim = obs_dim or int(model.obs_dim)
    was_training = model.training
    model.eval()
    agg = {k: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "mag_err": [], "mag_true": []} for k in KEY_EVENTS}
    drift = {"boss": [0.0, 0], "link": [0.0, 0]}            # phantom movement on QUIET ticks
    with torch.no_grad():
        for i in range(0, len(episodes), batch):
            group = episodes[i:i + batch]
            if len(group) < 1:
                break
            obs, act, msk = pad_episodes(group, obs_dim, device)
            h = model.init_hidden(obs.shape[0], device)
            for t in range(min(obs.shape[1] - 1, max_steps)):
                d, _, h = model.step(obs[:, t], act[:, t], h)
                pred_next = obs[:, t] + d * scales
                truth = key_event_masks(torch.stack([obs[:, t], obs[:, t + 1]], 1), g_hp, l_hp,
                                        eps=event_eps)
                pred = key_event_masks(torch.stack([obs[:, t], pred_next], 1), g_hp, l_hp, eps=event_eps)
                live = msk[:, t + 1] > 0.5
                for nm, col in (("boss", g_hp), ("link", l_hp)):
                    if col is None:
                        continue
                    quiet = live & ((obs[:, t + 1, col] - obs[:, t, col]).abs() <= event_eps)
                    if bool(quiet.any()):
                        drift[nm][0] += float((pred_next[:, col] - obs[:, t, col])[quiet].abs().sum())
                        drift[nm][1] += int(quiet.sum())
                for k in truth:
                    if k not in agg:
                        continue
                    y = truth[k][:, 0] & live
                    yh = pred[k][:, 0] & live
                    agg[k]["tp"] += int((y & yh).sum()); agg[k]["fn"] += int((y & ~yh).sum())
                    agg[k]["fp"] += int((~y & yh & live).sum()); agg[k]["tn"] += int((~y & ~yh & live).sum())
                    if k.endswith("_damage") and bool(y.any()):
                        col = g_hp if k.startswith("boss") else l_hp
                        real = obs[:, t + 1, col] - obs[:, t, col]
                        agg[k]["mag_err"].extend((pred_next[:, col] - obs[:, t + 1, col]).abs()[y].cpu().tolist())
                        agg[k]["mag_true"].extend(real.abs()[y].cpu().tolist())
    if was_training:
        model.train()
    out = {}
    for k, v in agg.items():
        n_ev = v["tp"] + v["fn"]
        quiet = v["fp"] + v["tn"]
        out[k] = {
            "n_events": n_ev, "n_ticks": n_ev + quiet,
            "recall": v["tp"] / max(n_ev, 1), "missed": v["fn"] / max(n_ev, 1),
            "precision": v["tp"] / max(v["tp"] + v["fp"], 1),
            "halluc_per_1k_quiet": 1000.0 * v["fp"] / max(quiet, 1),
            "mae_on_event": float(np.mean(v["mag_err"])) if v["mag_err"] else float("nan"),
            "mean_true_delta": float(np.mean(v["mag_true"])) if v["mag_true"] else float("nan")}
        out[k]["event_skill"] = (1 - out[k]["mae_on_event"] / out[k]["mean_true_delta"]
                                 if v["mag_true"] else float("nan"))
        nm = "boss" if k.startswith("boss") else "link"
        out[k]["phantom_drift_per_tick"] = (drift[nm][0] / drift[nm][1]) if drift[nm][1] else float("nan")
        out[k]["event_eps"] = event_eps
    return out


class ObsWM(nn.Module):
    """GRU over (obs, action) predicting the normalised delta to the next obs."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 1024, gru: int = 1024, n_btn: int = 12):
        super().__init__()
        self.obs_dim, self.n_actions, self.n_btn = obs_dim, n_actions, n_btn
        self.act_emb = nn.Embedding(n_actions, 64)
        # the action's GEOMETRY, not just its index (docs/WM_MOTION.md change 2)
        self.enc = nn.Sequential(nn.Linear(obs_dim + 64 + 2, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, gru)
        self.head = nn.Linear(gru, obs_dim)
        # THE MODEL IS PERSISTENCE BY CONSTRUCTION, and has to learn what changes before it can move
        # anything. `gate` predicts P(this field changes this step); the delta is gate * value, so with the
        # gate closed the prediction is EXACTLY the previous observation. At init the value head is zero and
        # the gate bias is -6 (sigmoid 0.0025), i.e. the model IS the identity.
        #
        # This is also the fix for rollout divergence. Most of the 1033 columns are static on any given
        # step; a small per-step error on each compounds, which is how epoch 1 read H1 0.234 (level with
        # persistence) but H100 18.71 against persistence's 0.67. Under rollout the gate is HARD: a field
        # the model does not believe changes is copied forward unchanged and cannot drift at all.
        self.gate = nn.Linear(gru, obs_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -6.0)

    def act_feat(self, a: torch.Tensor) -> torch.Tensor:
        d = (a // self.n_btn).clamp(max=N_DIRS - 1).float() * (2 * math.pi / N_DIRS)
        return torch.stack([torch.sin(d), torch.cos(d)], -1)

    def step(self, obs: torch.Tensor, a: torch.Tensor, h: torch.Tensor, hard: bool = False):
        """Returns (delta, gate_logits, h). `hard=True` copies un-gated fields forward EXACTLY."""
        x = torch.cat([obs, self.act_emb(a), self.act_feat(a)], -1)
        h = self.gru(self.enc(x), h)
        g = self.gate(h)
        val = self.head(h)
        p = torch.sigmoid(g)
        keep = (p > 0.5).float() if hard else p
        return val * keep, g, h

    def init_hidden(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, self.gru.hidden_size, device=device)




def load_wm(path, device="cpu"):
    """Reconstruct the WM and its per-field scales from a checkpoint written by train_wm_obs.py."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ObsWM(int(ck["obs_dim"]), int(ck["n_actions"]), int(ck["hidden"]), int(ck["gru"])).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    scales = torch.as_tensor(ck["scales"], dtype=torch.float32, device=device)
    return m, scales, ck
