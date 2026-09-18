"""Predict reward events directly from a latent transition, instead of differencing a decoded state.

    .venv/bin/python scripts/train_event_head.py

    input   [ z_t || one-hot action || z_{t+1} || z_{t+1} - z_t ]
    heads   damage 0/1/2/3+   kills 0/1/2/3   Link damage (binary)   terminal none/death/cleared
    labels  computed exactly from the real dataset: per-row Baba HP drops, rows crossing to zero,
            Link HP drops, and episode terminals

Why this rather than reading HP out of the decoded state and subtracting:

  * differencing a decoded signal is unsound when the latent comes from a drifting model. Clamping HP to be
    monotone (true on real data: Baba HP rises in 0 of 891,459 real row-steps) turns drift into permanent,
    unearned kills; NOT clamping lets a policy farm the same enemy forever by steering into a region where
    the decode oscillates. Both were observed.
  * a calibrated probability is bounded by construction. Expected reward under this head cannot exceed the
    real payout for an event, however wrong the latent is.

Trained and scored on REAL latent pairs, so the numbers here are comparable to the decoder-differencing
route: AP 0.887 damage / 0.895 kill / 0.827 Link damage on the same held-out slots.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.bc import (LINK_HP_VALUES, ROW_HP_VALUES, BCNet, Norm, decode_targets, featurise,  # noqa: E402
                         hp_class)
from scripts.train_bc_decode import DEFAULT_DATA  # noqa: E402

DMG_CLASSES, KILL_CLASSES, TERM_CLASSES = 4, 4, 3          # 3+ is the top damage bucket


class EventHead(nn.Module):
    def __init__(self, latent, n_actions, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(latent * 3 + n_actions, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.dmg = nn.Linear(hidden, DMG_CLASSES)
        self.kill = nn.Linear(hidden, KILL_CLASSES)
        self.link = nn.Linear(hidden, 1)
        self.term = nn.Linear(hidden, TERM_CLASSES)

    def forward(self, z, a1h, z2):
        h = self.trunk(torch.cat([z, a1h, z2, z2 - z], -1))
        return self.dmg(h), self.kill(h), self.link(h).squeeze(-1), self.term(h)

    @torch.no_grad()
    def reward(self, z, a1h, z2, w_dmg=1.0, w_kill=5.0, w_link=0.25, hp_unit=8.0,
               death=20.0, success=20.0, time_pen=0.001):
        """Expected reward under the predicted event distribution. Bounded by construction."""
        d, k, l, t = self(z, a1h, z2)
        e_d = (d.softmax(-1) * torch.arange(DMG_CLASSES, dtype=torch.float)).sum(-1)
        e_k = (k.softmax(-1) * torch.arange(KILL_CLASSES, dtype=torch.float)).sum(-1)
        pt = t.softmax(-1)
        r = w_dmg * e_d + w_kill * e_k - w_link * hp_unit * l.sigmoid() - time_pen
        r = r - death * pt[..., 1] + success * pt[..., 2]
        return r, pt[..., 1], pt[..., 2]


def load_with_terminal(data_dir: Path):
    """Like scripts.train_bc_decode.load, but keeps the FINAL state of each episode.

    That loader stores state[:T] for T actions, so the last transition of every episode -- the one that
    carries the death or the room-clear -- is dropped. Terminals are the two largest reward terms, so the
    event head needs them. Returns flat states plus explicit (from, action, to) transition triples.
    """
    meta = json.loads((data_dir / "dataset.json").read_text())
    rows = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    S, frm, acts, slots = [], [], [], []
    off = 0
    for r in rows:
        d = np.load(data_dir / r["file"])
        a_, st = d["action"], d["state"]
        T = len(a_)
        st = st[:T + 1]                                   # T+1 states: every transition is representable
        S.append(st)
        n = len(st) - 1                                   # == T when the episode stored its final state
        frm.append(np.arange(off, off + n))
        acts.append(a_[:n]); slots.append(np.full(n, r["slot"], np.int64))
        off += len(st)
    return (meta, np.concatenate(S), np.concatenate(frm), np.concatenate(acts), np.concatenate(slots))


def labels(Y, frm, action):
    """Event labels for each explicit transition frm[i] -> frm[i]+1."""
    i, j = frm, frm + 1
    lhp = Y["link_hp_c"]
    rhp = np.where((Y["row_present"] > .5) & (Y["row_is_baba"] > .5), Y["row_hp_c"], 0)
    drop = np.clip(rhp[i] - rhp[j], 0, None).sum(-1)
    kill = ((rhp[i] > 0) & (rhp[j] == 0)).sum(-1)
    link = (lhp[j] < lhp[i]).astype(np.int64)
    term = np.zeros(len(i), np.int64)
    term[lhp[j] <= 0] = 1                                        # death
    term[(rhp[j].sum(-1) == 0) & (rhp[i].sum(-1) > 0)] = 2       # room cleared
    return {"dmg": np.clip(drop, 0, DMG_CLASSES - 1), "kill": np.clip(kill, 0, KILL_CLASSES - 1),
            "link": link, "term": term}


def ap(score, truth):
    o = np.argsort(-score); t = truth[o]
    tp, fp = np.cumsum(t), np.cumsum(~t)
    p, r = tp / np.maximum(tp + fp, 1), tp / max(truth.sum(), 1)
    return float(np.sum(np.diff(np.r_[0, r]) * p))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--model", default="/tmp/joint_3e-4.pt")
    ap_.add_argument("--data", default=DEFAULT_DATA)
    ap_.add_argument("--epochs", type=int, default=25)
    ap_.add_argument("--batch", type=int, default=1024)
    ap_.add_argument("--lr", type=float, default=1e-3)
    ap_.add_argument("--hidden", type=int, default=512)
    ap_.add_argument("--patience", type=int, default=5)
    ap_.add_argument("--out", default="checkpoints/dynamics/event_head.pt")
    ap_.add_argument("--torch-threads", type=int, default=6)
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    norm = Norm.load(ck["norm"])

    meta, state, frm, action, slot = load_with_terminal(Path(a.data))
    Y = decode_targets(state)
    Y["link_hp_c"] = hp_class(Y["link_hp"][:, 0], LINK_HP_VALUES)
    Y["row_hp_c"] = hp_class(Y["row_hp"], ROW_HP_VALUES)
    L = labels(Y, frm, action)
    Fn = norm.apply(featurise(state))
    with torch.no_grad():
        Z = torch.cat([enc.encode(torch.from_numpy(Fn["link"][s:s + 8192]),
                                  torch.from_numpy(Fn["rows"][s:s + 8192]),
                                  torch.from_numpy(Fn["ids"][s:s + 8192]),
                                  torch.from_numpy(Fn["mask"][s:s + 8192]))
                       for s in range(0, len(state), 8192)])
    A1 = F.one_hot(torch.from_numpy(action), NA).float()
    test, tune = set(ck["split"]["val_slots"]), set(ck["split"]["tune_slots"])
    valid = np.arange(len(frm))
    tr = valid[~np.isin(slot, list(test | tune))]
    tu = valid[np.isin(slot, list(tune))]
    te = valid[np.isin(slot, list(test))]
    FR = torch.from_numpy(frm)
    Yt = {k: torch.from_numpy(v) for k, v in L.items()}
    print(f"train {len(tr):,} tune {len(tu):,} test {len(te):,} transitions")
    for k, n in (("dmg", DMG_CLASSES), ("kill", KILL_CLASSES), ("term", TERM_CLASSES)):
        c = np.bincount(L[k][tr], minlength=n)
        print(f"  {k:<5} class counts (train): {c.tolist()}")
    print(f"  link  positives (train): {int(L['link'][tr].sum())} of {len(tr):,}\n")

    head = EventHead(LAT, NA, a.hidden)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    cw = {k: torch.tensor(((lambda c: (c.sum() / np.maximum(c, 1)) ** 0.5)(
        np.bincount(L[k][tr], minlength=n).astype(float)) /
        ((lambda c: (c.sum() / np.maximum(c, 1)) ** 0.5)(
            np.bincount(L[k][tr], minlength=n).astype(float))).mean()).astype(np.float32))
        for k, n in (("dmg", DMG_CLASSES), ("kill", KILL_CLASSES), ("term", TERM_CLASSES))}
    pw = torch.tensor([(len(tr) - L["link"][tr].sum()) / max(L["link"][tr].sum(), 1)], dtype=torch.float)

    def fwd(i):
        f = FR[i]
        return head(Z[f], A1[i], Z[f + 1])

    def loss_fn(i):
        d, k, l, t = fwd(i)
        return (F.cross_entropy(d, Yt["dmg"][i], weight=cw["dmg"])
                + F.cross_entropy(k, Yt["kill"][i], weight=cw["kill"])
                + F.binary_cross_entropy_with_logits(l, Yt["link"][i].float(), pos_weight=pw)
                + F.cross_entropy(t, Yt["term"][i], weight=cw["term"]))

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = rng.permutation(tr)
        tot = n = 0.0
        for s in range(0, len(perm), a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            l = loss_fn(i)
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item(); n += 1
        head.eval()
        with torch.no_grad():
            vl = float(loss_fn(torch.from_numpy(tu)))
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:>3}/{a.epochs}  train {tot/n:.4f}  tune {vl:.4f}"
                  f"{'  *' if bad == 0 else ''}  ({time.time()-t0:.0f}s)", flush=True)
        if bad >= a.patience:
            print(f"  early stop at epoch {ep}")
            break
    head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        d, k, l, t = fwd(torch.from_numpy(te))
    res = {}
    print(f"\n{'event':<14} {'AP':>7} {'positives':>10}   (decoder-differencing AP, same slots)")
    print("-" * 66)
    for name, score, truth, prev in [
            ("damage", 1 - d.softmax(-1)[:, 0].numpy(), L["dmg"][te] > 0, 0.887),
            ("kill", 1 - k.softmax(-1)[:, 0].numpy(), L["kill"][te] > 0, 0.895),
            ("link damage", l.sigmoid().numpy(), L["link"][te] > 0, 0.827),
            ("death", t.softmax(-1)[:, 1].numpy(), L["term"][te] == 1, None),
            ("cleared", t.softmax(-1)[:, 2].numpy(), L["term"][te] == 2, None)]:
        v = ap(score, truth)
        res[name] = {"ap": v, "positives": int(truth.sum())}
        print(f"{name:<14} {v:>7.3f} {int(truth.sum()):>10}   {'' if prev is None else f'{prev:.3f}'}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "config": vars(a), "model": a.model, "latent": LAT,
                "n_actions": NA, "metrics": res}, a.out)
    Path("results/event_head.json").write_text(json.dumps({"config": vars(a), "metrics": res}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
