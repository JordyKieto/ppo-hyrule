"""Is the competent model-free policy's own representation a better predictive surface than our encoder?

    .venv/bin/python scripts/eval_policy_latent.py

The premise: `checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt` reached return 55.5 / 76% success on Deku
Baba combat. Whatever it represents internally is demonstrably sufficient to ACT well. Our BC/joint encoder is
only known to be sufficient to DECODE well. So take the policy's internals as a frozen encoder and ask the
one question the world model cares about: given (z_t, a_t), how predictable is z_{t+1}?

Two candidate latents, both 256-d:

  proj  tanh(W [global | mean-pool | max-pool]) -- memoryless, a pure function of obs_t
  gru   GRUCell(proj, h_{t-1})                  -- recurrent, carries history

THREE MEASUREMENTS, because predictability alone is meaningless:

  1. skill = 1 - mse(pred) / mse(identity). The zero-init residual predictor IS the identity, so skill > 0
     means the predictor beat "nothing changes". This is the same definition train_joint.py reports.
  2. movement ratio = ||z_{t+1} - z_t|| / ||z_i - z_j||. A latent that never moves scores skill ~0 against a
     near-zero denominator and tells you nothing. Every surface here gets the ratio printed next to its skill.
  3. linear decode probe (ridge) for link_pos / yaw / hp / baba_count. A latent can be perfectly predictable
     and carry none of the state the dream's reward reads. Constant is the degenerate optimum of (1).

THE ECHO CAVEAT, which is why this script exists rather than a one-liner. obs_version 4 (GLOBAL_FIELDS_V2)
contains `prev_action_world_dir_sin_cos` and `prev_action_button` -- dims 23..26 are a literal copy of
a_{t-1}. So obs_{t+1} contains a_t, and any encoder over it gets a free, trivially-predictable-from-a_t
component. That is exactly the control echo removed from our own encoder in LATENT_WORLD_MODEL.md §26. Every
latent is therefore measured twice: as the policy actually saw it, and with those four dims zeroed. The gap
between the two IS the echo's contribution to the skill number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bc_decode import load_many  # noqa: E402
from zelda_rl.bc import BCNet, LI, Norm, RI, _split, featurise, wants_v2  # noqa: E402
from zelda_rl.dynamics import build_predictor  # noqa: E402
from zelda_rl.ppo import RecurrentEntityActorCritic  # noqa: E402

POLICY = "checkpoints/kept/deku-combat-gru-v2b-20p_u00120.pt"


def _obs_slices() -> dict:
    """name -> slice into the 483-d obs, derived from GLOBAL_FIELDS_V2 rather than counted by hand.

    Counted by hand first, and the count was wrong by one (22 global dims, not 23), which zeroed half the
    echo and one state_flags bit instead. The field list is right there in zelda_rl.obs; use it.
    """
    from zelda_rl.obs import GLOBAL_FIELDS_V2
    out, o = {}, 0
    for nm, n in GLOBAL_FIELDS_V2:
        out[nm] = slice(o, o + n)
        o += n
    return out


OBS = _obs_slices()
ECHO = (OBS["prev_action_world_dir_sin_cos"], OBS["prev_action_button"])
VEL = (OBS["link_vel"],)
# `time_left` counts down by a fixed amount every step and `goal_*` are constant within an episode, so both
# move (or pin) the latent perfectly predictably without saying anything about game dynamics. If skill comes
# from them, the "policy latent is a better surface" claim is a clock artefact.
CLOCK = (OBS["time_left"], OBS["goal_pos"], OBS["goal_delta"], OBS["goal_dist"], OBS["goal_rupees"],
         OBS["rupee_gap"], OBS["same_scene_as_goal"], OBS["goal_flags_missing_extra"])
ABLATIONS = {"as-trained": (), "echo-zeroed": ECHO, "echo+vel-zeroed": ECHO + VEL,
             "echo+clock-zeroed": ECHO + CLOCK, "echo+clock+vel": ECHO + CLOCK + VEL}


def policy_latents(ck_path: Path, obs: np.ndarray, epid: np.ndarray, zero: tuple = (),
                   trained: bool = True):
    """Run the frozen policy over each episode in order; return (proj output, GRU hidden) per frame.

    `trained=False` keeps the architecture and randomises the weights. That is THE control for the premise:
    if an untrained network of the same shape is just as predictable and just as decodable, then none of the
    result comes from the policy having been competent -- it comes from the shape of the map.
    """
    ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    c = ck["config"]
    m = RecurrentEntityActorCritic(c["obs_dim"] - c["n_actors"] * c["actor_dim"], c["n_actors"],
                                   c["actor_dim"], c["n_actions"], hidden=c["hidden"], gru=c["gru"])
    if trained:
        m.load_state_dict(ck["model"])
    else:
        torch.manual_seed(0)
    m.eval()
    ob = obs.copy()
    for sl in zero:
        ob[:, sl] = 0.0
    P = np.zeros((len(ob), c["gru"]), np.float32)
    H = np.zeros((len(ob), c["gru"]), np.float32)
    bounds = np.r_[0, np.flatnonzero(np.diff(epid) != 0) + 1, len(ob)]
    with torch.no_grad():
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            x = torch.from_numpy(ob[lo:hi])
            p = m.proj(m.enc._features(x))                 # memoryless
            P[lo:hi] = p.numpy()
            h = m.init_hidden(1)
            for t in range(hi - lo):                       # GRU must run sequentially
                h = m.gru(p[t:t + 1], h)
                H[lo + t] = h.numpy()[0]
    return P, H, c["n_actions"]


def our_latents(ck_path: Path, state: np.ndarray):
    ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    v2 = wants_v2(ck["link_dim"])
    T = Norm.load(ck["norm"]).apply(featurise(state, link_v2=v2, control_echo=False))
    enc = BCNet(link_dim=T["link"].shape[1], row_dim=T["rows"].shape[2], n_actions=ck["n_actions"],
                latent=ck["latent"], gru=0)
    enc.load_state_dict(ck["encoder"])
    enc.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(state), 8192):
            sl = slice(i, i + 8192)
            out.append(enc.encode(torch.from_numpy(T["link"][sl]), torch.from_numpy(T["rows"][sl]),
                                  torch.from_numpy(T["ids"][sl]), torch.from_numpy(T["mask"][sl])).numpy())
    return np.concatenate(out)


class ShallowEnc(nn.Module):
    """The policy `proj` shape, over OUR v3 state features instead of the 483-d obs.

    One linear row encoder + mean/max pool + one linear layer + tanh. Deliberately untrained. It isolates the
    three candidate explanations for the policy latent's skill: the observation it reads, the shape of the map,
    or the training that produced the weights. Same features as our encoder + same shape as the policy's
    = only the shape differs from `ours`; same shape as the policy + different features = only the input does.
    """

    def __init__(self, link_dim: int, row_dim: int, latent: int = 256, row_hidden: int = 96, id_emb: int = 8,
                 deep: bool = False, layernorm: bool = False):
        super().__init__()
        self.id_embedding = nn.Embedding(256, id_emb)
        self.row_enc = nn.Sequential(nn.Linear(row_dim + id_emb, row_hidden), nn.Tanh())
        w = link_dim + 2 * row_hidden
        self.trunk = (nn.Sequential(nn.Linear(w, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, latent)) if deep
                      else nn.Sequential(nn.Linear(w, latent), nn.Tanh()))
        self.ln = nn.LayerNorm(latent) if layernorm else nn.Identity()

    def forward(self, link, rows, ids, mask):
        e = self.id_embedding(ids.clamp(0, 255))
        m = mask.unsqueeze(-1)
        h = self.row_enc(torch.cat([rows, e], -1)) * m
        mean = h.sum(-2) / m.sum(-2).clamp(min=1.0)
        mx = (h - (1 - m) * 1e4).max(-2).values * (m.sum(-2) > 0)
        return self.ln(self.trunk(torch.cat([link, mean, mx], -1)))


def shallow_latents(state: np.ndarray, ck_path: Path, **kw):
    ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    T = Norm.load(ck["norm"]).apply(featurise(state, link_v2=wants_v2(ck["link_dim"]), control_echo=False))
    torch.manual_seed(0)
    enc = ShallowEnc(T["link"].shape[1], T["rows"].shape[2], **kw).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(state), 8192):
            sl = slice(i, i + 8192)
            out.append(enc(torch.from_numpy(T["link"][sl]), torch.from_numpy(T["rows"][sl]),
                           torch.from_numpy(T["ids"][sl]), torch.from_numpy(T["mask"][sl])).numpy())
    return np.concatenate(out)


def measure(name: str, Z: np.ndarray, act: np.ndarray, epid: np.ndarray, na: int, probes: dict,
            epochs: int, seed: int = 0) -> dict:
    """Standardise, split by episode, train (z,a)->z' and report skill / movement / decode probes."""
    torch.manual_seed(seed)
    ok = np.diff(epid) == 0                                # t and t+1 in the same episode
    idx = np.flatnonzero(ok)
    eps = np.unique(epid)
    rng = np.random.default_rng(0)
    te_eps = set(rng.choice(eps, max(1, len(eps) // 5), replace=False).tolist())
    is_te = np.isin(epid[idx], list(te_eps))
    tr, te = idx[~is_te], idx[is_te]

    mu, sd = Z[tr].mean(0), Z[tr].std(0)                   # fit on train only
    sd = np.where(sd < 1e-3, 1.0, sd)                      # dead dims stay dead, not amplified 1e6x

    Zs = ((Z - mu) / sd).astype(np.float32)
    Zt = torch.from_numpy(Zs)
    A = F.one_hot(torch.from_numpy(act.astype(np.int64)).clamp(0, na - 1), na).float()

    # (2) movement: consecutive vs random-pair distance, on the test split
    d_con = float(np.abs(Zs[te + 1] - Zs[te]).mean())
    d_rnd = float(np.abs(Zs[te] - Zs[rng.permutation(te)]).mean())

    # (1) skill
    pred = build_predictor(Zs.shape[1], na, factored=False)
    opt = torch.optim.Adam(pred.parameters(), lr=3e-4)
    B = 512
    for ep in range(epochs):
        perm = np.random.permutation(tr)
        for i in range(0, len(perm) - B, B):
            b = perm[i:i + B]
            za = torch.cat([Zt[b], A[b]], -1)
            loss = F.mse_loss(pred(za), Zt[b + 1])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pm, im = 0.0, 0.0
        for i in range(0, len(te), 8192):
            b = te[i:i + 8192]
            za = torch.cat([Zt[b], A[b]], -1)
            pm += float(F.mse_loss(pred(za), Zt[b + 1], reduction="sum"))
            im += float(F.mse_loss(Zt[b], Zt[b + 1], reduction="sum"))
    skill = 1 - pm / max(im, 1e-9)

    # (3) ridge decode probes from z alone
    Xtr = np.c_[Zs[tr], np.ones(len(tr), np.float32)]
    Xte = np.c_[Zs[te], np.ones(len(te), np.float32)]
    G = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1], dtype=np.float32)
    pr = {}
    for k, Y in probes.items():
        W = np.linalg.solve(G, Xtr.T @ Y[tr])
        yh, yt = Xte @ W, Y[te]
        ss = ((yt - yt.mean(0)) ** 2).sum()
        pr[k] = (1 - ((yt - yh) ** 2).sum() / max(ss, 1e-9), np.abs(yt - yh).mean())
    return {"name": name, "dim": Zs.shape[1], "skill": skill, "pred_mse": pm / len(te) / Zs.shape[1],
            "id_mse": im / len(te) / Zs.shape[1], "d_con": d_con, "d_rnd": d_rnd,
            "ratio": d_con / max(d_rnd, 1e-9), "probes": pr, "n_tr": len(tr), "n_te": len(te)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["datasets/transitions/v3-main/20260919-084052"])
    ap.add_argument("--ours", default="checkpoints/dynamics/wm_fu_dist.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=200_000)
    a = ap.parse_args()

    meta, state, act, slot, epid, obs, has = load_many(a.data)
    keep = has.copy()
    if keep.sum() > a.max_frames:                          # keep whole episodes, in order
        cut = epid[np.flatnonzero(keep)[a.max_frames]]
        keep &= epid <= cut
    state, act, epid, obs = state[keep], act[keep], epid[keep], obs[keep]
    print(f"{len(state):,} frames with obs over {len(np.unique(epid))} episodes; "
          f"state width {state.shape[1]}, obs width {obs.shape[1]}, n_actions {meta['n_actions']}")

    lk, rows, _ = _split(state)
    present = (rows[:, :, RI["present"]] > 0.5)
    probes = {
        "link_pos": lk[:, [LI["pos_x"], LI["pos_y"], LI["pos_z"]]].astype(np.float32),
        "link_yaw": np.stack([np.sin(lk[:, LI["yaw"]] / 32768.0 * np.pi),
                              np.cos(lk[:, LI["yaw"]] / 32768.0 * np.pi)], 1).astype(np.float32),
        "link_hp": lk[:, [LI["hp"]]].astype(np.float32),
        "baba_count": present.sum(1, keepdims=True).astype(np.float32),
    }

    rows_out = []
    for tag, zero in ABLATIONS.items():
        P, H, na = policy_latents(Path(POLICY), obs, epid, zero=zero)
        for nm, Z in (("policy proj", P), ("policy gru-h", H)):
            rows_out.append(measure(f"{nm} ({tag})", Z, act, epid, na, probes, a.epochs))
            r = rows_out[-1]
            print(f"  {r['name']:<28} skill {r['skill']:+.3f}  move-ratio {r['ratio']:.3f}")

    # Controls: same architecture untrained, and the raw observation itself.
    P0, H0, na = policy_latents(Path(POLICY), obs, epid, zero=ECHO + CLOCK, trained=False)
    for nm, Z in (("UNTRAINED proj", P0), ("UNTRAINED gru-h", H0)):
        rows_out.append(measure(f"{nm} (echo+clock)", Z, act, epid, na, probes, a.epochs))
        r = rows_out[-1]
        print(f"  {r['name']:<28} skill {r['skill']:+.3f}  move-ratio {r['ratio']:.3f}")
    ob = obs.copy()
    for sl in ECHO + CLOCK:
        ob[:, sl] = 0.0
    rows_out.append(measure("raw obs (echo+clock)", ob, act, epid, meta["n_actions"], probes, a.epochs))
    r = rows_out[-1]
    print(f"  {r['name']:<28} skill {r['skill']:+.3f}  move-ratio {r['ratio']:.3f}")

    if Path(a.ours).exists():
        for nm, kw in (("OUR feats, shallow-tanh", {}),
                       ("OUR feats, deep+LN (untrained)", {"deep": True, "layernorm": True})):
            rows_out.append(measure(nm, shallow_latents(state, Path(a.ours), **kw), act, epid,
                                    meta["n_actions"], probes, a.epochs))
            r = rows_out[-1]
            print(f"  {r['name']:<28} skill {r['skill']:+.3f}  move-ratio {r['ratio']:.3f}")

        Z = our_latents(Path(a.ours), state)
        rows_out.append(measure("ours (frozen enc)", Z, act, epid, meta["n_actions"], probes, a.epochs))
        r = rows_out[-1]
        print(f"  {r['name']:<28} skill {r['skill']:+.3f}  move-ratio {r['ratio']:.3f}")

    print(f"\n{'latent':<30} {'dim':>4} {'skill':>7} {'pred':>8} {'ident':>8} {'move/rand':>10}"
          f" {'pos R2':>7} {'pos MAE':>8} {'yaw R2':>7} {'hp R2':>7} {'nact R2':>8}")
    for r in rows_out:
        p = r["probes"]
        print(f"{r['name']:<30} {r['dim']:>4} {r['skill']:>+7.3f} {r['pred_mse']:>8.4f} {r['id_mse']:>8.4f}"
              f" {r['ratio']:>10.3f} {p['link_pos'][0]:>7.3f} {p['link_pos'][1]:>8.1f}"
              f" {p['link_yaw'][0]:>7.3f} {p['link_hp'][0]:>7.3f} {p['baba_count'][0]:>8.3f}")
    print("\nskill = 1 - pred_mse/identity_mse on held-out episodes; identity is the zero-init residual.")
    print("move/rand = mean |z_{t+1}-z_t| / mean |z_i-z_j|: low means the latent barely moves per step,")
    print("            which makes skill easy and the surface uninformative. Read the two columns together.")
    print("probes are RIDGE (linear) from z alone -- a floor on decodability, not our MLP decoder's number.")


if __name__ == "__main__":
    main()
