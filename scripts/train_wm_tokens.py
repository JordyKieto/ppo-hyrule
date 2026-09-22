"""Action-conditioned next-STATE world model over entity tokens. No latent rollout, no CNN, starts at persistence.

    .venv/bin/python scripts/train_wm_tokens.py --epochs 3

    S_t (entity tokens) + a_t  ->  Delta S_{t+1}, per entity, per field, typed heads

Every design choice here is a defence against a specific pathology of putting structured game telemetry through
attention, rather than an NLP default:

1. NO DEGENERATE SOFTMAX ROW. A learnable NO-OP token is appended to every key/value pool and is never masked,
   so no query can ever face an all-masked row. `softmax([-inf,...])` is 0/0 = NaN, and the -1e9 workaround
   still backprops through a numeric cliff of 1e4-1e6. Masking uses -1e4, and invalid keys are hard-zeroed
   AFTER the softmax as well.
2. NO RAW COORDINATES. `zelda_rl.tokens` emits actors egocentric to Link and Link relative to his episode
   start, every continuous field divided by a fitted scale and clipped to [-1,1]. Unnormalised coordinates in a
   query-key dot product drive logits to +-50 and saturate the softmax, where one entity stepping a few units
   flips a saturated weight and spikes the gradient norm.
3. DISTANCE AS A CONTINUOUS LEARNABLE BIAS, NOT A HAND-CODED RADIUS:
       scores_ij = q.k/sqrt(d) - softplus(w_h) * D_ij + b_h
   `w_h` is PER HEAD, so a head can learn a large w_h and become local, or w_h ~ 0 and stay global. No cutoff
   is hardcoded, nothing is hard-masked by range, and it is differentiable throughout.
4. HETEROGENEOUS PROJECTIONS. Link and actors get separate encoders; ids, categories and behaviour states are
   embedding lookups, never floats through a shared linear layer.
5. STARTS AT PERSISTENCE. Every head predicts a DELTA and every output projection is zero-initialised, so at
   step 0 the model predicts "nothing changes" EXACTLY -- including the discrete heads, which use a logit
   residual (`k*(2*bit-1) + zero_init_delta`) so their argmax at init is the current value. Skill is therefore
   exactly 0.000 on the first batch instead of starting from garbage.
6. TYPED SPLIT HEADS. Positions use L1 in scaled units, HP/state/scene use cross-entropy, flags use BCE. A
   single unweighted MSE would let the position term (world-unit scale) drown the flag terms (0/1) -- the
   optimiser would trade away all of combat to shave a fraction off a coordinate.
7. NO ACTION LEAK. Token fields are verified disjoint from `bc.CONTROL_ECHO_FEATS` and the echo sf1 bits are
   excluded; the action enters ONLY concatenated to the query side, never as a token feature.

Gradient norm is logged every epoch (mean and max over steps) because that is the quantity the whole design is
meant to keep finite.
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
from scripts.train_bc_decode import DEFAULT_DATA, load_many  # noqa: E402
from zelda_rl.bc import LI, RI, _split  # noqa: E402
from zelda_rl.navgrid import NavGrid  # noqa: E402
from zelda_rl.tokens import ACTOR_DIM, LINK_DIM, REL_CLIP, SF1_KEEP, build  # noqa: E402

N_STATES, LOGIT_K = 10, 6.0
# Delta-cell window: 5x5 cells, so a step may move at most 2 cells on either axis (at n=32 that is ~58 x 89
# world units, far beyond the 13-unit moving step). Index 12 is (0,0) = SAME CELL, the persistence class.
DW, DC = 5, 25
CENTRE = (DW // 2) * DW + DW // 2      # LOGIT_K: how hard the persistence prior is asserted on discrete heads


class Attn(nn.Module):
    """Queries are entities; keys/values are entities plus an always-valid NO-OP token.

    Per-head learnable distance bias; -1e4 masking; invalid keys hard-zeroed post-softmax.
    """

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.o = nn.Linear(d, d)
        self.w = nn.Parameter(torch.zeros(heads))          # softplus(0)=0.693 -> mild locality at init
        self.b = nn.Parameter(torch.zeros(heads))
        self.noop = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x, mask, D):
        """x (B,N,d); mask (B,N) 1=valid; D (B,N,N) pairwise distance in scaled units."""
        B, N, d = x.shape
        kv = torch.cat([x, self.noop.expand(B, 1, d)], 1)                  # NO-OP: row can never be empty
        kmask = torch.cat([mask, torch.ones(B, 1, device=x.device)], 1)
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(B, N + 1, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(B, N + 1, self.h, self.dk).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / self.dk ** 0.5
        Dp = torch.cat([D, torch.zeros(B, N, 1, device=x.device)], -1)     # no-op sits at distance 0
        s = s - F.softplus(self.w)[None, :, None, None] * Dp[:, None] + self.b[None, :, None, None]
        s = s.masked_fill(kmask[:, None, None, :] < 0.5, -1e4)             # -1e4, not -1e9
        a = s.softmax(-1) * kmask[:, None, None, :]                        # hard zero padded keys post-softmax
        return self.o((a @ v).transpose(1, 2).reshape(B, N, d))


class TokenWM(nn.Module):
    def __init__(self, n_actions: int, d: int = 128, heads: int = 4, n_rows: int = 16, id_vocab: int = 256):
        super().__init__()
        self.n_rows = n_rows
        self.id_emb, self.cat_emb, self.st_emb = nn.Embedding(id_vocab, 8), nn.Embedding(16, 4), nn.Embedding(N_STATES, 4)
        # Position as EMBEDDINGS, not floats: a cell index has no magnitude to leak into a query-key dot product.
        self.cell_emb = nn.Embedding(64, 12)                      # Link's absolute cell, per axis
        self.rcell_emb = nn.Embedding(2 * REL_CLIP + 1, 12)       # actor cell RELATIVE to Link, per axis
        self.link_enc = nn.Sequential(nn.Linear(LINK_DIM + 24 + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.actor_enc = nn.Sequential(nn.Linear(ACTOR_DIM + 16 + 24 + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.attn = Attn(d, heads)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.act_emb = nn.Sequential(nn.Linear(n_actions, d), nn.ReLU())    # action: QUERY-SIDE ONLY
        # position OUT is dcell (25-way classification) + doffset (2, bounded) -- never a coordinate
        self.link_head = nn.Linear(2 * d, DC + 2 + 1 + 2 + 3 + len(SF1_KEEP) + 2)
        self.actor_head = nn.Linear(2 * d, DC + 2 + 3 + 1 + N_STATES)
        for h in (self.link_head, self.actor_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)                                          # PERSISTENCE at step 0

    def forward(self, T, a_onehot):
        link, actors, mask = T["link"], T["actors"], T["mask"]
        B = link.shape[0]
        e = torch.cat([self.id_emb(T["ids"]), self.cat_emb(T["cats"]), self.st_emb(T["states"])], -1)
        lce = self.cell_emb(T["link_cell"]).flatten(-2)
        rce = self.rcell_emb(T["actor_rcell"]).flatten(-2)
        lt = self.link_enc(torch.cat([link, lce, T["link_off"]], -1))[:, None] + self.type_emb[0]
        at = self.actor_enc(torch.cat([actors, e, rce, T["actor_off"]], -1)) + self.type_emb[1]
        x = torch.cat([lt, at], 1)
        m = torch.cat([torch.ones(B, 1, device=link.device), mask], 1)
        pos = torch.cat([torch.zeros(B, 1, 2, device=link.device), T["cell_xy"]], 1)
        D = (pos[:, :, None, :] - pos[:, None, :, :]).norm(dim=-1)
        x = self.ln1(x + self.attn(x, m, D))
        x = self.ln2(x + self.ff(x))
        ac = self.act_emb(a_onehot)[:, None].expand(-1, x.shape[1], -1)     # action conditions the QUERY
        h = torch.cat([x, ac], -1)
        return self.link_head(h[:, 0]), self.actor_head(h[:, 1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--ref", default="checkpoints/bc/bc-decode-final.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grid", type=int, default=32, help="nav-grid resolution (cell ~29x45 world units at 32)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dcell-weight", action="store_true",
                    help="class-weight the dcell head. OFF by default: sqrt-inverse weights over 25 classes "
                         "where 94%% are 'same cell' give the centre class ~0.01 against ~2.9 for the rare "
                         "ones, so the loss PUNISHES predicting no-movement -- correct 94%% of the time. "
                         "Measured: dcell accuracy 0.966 at epoch 1 collapsing to 0.52 by epoch 12. Rare-event "
                         "weighting is right for HP detection and wrong for position accuracy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--ckpt-dir", default=None)
    # --- fleet: data-parallel FedAvg over per-pod shards ------------------------------------------------
    # Each pod trains on ITS OWN collected shard and weights are averaged, so the fleet sees the whole dataset
    # with no data movement. FleetLink.average_weights is generic over any model with a state_dict.
    ap.add_argument("--fleet-experiment", default=None)
    ap.add_argument("--fleet-rank", type=int, default=0)
    ap.add_argument("--fleet-world-size", type=int, default=1)
    ap.add_argument("--fleet-launch-id", default="")
    ap.add_argument("--fleet-aggregator", type=int, default=0)
    ap.add_argument("--fleet-sync", default="avg", choices=["avg", "none"])
    # CRITICAL for FedAvg: every rank must use IDENTICAL preprocessing. Feature scales, nav-grid bounds and
    # delta scales are all fitted from data, so fitting them per-pod would average weights across pods whose
    # inputs mean different things. Fit once (--save-prep) and ship the JSON to every rank (--prep).
    ap.add_argument("--prep", default=None, help="load fixed scales + nav grid from JSON (required on a fleet)")
    ap.add_argument("--save-prep", default=None, help="fit scales + grid on this data and write the JSON")
    ap.add_argument("--out", default="checkpoints/wm_tokens/wm.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    split = torch.load(a.ref, weights_only=False, map_location="cpu")["split"]
    meta, state, act, slot, epid, obs, has = load_many([a.data, *a.extra_data])
    NA = meta["n_actions"]
    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    tr_m = ~np.isin(slot, list(test | tune))
    lk, rows, _ = _split(state)
    _pres = rows[:, :, RI["present"]] > 0.5
    prep = json.load(open(a.prep)) if a.prep else None
    grid = (NavGrid.load(prep["grid"]) if prep else
            NavGrid.fit(np.concatenate([lk[:, [LI["pos_x"], LI["pos_z"]]],
                                        rows[:, :, [RI["pos_x"], RI["pos_z"]]][_pres]]), n=a.grid))
    print(f"nav grid {grid.n}x{grid.n} | cell {np.round(grid.cell,1)} world units")
    T, sc = build(state, epid, train_mask=tr_m, grid=grid,
                  scales=(prep["scales"] if prep else None))

    # ---- targets, all DELTAS or EVENTS, never absolutes
    nxt = np.minimum(np.arange(len(state)) + 1, len(state) - 1)
    ok = np.r_[epid[1:] == epid[:-1], False]
    lpos = lk[:, [LI["pos_x"], LI["pos_z"]]]
    # DELTA scales, fitted on the DELTAS -- not the position scales. Positions span +-420/810 world units while
    # a single step moves ~1-13, so dividing a delta by the position scale makes the target ~0.002 and the
    # position head's loss numerically negligible beside the other heads. Measured consequence of getting this
    # wrong: dpos error 3.45 -> 4.12 world units over 20 epochs against a persistence baseline of 1.045, i.e.
    # the head drifted while its loss term was too small to matter. Same principle as normalising the inputs,
    # applied to the targets.
    # position targets in the GRID CODEC: which of the 5x5 neighbouring cells, plus the offset change
    lc_n, lo_n = grid.to_cell(lpos[nxt])
    lc_c, lo_c = T["link_cell"], T["link_off"]
    dcell = np.clip(lc_n - lc_c, -(DW // 2), DW // 2) + DW // 2
    dcell_i = (dcell[:, 0] * DW + dcell[:, 1]).astype(np.int64)
    doff = (lo_n - lo_c).astype(np.float32)
    print(f"dcell: SAME CELL on {float((dcell_i[tr_m & ok] == CENTRE).mean()):.1%} of steps | "
          f"doff range [{doff.min():.2f}, {doff.max():.2f}]")
    rdy = (lk[nxt, LI["pos_y"]] - lk[:, LI["pos_y"]])
    sdy = prep["d_link_y"] if prep else float(max(np.percentile(np.abs(rdy[tr_m & ok]), 99.5), 1e-3))
    sc["d_link_y"] = sdy
    dy = np.clip(rdy / sdy, -1, 1).astype(np.float32)[:, None]
    ys, yc = np.sin(lk[:, LI["yaw"]] * 2 * np.pi / 65536.0), np.cos(lk[:, LI["yaw"]] * 2 * np.pi / 65536.0)
    dyaw = np.stack([np.sin(lk[nxt, LI["yaw"]] * 2 * np.pi / 65536.0) - ys,
                     np.cos(lk[nxt, LI["yaw"]] * 2 * np.pi / 65536.0) - yc], 1).astype(np.float32)
    hp_ev = np.where(lk[nxt, LI["hp"]] < lk[:, LI["hp"]], 1, np.where(lk[nxt, LI["hp"]] > lk[:, LI["hp"]], 2, 0))
    sf1 = lk[:, LI["sf1"]].astype(np.int64)
    bits_now = np.stack([((sf1 >> i) & 1) for i in SF1_KEEP], 1).astype(np.float32)
    bits_nxt = np.stack([((sf1[nxt] >> i) & 1) for i in SF1_KEEP], 1).astype(np.float32)
    scene_now = (lk[:, LI["scene"]] != 0).astype(np.int64)
    scene_nxt = (lk[nxt, LI["scene"]] != 0).astype(np.int64)
    rpos = rows[:, :, [RI["pos_x"], RI["pos_z"]]]
    rc_n, ro_n = grid.to_cell(rpos[nxt].reshape(-1, 2))
    rc_n, ro_n = rc_n.reshape(len(state), -1, 2), ro_n.reshape(len(state), -1, 2)
    rc_c, ro_c = grid.to_cell(rpos.reshape(-1, 2))
    rc_c, ro_c = rc_c.reshape(len(state), -1, 2), ro_c.reshape(len(state), -1, 2)
    rdc = np.clip(rc_n - rc_c, -(DW // 2), DW // 2) + DW // 2
    r_dcell_i = (rdc[..., 0] * DW + rdc[..., 1]).astype(np.int64)
    r_doff = (ro_n - ro_c).astype(np.float32)
    r_hp_ev = np.where(rows[nxt, :, RI["hp"]] < rows[:, :, RI["hp"]], 1,
                       np.where(rows[nxt, :, RI["hp"]] > rows[:, :, RI["hp"]], 2, 0))
    pres_now = T["mask"]
    pres_nxt = (rows[nxt, :, RI["present"]] > 0.5).astype(np.float32)
    st_now = T["states"]
    st_nxt = rows[nxt, :, RI["state_idx"]].astype(np.int64).clip(0, N_STATES - 1)

    tt = {k: torch.from_numpy(v) for k, v in T.items() if k != "start_idx"}
    tt["rel_xz"] = torch.from_numpy(np.stack([T["actors"][:, :, 0], T["actors"][:, :, 1]], -1))
    Y = {k: torch.from_numpy(v) for k, v in
         dict(dcell=dcell_i, doff=doff, dy=dy, dyaw=dyaw, hp_ev=hp_ev, bits_now=bits_now, bits_nxt=bits_nxt,
              scene_now=scene_now, scene_nxt=scene_nxt, r_dcell=r_dcell_i, r_doff=r_doff, r_hp_ev=r_hp_ev,
              pres_now=pres_now, pres_nxt=pres_nxt, st_now=st_now, st_nxt=st_nxt).items()}
    A = F.one_hot(torch.from_numpy(act.astype(np.int64)).clamp(0, NA - 1), NA).float()
    idx = lambda m: np.flatnonzero(m & ok)                                  # noqa: E731
    tr, tu, te = idx(tr_m), idx(np.isin(slot, list(tune))), idx(np.isin(slot, list(test)))

    # Rare-event class weights, sqrt-inverse-frequency. HP change is ~0.1% of steps, and unweighted the head
    # simply never predicts it (measured: hp loss-recall 0.000, accuracy 0.9987 = the base rate). This is the
    # same treatment that took HP damage F1 from 0.286 to 0.906; a RAW inverse gave the rare class weight
    # 0.0005 and it was never predicted either, hence sqrt.
    def cw(lab, k, m=None):
        v = lab.reshape(-1)[(m.reshape(-1) > 0.5) if m is not None else slice(None)]
        c = np.bincount(np.asarray(v).astype(np.int64), minlength=k).astype(np.float64)
        w = np.sqrt(c.sum() / np.maximum(c, 1.0))
        return torch.from_numpy((w / w.mean()).astype(np.float32))

    CW_HP = cw(hp_ev[tr], 3)
    CW_RHP = cw(r_hp_ev[tr], 3, pres_now[tr])
    # dcell is ~94% "same cell"; without a weight the head predicts only the centre class and never a move.
    CW_DC = cw(dcell_i[tr], DC) if a.dcell_weight else None
    print(f"hp event rates train: {np.bincount(hp_ev[tr], minlength=3) / len(tr)} -> weights {CW_HP.numpy().round(2)}")

    if a.save_prep:
        Path(a.save_prep).parent.mkdir(parents=True, exist_ok=True)
        # position needs no delta scale: it is grid cell + offset now. Only y is still a scaled continuous delta.
        json.dump({"scales": sc, "grid": grid.state_dict(), "d_link_y": sdy, "split": split},
                  open(a.save_prep, "w"), indent=1)
        print(f"-> wrote prep bundle {a.save_prep} (ship this to every rank as --prep)")

    link = None
    if a.fleet_experiment and a.fleet_world_size > 1 and a.fleet_sync == "avg":
        from zelda_rl.fleet_link import FleetLink
        link = FleetLink(a.fleet_experiment, a.fleet_rank, a.fleet_world_size, a.fleet_launch_id)
        if not a.prep:
            raise SystemExit("--prep is REQUIRED with --fleet-sync avg: averaging weights across ranks whose "
                             "feature scales / nav grid were fitted on different shards is silently wrong")
        print(f"fleet: {a.fleet_experiment} rank {a.fleet_rank}/{a.fleet_world_size} "
              f"aggregator {a.fleet_aggregator}")

    net = TokenWM(NA, a.dim, a.heads)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"{len(state):,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | link tok {LINK_DIM} "
          f"actor tok {ACTOR_DIM} | d {a.dim} heads {a.heads} | params {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    nb_ = len(SF1_KEEP)

    def split_link(o, i):
        j = 0
        dcl, j = o[:, j:j + DC], j + DC
        dof, j = o[:, j:j + 2], j + 2
        dp = (dcl, dof)
        dyv, j = o[:, j:j + 1], j + 1
        dyw, j = o[:, j:j + 2], j + 2
        hp, j = o[:, j:j + 3], j + 3
        bits, j = o[:, j:j + nb_], j + nb_
        sc_, j = o[:, j:j + 2], j + 2
        # LOGIT RESIDUAL so argmax/sign at init equals the CURRENT value -> exact persistence
        dcl = dcl + LOGIT_K * F.one_hot(torch.full((len(i),), CENTRE), DC).float()   # persistence = same cell
        dp = (dcl, dof)
        bits = bits + LOGIT_K * (2 * Y["bits_now"][i] - 1)
        sc_ = sc_ + LOGIT_K * F.one_hot(Y["scene_now"][i], 2).float()
        hp = hp + LOGIT_K * F.one_hot(torch.zeros(len(i), dtype=torch.long), 3).float()
        return dp, dyv, dyw, hp, bits, sc_

    def split_actor(o, i):
        j = 0
        rdcl, j = o[..., j:j + DC], j + DC
        rdof, j = o[..., j:j + 2], j + 2
        hp, j = o[..., j:j + 3], j + 3
        pr, j = o[..., j:j + 1], j + 1
        st = o[..., j:j + N_STATES]
        hp = hp + LOGIT_K * F.one_hot(torch.zeros(o.shape[:2], dtype=torch.long), 3).float()
        pr = pr + LOGIT_K * (2 * Y["pres_now"][i][..., None] - 1)
        st = st + LOGIT_K * F.one_hot(Y["st_now"][i], N_STATES).float()
        rdcl = rdcl + LOGIT_K * F.one_hot(torch.full(o.shape[:2], CENTRE), DC).float()
        return (rdcl, rdof), hp, pr, st

    def loss_fn(i):
        ol, oa = net({k: v[i] for k, v in tt.items()}, A[i])
        dp, dyv, dyw, hp, bits, scn = split_link(ol, i)
        m = Y["pres_now"][i]
        rdp, rhp, rpr, rst = split_actor(oa, i)
        L = {"l_dcell": F.cross_entropy(dp[0], Y["dcell"][i], weight=CW_DC),   # WHICH cell: classification
             "l_doff": F.l1_loss(dp[1], Y["doff"][i]),                          # sub-cell: bounded L1
             "l_y": F.l1_loss(dyv, Y["dy"][i]),
             "l_yaw": F.l1_loss(dyw, Y["dyaw"][i]),
             "l_hp": F.cross_entropy(hp, Y["hp_ev"][i], weight=CW_HP),     # CE, class-weighted
             "l_bits": F.binary_cross_entropy_with_logits(bits, Y["bits_nxt"][i]),
             "l_scene": F.cross_entropy(scn, Y["scene_nxt"][i]),
             "a_dcell": (F.cross_entropy(rdp[0].reshape(-1, DC), Y["r_dcell"][i].reshape(-1),
                                         reduction="none").reshape(m.shape) * m).sum() / m.sum().clamp(min=1),
             "a_doff": ((rdp[1] - Y["r_doff"][i]).abs().mean(-1) * m).sum() / m.sum().clamp(min=1),
             "a_hp": (F.cross_entropy(rhp.reshape(-1, 3), Y["r_hp_ev"][i].reshape(-1), weight=CW_RHP,
                                      reduction="none").reshape(m.shape) * m).sum() / m.sum().clamp(min=1),
             "a_pres": F.binary_cross_entropy_with_logits(rpr.squeeze(-1), Y["pres_nxt"][i]),
             "a_state": (F.cross_entropy(rst.reshape(-1, N_STATES), Y["st_nxt"][i].reshape(-1),
                                         reduction="none").reshape(m.shape) * m).sum() / m.sum().clamp(min=1)}
        return sum(L.values()), L, (dp, hp)

    hist, best = [], {"werr": float("inf")}
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(tr)
        t0, tot, gn, gmax, nb = time.time(), 0.0, 0.0, 0.0, 0
        for s in range(0, len(perm) - a.batch, a.batch):
            i = torch.from_numpy(perm[s:s + a.batch])
            loss, parts, _ = loss_fn(i)
            opt.zero_grad(); loss.backward()
            g = float(nn.utils.clip_grad_norm_(net.parameters(), a.clip))
            opt.step()
            tot += float(loss.detach()); gn += g; gmax = max(gmax, g); nb += 1
            if not np.isfinite(g):
                print(f"  !! non-finite gradient at step {nb}"); break
        net.eval()
        with torch.no_grad():
            iu = torch.from_numpy(tu[:8192])
            _, parts, (dp, hp) = loss_fn(iu)
            k = iu.numpy()
            pc = dp[0].argmax(-1).numpy()
            pcell = np.stack([pc // DW, pc % DW], 1) - DW // 2 + T["link_cell"][k]
            pred_xz = grid.from_cell(pcell, (T["link_off"][k] + dp[1].numpy()).clip(-0.5, 0.5))
            true_xz = lpos[nxt][k]
            werr = float(np.linalg.norm(pred_xz - true_xz, axis=1).mean())
            pers = float(np.linalg.norm(lpos[k] - true_xz, axis=1).mean())
            dcell_acc = float((dp[0].argmax(-1) == Y["dcell"][iu]).float().mean())
            hp_acc = float((hp.argmax(-1) == Y["hp_ev"][iu]).float().mean())
            hp_rec = float((hp.argmax(-1)[Y["hp_ev"][iu] == 1] == 1).float().mean()) if (Y["hp_ev"][iu] == 1).any() else float("nan")
        row = {"epoch": ep + 1, "train": tot / nb, "grad_mean": gn / nb, "grad_max": gmax,
               "dpos_world": werr, "persistence_world": pers, "skill": 1 - werr / max(pers, 1e-9),
               "hp_acc": hp_acc, "hp_loss_recall": hp_rec, "dcell_acc": dcell_acc,
               **{k: float(v) for k, v in parts.items()}}
        hist.append(row)
        if link is not None:
            # FedAvg once per epoch: each rank has trained on its own shard, so this is data parallelism over
            # the whole fleet's dataset. Straggler tolerant -- a slow rank never holds the others back.
            info = link.average_weights(net, ep + 1, aggregator=a.fleet_aggregator)
            link.heartbeat({"epoch": ep + 1, "dpos_world": werr, "skill": row["skill"],
                            "dcell_acc": dcell_acc})
            print(f"    fleet: averaged over {info.get('n', '?')} ranks (from update {info.get('src', '?')})")
        # BEST-EPOCH SELECTION on held-out position error. The loss falls smoothly while dpos/dcell oscillate
        # between a good state (dcell 0.97, skill ~-0.1) and a bad one (0.76, -10), so the LAST epoch is a coin
        # flip -- and reading the last epoch is how several "degrades with training" conclusions got made this
        # session. Keep the best weights, and report best alongside last.
        if werr < best["werr"]:
            best = {**row, "werr": werr, "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
        print(f"  ep {ep+1:>3}/{a.epochs} loss {tot/nb:.4f} | grad mean {gn/nb:.3f} max {gmax:.3f} | "
              f"dpos {werr:.3f} vs persistence {pers:.3f} world units (skill {1-werr/max(pers,1e-9):+.3f}) | "
              f"dcell {dcell_acc:.4f} | hp rec {hp_rec:.3f}  ({time.time()-t0:.0f}s)")
    print(f"\nBEST epoch {best['epoch']}: dpos {best['werr']:.3f} vs persistence {best['persistence_world']:.3f} "
          f"(skill {best['skill']:+.3f}) dcell {best['dcell_acc']:.4f} | LAST epoch {hist[-1]['epoch']}: "
          f"dpos {hist[-1]['dpos_world']:.3f} (skill {hist[-1]['skill']:+.3f}) dcell {hist[-1]['dcell_acc']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best.pop("state", net.state_dict()), "last_model": net.state_dict(),
                "best": {k: v for k, v in best.items() if k != "state"}, "scales": sc, "split": split, "n_actions": NA,
                "config": vars(a), "history": hist}, a.out)
    print(f"\nper-head final losses: " + json.dumps({k: round(hist[-1][k], 5) for k in
          ("l_dcell", "l_doff", "l_y", "l_yaw", "l_hp", "l_bits", "l_scene", "a_dcell", "a_doff", "a_hp",
           "a_pres", "a_state")}))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
