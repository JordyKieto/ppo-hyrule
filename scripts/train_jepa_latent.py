#!/usr/bin/env python3
"""JEPA stage 2 done properly: roll forward IN LATENT SPACE and decode the reward fields out of it.

    tokens_t --E--> z_t --T(.,a_t)--> z_{t+1} --T(.,a_{t+1})--> ... --D--> reward fields at each step

The predicted latent is NEVER decoded and re-encoded. `scripts/train_wm_tokens.py` did exactly that -- it
predicted next-STATE deltas and fed its own decoded state back in as tokens -- and the consequence was that
every token field it consumed but did not predict (19 of Link's 33, 9 of an actor's 10, all 9 bgCheckFlags
bits) sat frozen at its step-0 value, so H>1 rollout numbers measured harness staleness as much as the model.
In a latent rollout that problem cannot exist: velocity, camera yaw and attack timers ride along inside z
without anything having to predict them. That is the entire point of the approach.

WHY THIS IS NOT THE SIX FAILURES IN LATENT_WORLD_MODEL.md 1-30. Those all scored next-latent distance, where
the identity is the GLOBAL OPTIMUM -- the latent moves ~0.04/step, so copying z is near-perfect and a zero-init
residual predictor IS the identity (measured: deep+LN -0.776, shallow+tanh -2.023, transformer -2.412, VQ
+0.000 = pure copying, FiLM -9.87). The objective here is the DECODED REWARD FIELDS after h steps, which the
identity cannot fake: a frozen z decodes to a Link who never moves and Babas that never attack, which is
exactly the persistence control we score against. Latent distance appears only as a small auxiliary term
against an EMA target encoder, whose job is to stop the transition drifting somewhere the decoder cannot read.

Design rules carried over because they were measured, not assumed:
  * ABSOLUTE decode, not deltas. The decoder emits the nav-grid cell (two 32-way softmaxes) plus a within-cell
    offset per entity, so a rollout needs no integration and no re-encoding. Positions never appear as raw
    coordinates anywhere (`zelda_rl/tokens.py`).
  * PERSISTENCE AT INIT. The transition is a zero-init residual, so at step 0 z_{t+h} == z_t for every h and
    the decode is constant over the horizon -- i.e. exactly the "nothing changes" control. Skill starts at 0.
  * HP and behaviour state are CLASSIFIED, never regressed (damage F1 0.286 -> 0.906).
  * The action reaches the transition on the QUERY side only, and the encoder's inputs are already verified
    disjoint from `bc.CONTROL_ECHO_FEATS`, so nothing can read a_{t-1} off the observation.
  * Scored as SKILL against persistence on decoded reward fields at H = 1, 2, 5, 10, 20 -- the thing the dream
    actually consumes -- never on loss (AGENTS.md 8; a run once halved its val loss while its rollout skill
    fell from +0.0026 to -0.41).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_wm_tokens import Attn  # noqa: E402  the masked, distance-biased attention block
from zelda_rl import wm_data  # noqa: E402
from zelda_rl.navgrid import NavGrid  # noqa: E402
from zelda_rl.tokens import ACTOR_DIM, LINK_DIM, REL_CLIP, SF1_KEEP  # noqa: E402

N_STATES = 10
NB = len(SF1_KEEP)
CH_LINK, CH_BABA, CH_ALIVE, N_GRID_CH = 0, 1, 2, 3
# displacement-from-start in nav-grid cells, clipped to +-DK. Link moves ~1-13 world units a
# step and a cell is 28-57, so +-12 covers a 20-step rollout with room to spare.
DK, LOGIT_K = 12, 6.0
DKN = 2 * DK + 1
# HP change from the rollout start, in quarter-hearts, clipped. Link has 48 quarter-hearts and a
# Baba 2 HP, so +-8 covers any realistic 20-step window.
DHP = 8
DHPN = 2 * DHP + 1
# offsets inside the packed token vectors (computed from LINK_FIELDS / ACTOR_FIELDS ordering)
L_RELY, L_HP, L_HPF, L_YAW = 0, 1, 2, slice(3, 5)
L_BITS = slice(16, 16 + NB)
A_HP = 3


class Encoder(nn.Module):
    """tokens -> one latent per entity, (B, 1+R, d). Same trunk as the token WM's encoder side."""

    def __init__(self, d=384, heads=6, n_rows=9, id_vocab=256):
        super().__init__()
        self.n_rows = n_rows
        self.id_emb, self.cat_emb, self.st_emb = nn.Embedding(id_vocab, 8), nn.Embedding(16, 4), nn.Embedding(N_STATES, 4)
        self.cell_emb = nn.Embedding(64, 12)
        self.rcell_emb = nn.Embedding(2 * REL_CLIP + 1, 12)
        self.link_enc = nn.Sequential(nn.Linear(LINK_DIM + 24 + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.actor_enc = nn.Sequential(nn.Linear(ACTOR_DIM + 16 + 24 + 2, d), nn.ReLU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.attn = Attn(d, heads)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, T):
        B = T["link"].shape[0]
        e = torch.cat([self.id_emb(T["ids"]), self.cat_emb(T["cats"]), self.st_emb(T["states"])], -1)
        lt = self.link_enc(torch.cat([T["link"], self.cell_emb(T["link_cell"]).flatten(-2),
                                      T["link_off"]], -1))[:, None] + self.type_emb[0]
        at = self.actor_enc(torch.cat([T["actors"], e, self.rcell_emb(T["actor_rcell"]).flatten(-2),
                                       T["actor_off"]], -1)) + self.type_emb[1]
        x = torch.cat([lt, at], 1)
        m = torch.cat([torch.ones(B, 1, device=x.device), T["mask"]], 1)      # Link is always valid
        pos = torch.cat([torch.zeros(B, 1, 2, device=x.device), T["cell_xy"]], 1)
        D = (pos[:, :, None, :] - pos[:, None, :, :]).norm(dim=-1)
        x = self.ln1(x + self.attn(x, m, D))
        return self.ln2(x + self.ff(x))


class Transition(nn.Module):
    """z, a -> z'. Zero-init residual, so at step 0 this is EXACTLY the identity and the decode is persistence."""

    def __init__(self, d=384, heads=6, n_actions=63):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(n_actions, d), nn.ReLU())
        self.attn = Attn(d, heads)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, z, a_onehot):
        """PURE latent dynamics: a function of z and a ONLY.

        It deliberately takes no mask and no distance matrix. Those are properties of the OBSERVED state at
        step 0, and threading them through a rollout would pin them for the whole horizon -- which is exactly
        the frozen-field failure this whole approach exists to avoid. Presence and geometry are already inside
        z, because the encoder put them there; an absent actor's latent is a constant the model can learn to
        ignore. Nothing decoded ever re-enters here.
        """
        B, Nt, _ = z.shape
        ones = torch.ones(B, Nt, device=z.device)
        zeroD = torch.zeros(B, Nt, Nt, device=z.device)
        h = self.ln1(z + self.attn(z, ones, zeroD))
        a = self.act(a_onehot)[:, None].expand(-1, Nt, -1)              # action on the QUERY side only
        return z + self.ff(torch.cat([h, a], -1))                       # zero-init -> identity at step 0


class Decoder(nn.Module):
    """z -> ABSOLUTE reward fields. Cells are classified (two 32-way heads), HP and state classified."""

    def __init__(self, d=384, grid_n=32, n_hp_l=49, n_hp_a=5):
        super().__init__()
        self.grid_n, self.n_hp_l, self.n_hp_a = grid_n, n_hp_l, n_hp_a
        # HP is CLASSIFIED, not regressed. The packed value is scaled, but meta.json carries the scale, so the
        # integer HP is recoverable per batch. Regressing it washes out damage, which is a rare small change:
        # classification took damage F1 from 0.286 to 0.906 (LATENT_WORLD_MODEL.md 3).
        # POSITION IS DECODED AS DISPLACEMENT FROM THE ROLLOUT START, not as an absolute cell.
        # Decoding absolute position makes the experiment unmeasurable: the persistence control is the TRUE
        # position held constant, and one cell of classification error is 28-57 world units against a
        # persistence error of ~6 at H=1, so the decoder would have to nail absolute position before any
        # dynamics skill became visible (measured: H1 skill -2.94 with an absolute head).
        # Displacement also makes the identity trap collapse honestly: predicting zero displacement IS
        # persistence, i.e. skill exactly 0.000, so the identity can score 0 and never better.
        # Integration happens OUTSIDE the latent -- start cell plus decoded displacement -- so nothing decoded
        # is ever fed back into z.
        # Every field is decoded RELATIVE TO THE OBSERVED START STATE, so persistence is the exact init:
        # positions as a displacement, HP as a bounded delta class, and the categorical fields via a logit
        # residual toward their step-0 value. The reference is the OBSERVED state at t, never a decoded one, so
        # this is integration outside the latent -- the same thing as adding a displacement to a start cell --
        # and not a feedback path into z.
        # The decoder reads BOTH ENDPOINTS, (z_h, z_0). A displacement is a function of two states, and z_h
        # alone does not identify the start -- at init z_h == z_0 exactly, so nothing distinguishes h=1 from
        # h=10 and one shared decoder can only learn an AVERAGE displacement. Measured with z_h alone: position
        # error 17.46 at H=1 against a persistence baseline of 3.81, and 65.84 at H=10 against 25.92, i.e. the
        # same absolute over-prediction at every horizon. z_0 is a LATENT, not a decoded value, so passing it
        # is not a feedback path into z.
        self.link = nn.Linear(2 * d, 2 * DKN + 2 + 1 + DHPN + 2 + NB + 2)
        self.actor = nn.Linear(2 * d, 2 * DKN + 2 + DHPN + 1 + N_STATES)
        for head in (self.link, self.actor):        # zero-init: with the centre logit below, step 0 = no motion
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        # TYPED NAV-GRID OCCUPANCY, decoded from the pooled latent: one channel per entity type over the
        # grid_n x grid_n cells (Link / Baba present / Baba alive). The wall channel is static, so it is never
        # predicted. This is spatial grounding -- it forces the latent to carry WHERE things are in a form that
        # is read as a map rather than as 2 numbers -- and like every other head it is READ-ONLY: its output is
        # never fed back into z. The rollout is z -> T -> z, and decoders hang off the side.
        self.grid = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(),
                                  nn.Linear(2 * d, N_GRID_CH * grid_n * grid_n))

    def forward(self, z, z0, S):
        """z = latent at step h, z0 = latent at the rollout start, S = the OBSERVED start state's discrete
        values (logit residuals only). Nothing decoded ever enters here."""
        g, o = self.grid_n, {}
        zc = torch.cat([z, z0], -1)
        ctr = torch.zeros(DKN, device=z.device); ctr[DK] = LOGIT_K      # argmax at init = zero displacement
        hctr = torch.zeros(DHPN, device=z.device); hctr[DHP] = LOGIT_K  # argmax at init = no HP change
        l = self.link(zc[:, 0]); j = 0
        o["l_cx"], j = l[:, j:j + DKN] + ctr, j + DKN
        o["l_cy"], j = l[:, j:j + DKN] + ctr, j + DKN
        o["l_off"], j = l[:, j:j + 2], j + 2                            # cumulative, so NOT bounded to +-0.5
        o["l_rely"], j = l[:, j:j + 1], j + 1
        o["l_dhp"], j = l[:, j:j + DHPN] + hctr, j + DHPN
        o["l_yaw"], j = l[:, j:j + 2], j + 2
        o["l_bits"], j = l[:, j:j + NB] + LOGIT_K * (2 * S["bits0"] - 1), j + NB
        o["l_scene"] = l[:, j:j + 2] + LOGIT_K * F.one_hot(S["scene0"], 2).float()
        a = self.actor(zc[:, 1:]); j = 0
        o["a_cx"], j = a[..., j:j + DKN] + ctr, j + DKN
        o["a_cy"], j = a[..., j:j + DKN] + ctr, j + DKN
        o["a_off"], j = a[..., j:j + 2], j + 2
        o["a_dhp"], j = a[..., j:j + DHPN] + hctr, j + DHPN
        o["a_pres"], j = a[..., j:j + 1] + LOGIT_K * (2 * S["pres0"][..., None] - 1), j + 1
        o["a_state"] = a[..., j:j + N_STATES] + LOGIT_K * F.one_hot(S["state0"], N_STATES).float()
        # pooled over entities: the map is a property of the whole scene, not of one token
        o["grid"] = self.grid(zc.mean(1)).view(-1, N_GRID_CH, self.grid_n, self.grid_n)
        return o


def occupancy_target(link_cell, actor_cell, pres, is_baba, hp_alive, grid_n):
    """The typed nav-grid the `grid` head is scored against, built from TRUE positions. Deterministic, so it
    needs no extra packed field -- `zelda_rl/navgrid.py` is the same codec the inputs use."""
    B, R = pres.shape
    occ = torch.zeros(B, N_GRID_CH, grid_n * grid_n, device=pres.device)
    lf = (link_cell[:, 0] * grid_n + link_cell[:, 1]).clamp(0, grid_n * grid_n - 1)
    occ[:, CH_LINK].scatter_(1, lf[:, None], 1.0)
    af = (actor_cell[..., 0] * grid_n + actor_cell[..., 1]).clamp(0, grid_n * grid_n - 1)
    occ[:, CH_BABA].scatter_add_(1, af, (pres * is_baba).float())
    occ[:, CH_ALIVE].scatter_add_(1, af, (pres * is_baba * hp_alive).float())
    return occ.clamp(max=1.0).view(B, N_GRID_CH, grid_n, grid_n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", required=True, help="directory from scripts/pack_dataset.py")
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=1000,
                    help="A latent rollout costs ~(2H+2) network passes per sample, so using every frame as a "
                         "rollout start makes an epoch ~17 min. Consecutive starts overlap almost entirely "
                         "(they share H-1 of their H steps), so subsampling them is nearly free statistically "
                         "and an 'epoch' becomes a fixed step budget that is comparable across runs.")
    ap.add_argument("--horizon", type=int, default=5, help="latent rollout steps trained on, per sample")
    ap.add_argument("--ema", type=float, default=0.996, help="target-encoder decay")
    ap.add_argument("--w-recon", type=float, default=1.0, help="decode loss at step 0 (grounding)")
    ap.add_argument("--w-roll", type=float, default=1.0, help="decode loss at steps 1..H -- the PRIMARY term")
    ap.add_argument("--w-bc", type=float, default=0.1, help="behaviour-cloning pressure on the encoder")
    ap.add_argument("--w-lat", type=float, default=0.05, help="aux latent distance to the EMA target")
    ap.add_argument("--w-grid", type=float, default=0.5, help="typed nav-grid occupancy decode")
    ap.add_argument("--w-pos", type=float, default=4.0,
                    help="multiplier on LINK position (cell + offset). The decode loss has ~12 terms over 9 "
                         "actors plus a 3072-output occupancy map, so without this the one field the reward "
                         "most depends on is a small fraction of the objective and gets traded away.")
    ap.add_argument("--eval-h", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--eval-n", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=20.0,
                    help="1.0 was calibrated for the single-step token WM; this model backprops "
                         "through H decode passes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=2)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="checkpoints/jepa/latent.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    amp = bool(a.amp and dev.type == "cuda")

    pm = json.load(open(Path(a.packed) / "meta.json"))
    N, NA = int(pm["n_frames"]), pm["n_actions"]
    if a.rows != pm["rows"]:
        sys.exit(f"--rows {a.rows} but packed with rows={pm['rows']}")
    grid = NavGrid.load(pm["grid"])
    sc = pm["scales"]
    # Only the fields the ENCODER reads and the DECODER is scored on. The token WM's delta targets
    # (dcell/doff/dy/...) are not loaded at all -- this model decodes ABSOLUTE state -- which saves ~11 GiB.
    NEED = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
            "actor_rcell", "actor_off", "cell_xy", "act", "scene_now")
    P = {k: np.load(Path(a.packed) / f"{k}.npy", mmap_mode="r")[:N] for k in NEED + ("ok", "lpos")}
    D = {}
    for k in NEED:
        t = torch.from_numpy(np.ascontiguousarray(P[k])).to(dev)
        D[k] = t
    D["link_off"] = D["link_off"].half()
    ok = np.asarray(P["ok"])
    lpos = np.asarray(P["lpos"])
    nb = sum(v.element_size() * v.nelement() for v in D.values())
    print(f"packed {a.packed}: {N:,} frames | resident {nb / 2**30:.1f} GiB on {dev} | grid {grid.n}x{grid.n}")

    # HP class counts recovered from the fitted scales: the packed value is hp/scale clipped to [-1,1].
    n_hp_l = int(round(sc["l_hp"])) + 1
    n_hp_a = int(round(sc["a_hp"])) + 1
    print(f"hp classes: link {n_hp_l}, actor {n_hp_a} (from scales l_hp={sc['l_hp']:.2f} a_hp={sc['a_hp']:.2f})")

    # VALID START INDICES: a rollout of H steps must stay inside one episode. `ok` is False exactly on the last
    # frame of an episode, so the next False at or after j gives the run length from j.
    ends = np.flatnonzero(~ok)
    runlen = ends[np.searchsorted(ends, np.arange(N))] - np.arange(N)
    test_slots, tune_slots = set(pm["split"]["val_slots"]), set(pm["split"].get("tune_slots", []))
    slot = np.asarray(np.load(Path(a.packed) / "slot.npy", mmap_mode="r")[:N])
    tr = np.flatnonzero((runlen >= a.horizon) & ~np.isin(slot, list(test_slots | tune_slots)))
    # ONE EVAL SUBSET PER HORIZON, not the intersection. Taking starts with runlen >= max(eval_h) makes the
    # H=1 number depend on which OTHER horizons were requested: adding H=20 restricted every horizon to frames
    # 20+ steps from an episode end, a slower-moving subset where persistence at H10 is 14.30 instead of 25.92,
    # so the same model scores lower on every horizon. Per-horizon subsets keep each number comparable across
    # runs, and H=1 is then measured on essentially the whole tune split.
    tu = {h: np.flatnonzero((runlen >= h) & np.isin(slot, list(tune_slots))) for h in a.eval_h}
    print(f"train starts {len(tr):,} (H>={a.horizon}) | tune starts per horizon: "
          + ", ".join(f"H{h} {len(v):,}" for h, v in tu.items()))

    from zelda_rl.bc import BABA_ID
    enc = Encoder(a.dim, a.heads, a.rows).to(dev)
    tgt_enc = copy.deepcopy(enc).to(dev)                 # EMA TARGET: never gets gradients
    for p_ in tgt_enc.parameters():
        p_.requires_grad_(False)
    trans = Transition(a.dim, a.heads, NA).to(dev)
    dec = Decoder(a.dim, grid.n, n_hp_l, n_hp_a).to(dev)
    bc_head = nn.Linear(a.dim, NA).to(dev)
    params = list(enc.parameters()) + list(trans.parameters()) + list(dec.parameters()) + list(bc_head.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    print(f"params: enc {sum(p_.numel() for p_ in enc.parameters())/1e6:.2f}M  "
          f"trans {sum(p_.numel() for p_ in trans.parameters())/1e6:.2f}M  "
          f"dec {sum(p_.numel() for p_ in dec.parameters())/1e6:.2f}M")

    LONG = ("ids", "cats", "states", "link_cell", "actor_rcell")
    TOK = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
           "actor_rcell", "actor_off", "cell_xy")

    def tokens(j):
        o = {k: (D[k][j].long() if k in LONG else D[k][j].float()) for k in TOK}
        return o

    def acell(j, lc=None):
        """Absolute actor nav-grid cell: actor_rcell is stored RELATIVE to Link's cell."""
        lc = D["link_cell"][j].long() if lc is None else lc
        return (lc[:, None, :] + D["actor_rcell"][j].long() - REL_CLIP).clamp(0, grid.n - 1)

    def start_state(i):
        """The OBSERVED discrete state at the rollout start. Used as logit residuals and to integrate deltas.
        Never a decoded value -- this is the real state at t, which the rollout legitimately knows."""
        lk = D["link"][i].float()
        return {"bits0": lk[:, L_BITS], "scene0": D["scene_now"][i].long(),
                "pres0": D["mask"][i].float(), "state0": D["states"][i].long(),
                "hp0": (lk[:, L_HP] * sc["l_hp"]).round().clamp(0, n_hp_l - 1).long(),
                "ahp0": (D["actors"][i].float()[:, :, A_HP] * sc["a_hp"]).round().clamp(0, n_hp_a - 1).long()}

    def targets(i, j):
        """Targets for decoding at frame `j` given the rollout STARTED at `i`.

        Position is a DISPLACEMENT i -> j; everything else is the absolute value at j. At j == i the
        displacement is zero, which is what makes the zero-init decoder start exactly at persistence.
        """
        lk, ac = D["link"][j].float(), D["actors"][j].float()
        lc, lc0 = D["link_cell"][j].long(), D["link_cell"][i].long()
        acj, ac0 = acell(j, lc), acell(i, lc0)
        # an actor only has a meaningful displacement if it is present at BOTH ends
        m = (D["mask"][j].float() * D["mask"][i].float())
        S0 = start_state(i)
        hp_l = (lk[:, L_HP] * sc["l_hp"]).round().clamp(0, n_hp_l - 1).long()
        hp_a = (ac[:, :, A_HP] * sc["a_hp"]).round().clamp(0, n_hp_a - 1).long()
        dhp_l = (hp_l - S0["hp0"]).clamp(-DHP, DHP) + DHP
        dhp_a = (hp_a - S0["ahp0"]).clamp(-DHP, DHP) + DHP
        dl = (lc - lc0).clamp(-DK, DK) + DK
        da = (acj - ac0).clamp(-DK, DK) + DK
        # the same displacement as a CONTINUOUS value in cell units, for the world-distance term
        l_disp = (lc - lc0).float() + (D["link_off"][j].float() - D["link_off"][i].float())
        a_disp = (acj - ac0).float() + (D["actor_off"][j].float() - D["actor_off"][i].float())
        return {"l_cx": dl[:, 0], "l_cy": dl[:, 1], "l_cell": lc,
                "l_disp": l_disp, "a_disp": a_disp,
                "l_off": D["link_off"][j].float() - D["link_off"][i].float(),
                "a_cell_d": da, "a_cell": acj,
                "a_off": D["actor_off"][j].float() - D["actor_off"][i].float(), "l_off": D["link_off"][j].float(),
                "l_rely": lk[:, L_RELY:L_RELY + 1], "l_yaw": lk[:, L_YAW],
                "l_bits": lk[:, L_BITS], "l_scene": D["scene_now"][j].long(),
                "l_hp": hp_l, "l_dhp": dhp_l, "a_hp": hp_a, "a_dhp": dhp_a,
                "a_pres": D["mask"][j].float(), "a_state": D["states"][j].long(), "mask": m,
                "is_baba": (D["ids"][j].long() == BABA_ID).float(), "hp_alive": (hp_a > 0).float()}

    # cell size in world units per axis, so the position loss can be expressed in the SAME units the metric is.
    # Relative axis weighting is kept (a cell is 28.8 x 57.1 world units, so an error along z costs more) but
    # normalised to a mean of 1, so the position term is O(1) like every other term. In raw world units it was
    # ~40x larger than the rest and produced grad norms of 1400-1600 against clip 10 -- 160x clipping, i.e.
    # direction-only steps, which makes the learning rate meaningless.
    _cell = np.asarray(grid.cell, np.float32)
    CELL = torch.tensor(_cell / _cell.mean(), device=dev)
    WORLD_PER_CELL = float(_cell.mean())
    KOFF = torch.arange(-DK, DK + 1, device=dev, dtype=torch.float32)

    def soft_disp(cx, cy, off):
        """Expected displacement in CELL units: softmax over the cell classes plus the continuous offset.
        Differentiable, so the world-distance loss below can actually train the cell head."""
        ex = (cx.softmax(-1) * KOFF).sum(-1)
        ey = (cy.softmax(-1) * KOFF).sum(-1)
        return torch.stack([ex, ey], -1) + off

    def decode_loss(o, T):
        m, ms = T["mask"], T["mask"].sum().clamp(min=1)
        L = {
            # WORLD-UNIT DISTANCE, which is the metric. Training CE on the cell class plus L1 on the offset
            # instead optimises a redundant decomposition: a wrong-but-confident cell is cheap in CE and costs a
            # whole cell (28-57 world units), while a neighbouring cell with a compensating offset is expensive
            # in CE and costs almost nothing. Measured with CE+L1 only: position error climbed from persistence
            # (5.55) to 17.09 while the loss fell, at every horizon. CE is kept at low weight to sharpen the
            # head; the world term is what is actually being asked for.
            "l_world": (((soft_disp(o["l_cx"], o["l_cy"], o["l_off"])
                          - (T["l_disp"])) * CELL).abs().mean()) * a.w_pos,
            "l_cell": (F.cross_entropy(o["l_cx"], T["l_cx"]) + F.cross_entropy(o["l_cy"], T["l_cy"])) * 0.1,
            "l_rely": F.l1_loss(o["l_rely"], T["l_rely"]),
            "l_hp": F.cross_entropy(o["l_dhp"], T["l_dhp"]),
            "l_yaw": F.l1_loss(o["l_yaw"], T["l_yaw"]),
            "l_bits": F.binary_cross_entropy_with_logits(o["l_bits"], T["l_bits"]),
            "l_scene": F.cross_entropy(o["l_scene"], T["l_scene"]),
            "a_cell": (((F.cross_entropy(o["a_cx"].reshape(-1, DKN), T["a_cell_d"][..., 0].reshape(-1),
                                         reduction="none")
                         + F.cross_entropy(o["a_cy"].reshape(-1, DKN), T["a_cell_d"][..., 1].reshape(-1),
                                           reduction="none")).reshape(m.shape) * m).sum() / ms) * 0.1,
            "a_world": ((((soft_disp(o["a_cx"], o["a_cy"], o["a_off"]) - T["a_disp"]) * CELL).abs().mean(-1)
                         * m).sum() / ms) * a.w_pos,
            "a_hp": (F.cross_entropy(o["a_dhp"].reshape(-1, DHPN), T["a_dhp"].reshape(-1),
                                     reduction="none").reshape(m.shape) * m).sum() / ms,
            "a_pres": F.binary_cross_entropy_with_logits(o["a_pres"].squeeze(-1), T["a_pres"]),
            "a_state": (F.cross_entropy(o["a_state"].reshape(-1, N_STATES), T["a_state"].reshape(-1),
                                        reduction="none").reshape(m.shape) * m).sum() / ms,
        }
        occ = occupancy_target(T["l_cell"], T["a_cell"], T["mask"], T["is_baba"], T["hp_alive"], grid.n)
        L["grid"] = F.binary_cross_entropy_with_logits(o["grid"], occ) * a.w_grid
        return L

    def onehot(j):
        return F.one_hot(D["act"][j].long(), NA).float()

    def step_losses(i_np):
        """One training sample: encode at t, then roll the LATENT forward --horizon steps, decoding the reward
        fields at every step. No decoded value is ever re-encoded."""
        i = torch.from_numpy(i_np).to(dev)
        z = z0 = enc(tokens(i))
        S0 = start_state(i)
        L = {f"0/{k}": v for k, v in decode_loss(dec(z, z0, S0), targets(i, i)).items()}
        for k in list(L):
            L[k] = L[k] * a.w_recon
        # BC pressure on the encoder: predict the action FROM THE STATE. The encoder's inputs are verified
        # disjoint from CONTROL_ECHO_FEATS, so this cannot be read off a_{t-1}.
        L["bc"] = F.cross_entropy(bc_head(z.mean(1)), D["act"][i].long()) * a.w_bc
        for h in range(1, a.horizon + 1):
            z = trans(z, onehot(i + h - 1))
            j = i + h
            for k, v in decode_loss(dec(z, z0, S0), targets(i, j)).items():
                L[f"{h}/{k}"] = v * a.w_roll / a.horizon
            if a.w_lat > 0:
                with torch.no_grad():
                    zt = tgt_enc(tokens(j))
                L[f"{h}/lat"] = F.mse_loss(z, zt) * a.w_lat / a.horizon
        return sum(L.values()), L

    @torch.no_grad()
    def evaluate(tu_by_h, horizons, n):
        """Roll each horizon on ITS OWN valid subset, so a number never depends on which other horizons were
        asked for."""
        out = {}
        for h in horizons:
            pool = tu_by_h[h]
            out.update(_eval_one(pool[:n], h))
        return out

    @torch.no_grad()
    def _eval_one(sel_np, H):
        """SKILL against the persistence control, on the DECODED reward fields -- what the dream consumes.

        Persistence is the true state at t held constant, the same control every other result here is scored
        against. Position error is in world units, decoded through the nav-grid codec.
        """
        enc.eval(); trans.eval(); dec.eval()
        out = {}
        i = torch.from_numpy(sel_np).to(dev)
        z = z0 = enc(tokens(i))
        S0 = start_state(i)
        true0 = lpos[sel_np]
        for h in range(1, H + 1):
            z = trans(z, onehot(i + h - 1))
            if h != H:
                continue
            o = dec(z, z0, S0)
            T = targets(i, i + h)
            # integrate OUTSIDE the latent: start cell/offset plus the decoded displacement. from_cell is
            # linear in (idx + off), so an offset outside [-0.5, 0.5] needs no renormalising.
            # decode the SAME quantity that is trained: expected displacement in cell units, integrated onto
            # the observed start cell OUTSIDE the latent.
            disp = soft_disp(o["l_cx"], o["l_cy"], o["l_off"]).float()
            cell = (D["link_cell"][i].float() + disp).cpu().numpy()      # from_cell is linear in (idx + off)
            pred = grid.from_cell(cell, D["link_off"][i].float().cpu().numpy())
            truth = lpos[sel_np + h]
            err = float(np.linalg.norm(pred - truth, axis=1).mean())
            per = float(np.linalg.norm(true0 - truth, axis=1).mean())
            out[f"h{h}/pos_err"] = err
            out[f"h{h}/pos_pers"] = per
            out[f"h{h}/pos_skill"] = 1 - err / max(per, 1e-9)
            # HP / presence / state: accuracy against the persistence answer "unchanged since t"
            hp_pred = (S0["hp0"] + o["l_dhp"].argmax(-1) - DHP).clamp(0, n_hp_l - 1)
            out[f"h{h}/hp_acc"] = float((hp_pred == T["l_hp"]).float().mean())
            out[f"h{h}/hp_pers"] = float((S0["hp0"] == T["l_hp"]).float().mean())
            m = T["mask"]
            out[f"h{h}/state_acc"] = float(((o["a_state"].argmax(-1) == T["a_state"]).float() * m).sum()
                                           / m.sum().clamp(min=1))
            out[f"h{h}/pres_acc"] = float(((o["a_pres"].squeeze(-1) > 0).float() == T["a_pres"]).float().mean())
            occ = occupancy_target(T["l_cell"], T["a_cell"], m, T["is_baba"], T["hp_alive"], grid.n)
            out[f"h{h}/grid_iou"] = float(((o["grid"] > 0) & (occ > 0.5)).sum()
                                          / (((o["grid"] > 0) | (occ > 0.5)).sum().clamp(min=1)))
            # LATENT DRIFT DIAGNOSTIC. The transition is a residual applied h times, so a small systematic
            # update compounds; rolled to 20 steps having been trained on 5, position error hit 80.13 against a
            # persistence baseline of 23.91. These two numbers say whether that is the latent EXPLODING (norm
            # grows) or merely going somewhere the decoder cannot read (norm fine, distance from the real
            # encoding large). Measure before fixing -- AGENTS.md 22.
            out[f"h{h}/z_norm"] = float(z.norm(dim=-1).mean())
            out[f"h{h}/z0_norm"] = float(z0.norm(dim=-1).mean())
            out[f"h{h}/z_drift"] = float((z - tgt_enc(tokens(i + h))).norm(dim=-1).mean())
        enc.train(); trans.train(); dec.train()
        return out

    wb = None
    if a.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, name=a.run_name, config=vars(a), reinit=True)
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled: {e}")
    cdir = Path(a.ckpt_dir) if a.ckpt_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)

    hist, best = [], {"skill": -9e9}
    for ep in range(a.epochs):
        want = a.steps_per_epoch * a.batch
        perm = (rng.choice(tr, want, replace=False) if 0 < want < len(tr) else rng.permutation(tr))
        t0, tot, nb_, gn = time.time(), 0.0, 0, 0.0
        agg = {}
        for s0 in range(0, len(perm) - a.batch, a.batch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, parts = step_losses(perm[s0:s0 + a.batch])
            opt.zero_grad(); loss.backward()
            g = float(nn.utils.clip_grad_norm_(params, a.clip))
            opt.step()
            with torch.no_grad():                                  # EMA TARGET ENCODER
                for pt, po in zip(tgt_enc.parameters(), enc.parameters()):
                    pt.mul_(a.ema).add_(po.detach(), alpha=1 - a.ema)
                for bt, bo in zip(tgt_enc.buffers(), enc.buffers()):
                    bt.copy_(bo)
            tot += float(loss.detach()); gn += g; nb_ += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            if not np.isfinite(g):
                print(f"  !! non-finite gradient at step {nb_}"); break
        ev = evaluate(tu, a.eval_h, a.eval_n)
        hh = a.eval_h
        row = {"epoch": ep + 1, "loss": tot / max(nb_, 1), "grad": gn / max(nb_, 1),
               **{k: v / max(nb_, 1) for k, v in agg.items()}, **ev}
        hist.append(row)
        print(f"  ep {ep+1:>3}/{a.epochs} loss {row['loss']:.4f} grad {row['grad']:.1f}  "
              + "  ".join(f"H{h} skill {ev[f'h{h}/pos_skill']:+.4f}" for h in hh) + f"  ({time.time()-t0:.0f}s)")
        print("      pos err/pers: " + "  ".join(
            f"H{h} {ev[f'h{h}/pos_err']:.2f}/{ev[f'h{h}/pos_pers']:.2f}" for h in hh))
        print("      latent: " + "  ".join(
            f"H{h} |z| {ev[f'h{h}/z_norm']:.2f} drift {ev[f'h{h}/z_drift']:.2f}" for h in hh)
            + f"   |z0| {ev[f'h{hh[0]}/z0_norm']:.2f}")
        print("      hp acc/pers:  " + "  ".join(
            f"H{h} {ev[f'h{h}/hp_acc']:.3f}/{ev[f'h{h}/hp_pers']:.3f}" for h in hh)
            + "   state " + " ".join(f"H{h} {ev[f'h{h}/state_acc']:.3f}" for h in hh)
            + "   gridIoU " + " ".join(f"H{h} {ev[f'h{h}/grid_iou']:.3f}" for h in hh))
        key = ev[f"h{hh[-1]}/pos_skill"]
        if key > best["skill"]:
            best = {"skill": key, "epoch": ep + 1,
                    "state": {n: {k: v.detach().clone() for k, v in m.state_dict().items()}
                              for n, m in (("enc", enc), ("trans", trans), ("dec", dec))}}
        if wb is not None:
            wb.log(row)
        if cdir:
            with open(cdir / "metrics.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if (ep + 1) % a.ckpt_every == 0:
                torch.save({"enc": enc.state_dict(), "trans": trans.state_dict(), "dec": dec.state_dict(),
                            "tgt_enc": tgt_enc.state_dict(), "bc": bc_head.state_dict(),
                            "opt": opt.state_dict(), "epoch": ep + 1, "metrics": row, "hist": hist,
                            "meta": pm, "config": vars(a)}, cdir / f"ck_{ep+1:04d}.pt")
    print(f"\nBEST H{a.eval_h[-1]} position skill {best['skill']:+.4f} at epoch {best.get('epoch')}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"best": {k: v for k, v in best.items() if k != "state"}, "state": best.get("state"),
                "hist": hist, "meta": pm, "config": vars(a)}, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
