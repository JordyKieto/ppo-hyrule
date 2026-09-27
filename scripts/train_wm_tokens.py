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
import glob
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
from scripts.train_bc_decode import DEFAULT_DATA, load_many  # noqa: E402
from zelda_rl.bc import LI, RI, _split  # noqa: E402
from zelda_rl import wm_data
from zelda_rl.navgrid import NavGrid  # noqa: E402
from zelda_rl.combat import BG_BITS
from zelda_rl.tokens import ACTOR_DIM, LINK_DIM, REL_CLIP, SF1_KEEP, build  # noqa: E402

N_STATES, LOGIT_K = 10, 6.0
# ---- FULL-CLOSURE head layout -----------------------------------------------------------------------------
# The token WM predicts EVERY field it consumes. Anything it reads and cannot advance would be frozen during an
# autoregressive rollout, and at H=1800 a frozen input is 180 seconds stale -- the rollout would be partly
# fiction. Column offsets below follow LINK_FIELDS / ACTOR_FIELDS order exactly.
L_RELY, L_HP, L_HPF, L_YAW, L_SPEED, L_VEL, L_CAM, L_HEAD, L_FLOOR, L_NPRES, L_NBABA = \
    0, 1, 2, slice(3, 5), 5, slice(6, 9), slice(9, 11), slice(11, 13), 13, 14, 15
A_RELY, A_BEAR, A_HP, A_YAW, A_SHAPE, A_Y2P, A_TIMER, A_VEL = \
    0, slice(1, 3), 3, slice(4, 6), slice(6, 8), slice(8, 10), 10, slice(11, 14)
# HP is classified, never regressed (damage F1 0.286 -> 0.906). A bounded DELTA class gives exact magnitude
# instead of the old 3-way same/down/up, which a rollout could not turn into an HP value.
HPD_STEP, HPD_K = 4.0, 8          # Link: deltas of +-32 raw units in steps of 4 -> 17 classes
HPD_N = 2 * HPD_K + 1
AHPD_K = 2                        # a Baba has 2 HP, so +-2 covers it -> 5 classes
AHPD_N = 2 * AHPD_K + 1
# Delta-cell window: 5x5 cells, so a step may move at most 2 cells on either axis (at n=32 that is ~58 x 89
# world units, far beyond the 13-unit moving step). Index 12 is (0,0) = SAME CELL, the persistence class.
# CHANGE GATES. Only the REGRESSION heads get one: the discrete heads already default to "no change" through
# the LOGIT_K persistence residual, and a second gate would be two defaults fighting. Movement is excluded --
# l_doff changes on 60.1% of steps, which is MOBA's own named exclusion, and l_cam / l_vel / a_rely are all
# above 28%. These eight are the sparse ones (a_vel 0.03%, a_yaw 0.04%, a_shape 1.82%, a_doff 3.55%,
# a_timer 4.44%, l_floor 10.4%, l_yaw 19.8%, l_y 22.0%); see docs/WM_MOTION.md for the full table.
# MEASURED SET, corrected. The threshold is a change rate around 4%, not MOBA's coarser "everything except
# movement": gating a field that changes 14-33% of steps makes it WORSE, because the gate misfires often and
# the value head -- trained only on changed frames -- emits a large delta on each misfire. Measured on the
# identical per-field eval, no-gate -> gate: a_yaw -6.05 -> +0.00, a_shape -1.69 -> +0.00, a_timer -0.40 ->
# +0.01, a_doff -0.22 -> -0.02 (all <=3.7% change), against l_y -0.09 -> -1.01 (23%), l_floor -0.02 -> -0.96
# (14.5%) and a_vel +0.17 -> +0.14 (33%, already fine without one).
GATE_L = ("l_yaw",)                                          # 19.9%: marginal (-0.011 -> +0.004), kept
GATE_A = ("a_doff", "a_yaw", "a_shape", "a_timer")           # 3.7 / 0.02 / 1.7 / 3.7 %
GATE_W = 0.2          # BCE weight: MOBA down-weights it x0.2 because the yes/no loss crowds out the value head
DW, DC = 5, 25
CENTRE = (DW // 2) * DW + DW // 2      # LOGIT_K: how hard the persistence prior is asserted on discrete heads


def entmax(z, dim=-1, alpha=1.5, n_iter=24):
    """alpha-entmax: softmax's sparse cousin. Low-scoring keys get EXACTLY zero, not 1e-6.

    softmax(z)_i = exp(z_i)/sum(exp) is strictly positive for every key, so every entity in the pool sits in
    the denominator and carries a gradient path to the query -- the "O(N^2) entity noise" problem. entmax
    solves argmax_p (p.z - H_alpha(p)) over the simplex; for alpha > 1 the solution has a hard threshold, so
    the model LEARNS how many entities to attend to per query. alpha=1 is softmax, alpha=2 is sparsemax,
    alpha=1.5 is the usual compromise.

    Solved by bisection on the threshold tau: p_i = [(alpha-1) z_i - tau]_+^(1/(alpha-1)). At tau = max(y) the
    sum is 0 and at tau = max(y) - 1 the largest term alone is >= 1, so the root is bracketed. Differentiable
    through the clamp and the power, which is the standard bisecting implementation.

    Masked keys must arrive already pushed to a large negative value; they then fall below tau and receive
    exactly zero, and the never-masked NO-OP token guarantees the support is non-empty so there is no 0/0.
    """
    y = z * (alpha - 1)
    ymax = y.max(dim, keepdim=True).values
    lo, hi = ymax - 1.0, ymax
    ex = 1.0 / (alpha - 1)
    for _ in range(n_iter):
        tau = (lo + hi) / 2
        s = (y - tau).clamp(min=0).pow(ex).sum(dim, keepdim=True)
        big = s > 1
        lo = torch.where(big, tau, lo)
        hi = torch.where(big, hi, tau)
    p = (y - (lo + hi) / 2).clamp(min=0).pow(ex)
    return p / p.sum(dim, keepdim=True).clamp(min=1e-12)


class Attn(nn.Module):
    """Queries are entities; keys/values are entities plus an always-valid NO-OP token.

    Per-head learnable distance bias; -1e4 masking; invalid keys hard-zeroed post-softmax.
    """

    def __init__(self, d: int, heads: int = 4, attn: str = "softmax"):
        super().__init__()
        self.attn = attn
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.o = nn.Linear(d, d)
        self.w = nn.Parameter(torch.zeros(heads))          # softplus(0)=0.693 -> mild locality at init
        self.b = nn.Parameter(torch.zeros(heads))
        self.noop = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x, mask, D, ctx=None, ctx_mask=None, ctx_D=None):
        """ASYMMETRIC attention: `x` are the QUERIES (and also keys/values); `ctx` is a keys/values-ONLY pool.

        Static environment tokens belong in `ctx`. Wall cells outnumber actors roughly 12:1, so if they were
        queries too the attention would go from 10x11 to 130x131 pairs, we would compute 120 query outputs that
        no head reads, and every one of those rows would contribute gradient paths with no causal bearing on the
        dynamics. A wall has no dynamics to predict; it is context. This is the bipartite/cross-attention
        constraint AlphaStar and Wayformer both impose, and the reason their entity transformers stay stable at
        entity counts where a dense block does not.

          x   (B, N, d)    queries: Link + actors
          ctx (B, M, d)    keys/values only: static environment tokens
          D   (B, N, N)    pairwise distance query->x, in scaled units
          ctx_D (B, N, M)  distance query->ctx
        """
        B, N, d = x.shape
        parts = [x] if ctx is None else [x, ctx]
        kv = torch.cat(parts + [self.noop.expand(B, 1, d)], 1)             # NO-OP: row can never be empty
        km = [mask] if ctx is None else [mask, ctx_mask]
        kmask = torch.cat(km + [torch.ones(B, 1, device=x.device)], 1)
        M = kv.shape[1]
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(B, M, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(B, M, self.h, self.dk).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / self.dk ** 0.5
        Dp = [D] if ctx is None else [D, ctx_D]
        Dp = torch.cat(Dp + [torch.zeros(B, N, 1, device=x.device)], -1)   # no-op sits at distance 0
        s = s - F.softplus(self.w)[None, :, None, None] * Dp[:, None] + self.b[None, :, None, None]
        s = s.masked_fill(kmask[:, None, None, :] < 0.5, -1e4)             # -1e4, not -1e9
        a = (entmax(s, -1) if self.attn == "entmax" else s.softmax(-1))
        a = a * kmask[:, None, None, :]                                    # hard zero padded keys post-softmax
        return self.o((a @ v).transpose(1, 2).reshape(B, N, d))


class TokenWM(nn.Module):
    """Entity tokens + asymmetric attention, with THREE entity classes: Link, actors, and static environment.

    The environment class is keys/values only. At a 7x7 wall patch there are 49 environment tokens against at
    most 10 entity tokens, so they are 83% of the pool; letting them query would take attention from 10x11 to
    59x60 pairs and compute 49 query rows no head reads. A wall has no dynamics to predict -- it is context.
    """

    def __init__(self, n_actions: int, d: int = 128, heads: int = 4, n_rows: int = 16, id_vocab: int = 256,
                 attn: str = "softmax", layers: int = 3, hist: int = 8, env_p: int = 49, hp_bins: int = 16,
                 grid_n: int = 64, gate: bool = False):
        super().__init__()
        self.n_rows, self.hist, self.env_p, self.hp_bins = n_rows, hist, env_p, hp_bins
        self.id_emb, self.cat_emb, self.st_emb = nn.Embedding(id_vocab, 8), nn.Embedding(16, 4), nn.Embedding(N_STATES, 4)
        # Position as EMBEDDINGS, not floats: a cell index has no magnitude to leak into a query-key dot product.
        # Sized from the ACTUAL grid. This was hardcoded to 64, which silently made every --grid above 64
        # crash with "index out of range in self" -- so the one motion change with a measured mechanism behind
        # it (docs/WM_MOTION.md step 4) could not be run at all.
        self.cell_emb = nn.Embedding(max(64, grid_n), 12)          # Link's absolute cell, per axis
        self.rcell_emb = nn.Embedding(2 * REL_CLIP + 1, 12)       # actor cell RELATIVE to Link, per axis
        # HP as a BINNED EMBEDDING, not a scaled float. Our own measured result -- damage F1 0.286 -> 0.906 --
        # says classify HP, and the output head is already a bounded delta class; the input side was the
        # inconsistent one.
        self.lhp_emb, self.ahp_emb = nn.Embedding(hp_bins, 8), nn.Embedding(hp_bins, 4)
        # LOCAL HISTORY SUMMARISER (Wayformer polyline factorisation): K frames per entity are compressed by a
        # local encoder into ONE token before the global attention runs -- never K separate global tokens.
        # K=8 because the phase variables this task turns on are only observable as rates: the OOCDM ablation
        # measured a Baba's height (a_rel_y, how far it has emerged) as the dominant predictor of its behaviour
        # state at ~90x baseline loss, and acceleration -- what a button press physically controls -- does not
        # exist in a single frame.
        L_IN = LINK_DIM + 24 + 2 + 8
        A_IN = ACTOR_DIM + 16 + 24 + 2 + 4 + 4          # +4 = dual local frames (Link-facing and camera)
        self.link_enc = nn.Sequential(nn.Linear(L_IN * hist, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.actor_enc = nn.Sequential(nn.Linear(A_IN * hist, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        # static environment tokens: WHICH cell of the patch, and whether it is a wall. Offset, never a coordinate.
        self.env_off_emb = nn.Embedding(env_p, d // 2)
        self.env_occ_emb = nn.Embedding(2, d // 2)
        self.env_enc = nn.Linear(d, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))             # 0 Link, 1 actor, 2 ENVIRONMENT
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "attn": Attn(d, heads, attn),
            "ln1": nn.LayerNorm(d), "ln2": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)),
        }) for _ in range(layers)])
        self.act_emb = nn.Sequential(nn.Linear(n_actions, d), nn.ReLU())    # action: QUERY-SIDE ONLY
        self.n_gl, self.n_ga = (len(GATE_L), len(GATE_A)) if gate else (0, 0)
        self.link_head = nn.Linear(2 * d, self.n_gl + DC + 2 + 1 + 2 + HPD_N + len(SF1_KEEP) + 2
                                   + 3 + 2 + 1 + len(BG_BITS))
        self.actor_head = nn.Linear(2 * d, self.n_ga + DC + 2 + AHPD_N + 1 + N_STATES + 1 + 2 + 2 + 1 + 3)
        for h in (self.link_head, self.actor_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)                                          # PERSISTENCE at step 0

    def forward(self, T, a_onehot):
        """T fields carry a leading history axis (B, K, ...) for `link`/`actors` and friends; the newest frame
        is last. Env inputs are `env_occ` (B, P) and `env_d` (B, N, P)."""
        link, actors = T["link"], T["actors"]
        B = link.shape[0]
        K = link.shape[1] if link.dim() == 3 else 1
        if link.dim() == 2:                                   # K=1 convenience: add the history axis
            link = link[:, None]; actors = actors[:, None]
        def hx(k):
            v = T[k]
            return v if (v.dim() >= 3 and v.shape[1] == K) else v[:, None].expand(-1, K, *([-1] * (v.dim() - 1)))
        ids, cats, sts = hx("ids"), hx("cats"), hx("states")
        e = torch.cat([self.id_emb(ids), self.cat_emb(cats), self.st_emb(sts)], -1)
        lce = self.cell_emb(hx("link_cell")).flatten(-2)
        rce = self.rcell_emb(hx("actor_rcell")).flatten(-2)
        lhp = self.lhp_emb(T["lhp_bin"] if T["lhp_bin"].dim() == 2 else T["lhp_bin"][:, None])
        ahp = self.ahp_emb(T["ahp_bin"])
        lt_in = torch.cat([link, lce, hx("link_off"), lhp], -1).reshape(B, -1)
        at_in = torch.cat([actors, e, rce, hx("actor_off"), ahp, hx("frames")], -1)
        at_in = at_in.transpose(1, 2).reshape(B, self.n_rows, -1)            # (B, R, K*A_IN)
        lt = self.link_enc(lt_in)[:, None] + self.type_emb[0]
        at = self.actor_enc(at_in) + self.type_emb[1]
        x = torch.cat([lt, at], 1)
        mask = T["mask"][:, -1] if T["mask"].dim() == 3 else T["mask"]
        m = torch.cat([torch.ones(B, 1, device=link.device), mask], 1)
        cxy = T["cell_xy"][:, -1] if T["cell_xy"].dim() == 4 else T["cell_xy"]
        pos = torch.cat([torch.zeros(B, 1, 2, device=link.device), cxy], 1)
        D = (pos[:, :, None, :] - pos[:, None, :, :]).norm(dim=-1)
        # static environment: keys/values ONLY, and a third class so "I am environment" is learned, not inferred
        ctx = ctx_m = ctx_D = None
        if "env_occ" in T:
            occ = T["env_occ"]
            oid = torch.arange(occ.shape[1], device=occ.device)[None].expand(B, -1)
            ctx = self.env_enc(torch.cat([self.env_off_emb(oid), self.env_occ_emb(occ.long())], -1)) \
                + self.type_emb[2]
            ctx_m = torch.ones(B, occ.shape[1], device=occ.device)
            ctx_D = T["env_d"]
        for blk in self.blocks:
            x = blk["ln1"](x + blk["attn"](x, m, D, ctx, ctx_m, ctx_D))
            x = blk["ln2"](x + blk["ff"](x))
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
    ap.add_argument("--gate", action="store_true",
                    help="per-head CHANGE GATE on the sparse REGRESSION heads (GATE_L / GATE_A). The value "
                         "head is supervised only where the field changed and, in a rollout, is multiplied by "
                         "gate>0.5 so an unchanged field advances by exactly zero instead of by noise. "
                         "Measured +0.080 position skill on the MLP probe; docs/WM_MOTION.md.")
    ap.add_argument("--grid", type=int, default=32, help="nav-grid resolution. MEASURED: at 32 the fitted cell is 142x66 world units against a 5.5-unit mean step, so 87%% of MOVING steps never leave their cell and the dcell head is nearly constant (docs/WM_MOTION.md)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--rows", type=int, default=16,
                    help="actor token slots to KEEP. The state layout has 16, but measured over the v5 "
                         "dataset slots 9-15 are occupied in 0.000%% of frames (max 9 actors present, and "
                         "slots are stable across time so they cannot be compacted). 9 costs nothing and cuts "
                         "every per-actor array by 44%%, which is what lets the whole dataset sit in VRAM.")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (cuda only)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hist", type=int, default=8,
                    help="frames of history summarised into each entity token by a LOCAL encoder before the "
                         "global attention (Wayformer polyline factorisation). 1 = the old Markovian behaviour.")
    ap.add_argument("--layers", type=int, default=3, help="stacked attention blocks (was hard-coded to 1)")
    ap.add_argument("--env-radius", type=int, default=3,
                    help="half-width of the static wall patch fed as KEYS/VALUES-ONLY environment tokens; "
                         "radius 3 gives a 7x7 = 49-token patch")
    ap.add_argument("--hp-bins", type=int, default=16)
    ap.add_argument("--attn", choices=["softmax", "entmax"], default="softmax",
                    help="entmax (alpha=1.5) gives low-scoring keys EXACTLY zero weight, so the "
                         "model learns how many entities to attend to instead of spreading a "
                         "little mass over all of them")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dcell-weight", action="store_true",
                    help="class-weight the dcell head. OFF by default: sqrt-inverse weights over 25 classes "
                         "where 94%% are 'same cell' give the centre class ~0.01 against ~2.9 for the rare "
                         "ones, so the loss PUNISHES predicting no-movement -- correct 94%% of the time. "
                         "Measured: dcell accuracy 0.966 at epoch 1 collapsing to 0.52 by epoch 12. Rare-event "
                         "weighting is right for HP detection and wrong for position accuracy.")
    ap.add_argument("--packed", default=None,
                    help="directory written by scripts/pack_dataset.py. Skips load_many and "
                         "tokenisation entirely: mmaps the packed arrays and moves them to the "
                         "device. The full v5 load is ~25 min and OOMed a 188 GB pod; this is ~1 min.")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap the load, sampling episodes uniformly at random across ALL --data/--extra-data "
                         "shards. This machine has 16 GB and a frame costs ~3 KB of state+tokens, so the full "
                         "66M-transition v5 dataset does not fit locally -- it needs the fleet (FedAvg over "
                         "the per-pod shards) or a streaming loader. Measure before raising it.")
    ap.add_argument("--shards", default=None,
                    help="glob of shard dirs to use as --data/--extra-data, e.g. 'datasets/transitions/v5/rank*'")
    ap.add_argument("--lr-sweep", default=None,
                    help="comma-separated LRs. Loads the data ONCE, then trains a fresh model per LR for "
                         "--epochs and prints a comparison table. No checkpoints, no wandb.")
    ap.add_argument("--sched", choices=["none", "cosine"], default="none",
                    help="cosine decays lr to 0 over --epochs after --warmup epochs of linear ramp")
    ap.add_argument("--warmup", type=float, default=2.0, help="warmup epochs (used by --sched cosine)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=6)
    ap.add_argument("--ckpt-dir", default=None, help="save ck_XXXX.pt every --ckpt-every epochs, plus best/last")
    ap.add_argument("--ckpt-every", type=int, default=1)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from (keeps going to --epochs)")
    ap.add_argument("--resume-fresh-opt", action="store_true",
                    help="allow --resume from a checkpoint with no optimizer state; the run is then "
                         "discontinuous and the epochs after it are not comparable to the ones before")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="cap the batches per epoch (0 = a full pass over the training starts). With the depth "
                         "mixture a full pass is ~12,700 batches at mean depth 17.4, i.e. ~220k forward passes "
                         "and over an hour per row of metrics -- far too coarse to watch. A fixed step budget "
                         "makes epochs comparable across runs as well.")
    ap.add_argument("--h-mix", default=None,
                    help='MIXED-DEPTH autoregressive training, "H:weight,H:weight,...". Not a curriculum -- a '
                         'fixed mixture sampled per batch, so the model sees every depth throughout. Default '
                         'mix is 60%% one-step and 0.15%% full-match (1800 steps = 180s at 10 steps/s), with '
                         'geometric buckets between; mean depth 17.4, and note 60%% of BATCHES are one-step but '
                         'they are only 3%% of the COMPUTE while the 0.15%% at H=1800 is 15%% of it.')
    ap.add_argument("--h-half-life", type=float, default=600.0,
                    help="STEP DECAY: step h's loss is weighted 0.5^((h-1)/half_life), renormalised over the "
                         "steps. Parameterised as a half-life rather than a raw gamma because gamma does not "
                         "survive this range -- at 0.95, gamma^1800 is 1e-40 and every deep step you paid for "
                         "would get exactly zero weight. At 600: step 1 is 1.0, step 600 is 0.5, step 1800 "
                         "is 0.125.")
    ap.add_argument("--bptt", type=int, default=32,
                    help="detach the token state every N rollout steps. Backprop through 1800 chained forwards "
                         "is a memory impossibility; truncating bounds it while the model still SEES the full "
                         "depth.")
    ap.add_argument("--train-h", type=int, default=1,
                    help="AUTOREGRESSIVE training depth. 1 = teacher-forced one step (the old behaviour). >1 "
                         "feeds the model its own predicted state forward, which is the distribution the dream "
                         "actually runs in. Cost is linear in this.")
    ap.add_argument("--rollout-every", type=int, default=2,
                    help="epochs between the AUTOREGRESSIVE validation (expensive, so low frequency)")
    ap.add_argument("--rollout-h", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="horizons for the autoregressive rollout, scored against the persistence control")
    # --- fleet: data-parallel FedAvg over per-pod shards ------------------------------------------------
    # Each pod trains on ITS OWN collected shard and weights are averaged, so the fleet sees the whole dataset
    # with no data movement. FleetLink.average_weights is generic over any model with a state_dict.
    # accepted and ignored: fleet's launch path passes these to every script it starts. Without them argparse
    # aborts before the first batch, on all 39 pods at once.
    ap.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--slots-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--video-backend", default=None, help=argparse.SUPPRESS)
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

    # The prep bundle carries the split, so a fleet pod needs no reference checkpoint (checkpoints/ is
    # excluded from the code push, and loading --ref there fails with FileNotFoundError on every rank).
    _prep0 = json.load(open(a.prep)) if a.prep else None
    # A packed dataset carries its own split in meta.json, so it must not need --ref either: requiring a
    # checkpoint that checkpoints/ is excluded from the code push would fail for no reason.
    split = (_prep0["split"] if _prep0 and "split" in _prep0
             else json.load(open(Path(a.packed) / "meta.json"))["split"] if a.packed
             else torch.load(a.ref, weights_only=False, map_location="cpu")["split"])
    dirs = [a.data, *a.extra_data]
    if a.shards:
        # sorted() so the shard list -- and therefore the episode sampling under a fixed --seed -- is
        # reproducible; a bare glob is filesystem-order.
        # glob, not Path().glob: the latter cannot take an ABSOLUTE pattern, which is what a pod path is.
        dirs = sorted(d for d in glob.glob(a.shards) if (Path(d) / "manifest.jsonl").exists())
        if not dirs:
            sys.exit(f"--shards {a.shards!r} matched no directory containing a manifest.jsonl")
        if not (Path(dirs[0]) / "dataset.json").exists():
            sys.exit(f"{dirs[0]} has no dataset.json; load_many needs it on the FIRST dir for obs_dim")
        # A shard pulled from S3 gets its manifest before all of its episodes, so a glob can pick up a
        # HALF-SYNCED shard. load_many would then raise on a missing .npz -- but only after minutes of
        # loading, and only if the shuffle happened to reach it. Check it up front instead.
        # A manifest row pointing at a file that is NOT on disk is a half-synced shard: load_many would
        # raise, but only minutes into the load and only if the shuffle reached it, so fail up front.
        # Files on disk that are NOT in the manifest are the opposite problem -- unused data, not a crash --
        # so that is a warning. After scripts/rebuild_manifests.py the only ones left are corrupt .npz.
        missing, extra = [], 0
        for d in dirs:
            man = {json.loads(l)["file"].split("/")[-1]
                   for l in (Path(d) / "manifest.jsonl").read_text().splitlines() if l.strip()}
            disk = {q.name for q in (Path(d) / "episodes").glob("*.npz")}
            if man - disk:
                missing.append(f"{d}: {len(man - disk)} manifest rows have no file on disk")
            extra += len(disk - man)
        if missing:
            sys.exit("incomplete shard(s) -- finish the pull (scripts/pull_dataset.sh resumes) or narrow "
                     "--shards:\n  " + "\n  ".join(missing[:10])
                     + (f"\n  ... and {len(missing)-10} more" if len(missing) > 10 else ""))
        if extra:
            print(f"note: {extra} episode file(s) on disk are not in any manifest and will NOT be used "
                  f"(run scripts/rebuild_manifests.py; whatever it cannot add is a corrupt .npz)")
        print(f"shards: {len(dirs)} dirs matching {a.shards} (all complete)")
    # ---- BOTH PATHS CONVERGE ON `P`: the dict of packed numpy arrays defined by zelda_rl.wm_data. The token
    # and target maths lives THERE, not here, so the in-memory path and scripts/pack_dataset.py cannot drift
    # apart and make a sweep number incomparable to a run number.
    if a.packed:
        pm = json.load(open(Path(a.packed) / "meta.json"))
        N = int(pm["n_frames"])
        if a.rows != pm["rows"]:
            sys.exit(f"--rows {a.rows} but {a.packed} was packed with rows={pm['rows']}")
        NA, sdy, sc, split = pm["n_actions"], pm["d_link_y"], pm["scales"], pm["split"]
        grid = NavGrid.load(pm["grid"])
        P = {k: np.load(Path(a.packed) / f"{k}.npy", mmap_mode="r")[:N]
             for k in wm_data.field_shapes(pm["rows"], pm["link_dim"], pm["actor_dim"])}
        print(f"packed dataset {a.packed}: {N:,} frames | {pm['bytes_per_frame']} B/frame | "
              f"grid {grid.n}x{grid.n} | rows {pm['rows']}")
    else:
        # REFUSE BEFORE SPENDING 20 MINUTES. The un-packed load holds the f32 state array and 16-row f32
        # tokens at once, and np.concatenate copies them, so peak is ~6 KB/frame. A 100-epoch run was
        # OOM-killed after 20 minutes of loading because `free` reports the HOST's 1.5 TB inside a 188 GB
        # container; check_load_fits reads the cgroup instead (AGENTS.md 21 -- a rule that is not a check
        # will be broken).
        if not a.max_frames:
            est = 0
            for d in dirs:
                for l in (Path(d) / "manifest.jsonl").read_text().splitlines():
                    if l.strip():
                        est += int(json.loads(l)["steps"]) + 1
            fits, why = wm_data.check_load_fits(est, note=f" for {est:,} frames")
            print(f"load estimate: {why}")
            if not fits:
                sys.exit(f"this load will not fit -- {why}.\n"
                         f"Pack it once instead, then every run starts in ~2.5 minutes:\n"
                         f"  python scripts/pack_dataset.py {dirs[0]} --extra {' '.join(dirs[1:3])} ... "
                         f"--out <dir> --rows {a.rows}\n"
                         f"  python scripts/train_wm_tokens.py --packed <dir> --rows {a.rows} ...\n"
                         f"(or pass --max-frames to load a subset, which skips this check)")
        # want_obs=False: this model never touches the 483-d obs, and it is 57% of the memory.
        meta, state, act, slot, epid, obs, has = load_many(dirs, want_obs=False,
                                                          max_frames=a.max_frames, seed=a.seed)
        NA, N = meta["n_actions"], len(state)
        _, _rows, _ = _split(state)
        occupied = int((_rows[:, :, RI["present"]] > 0.5).any(0).nonzero()[0].max()) + 1
        if occupied > a.rows:
            sys.exit(f"--rows {a.rows} would DROP actors: slot {occupied - 1} is occupied somewhere in this "
                     f"data. Actor slots are stable across time, so they cannot be compacted -- raise --rows.")
        print(f"actor slots: {occupied} ever occupied of {_rows.shape[1]} -> keeping {a.rows}")
        _trm = ~np.isin(slot, list(set(split["val_slots"]) | set(split.get("tune_slots", []))))
        if _prep0:
            sc, grid, sdy = _prep0["scales"], NavGrid.load(_prep0["grid"]), _prep0["d_link_y"]
        else:
            sc, grid, sdy = wm_data.fit_prep(state, epid, _trm, a.rows, a.grid)
        print(f"nav grid {grid.n}x{grid.n} | cell {np.round(grid.cell,1)} world units")
        P = wm_data.tokens_and_targets(state, act, slot, epid, grid=grid, scales=sc, d_link_y=sdy,
                                       rows_keep=a.rows, n_actions=NA)
        del state, _rows

    test, tune = set(split["val_slots"]), set(split.get("tune_slots", []))
    ok = np.asarray(P["ok"])
    slot = np.asarray(P["slot"])
    lpos = np.asarray(P["lpos"])
    nxt = np.minimum(np.arange(N) + 1, N - 1)
    tr_m = ~np.isin(slot, list(test | tune))
    # host-side copies for the grid decode in eval / rollout (cheap: 6 B/frame)
    T = {"link_cell": np.asarray(P["link_cell"]).astype(np.int64),
         "link_off": np.asarray(P["link_off"]).astype(np.float32)}
    dcell_i, hp_ev = np.asarray(P["dcell"]), np.asarray(P["hp_ev"])
    r_hp_ev, pres_now = np.asarray(P["r_hp_ev"]), np.asarray(P["mask"])
    print(f"dcell: SAME CELL on {float((dcell_i[tr_m & ok] == CENTRE).mean()):.1%} of steps")

    dev = torch.device(a.device)
    # Resident on the device in the packed dtypes, cast per batch. `pres_now`/`st_now` are bit-identical to
    # `mask`/`states` so they are ALIASED rather than stored twice, and `rel_xz` is sliced out of `actors`.
    TOK = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
           "actor_rcell", "actor_off", "cell_xy")
    TGT = ("dcell", "doff", "dy", "dyaw", "hp_ev", "bits_now", "bits_nxt", "scene_now", "scene_nxt",
           "r_dcell", "r_doff", "r_hp_ev", "pres_nxt", "st_nxt")
    tt = {k: torch.from_numpy(np.ascontiguousarray(P[k])).to(dev) for k in TOK}
    tt["link_off"] = tt["link_off"].half()        # f32 on the host for the decode, f16 on the device
    Y = {k: torch.from_numpy(np.ascontiguousarray(P[k])).to(dev) for k in TGT}
    Y["pres_now"], Y["st_now"] = tt["mask"], tt["states"]
    A = torch.from_numpy(np.ascontiguousarray(P["act"])).to(dev)
    _nbytes = sum(v.element_size() * v.nelement() for v in list(tt.values()) + list(Y.values())) + A.nelement()
    print(f"resident dataset: {_nbytes / 2**30:.1f} GiB on {dev} ({_nbytes / max(N, 1):.0f} B/frame)")

    _LONG = ("ids", "cats", "states", "link_cell", "actor_rcell")     # embedding lookups need an index dtype
    _YLONG = ("dcell", "hp_ev", "scene_now", "scene_nxt", "r_dcell", "r_hp_ev", "st_now", "st_nxt")

    # ---- static environment + history scaffolding -------------------------------------------------------
    from zelda_rl.wm_context import local_frames, patch_offsets, wall_patch, wall_map_from_packed, window_index
    _bgoff = BG_BITS.index(3)                                    # BG_WALL == bit 3, index within the bg block
    WMAP = torch.from_numpy(wall_map_from_packed(
        np.asarray(P["link_cell"]), np.asarray(P["link"][:, 16 + len(SF1_KEEP):], dtype=np.float32), _bgoff, grid.n)).to(dev)
    print(f"static wall map: {int(WMAP.sum())} of {grid.n * grid.n} cells ever registered a wall contact")
    ENV_OFF = patch_offsets(a.env_radius, dev)                   # (P, 2), nearest-first
    print(f"environment tokens: {ENV_OFF.shape[0]} per frame (radius {a.env_radius}), KEYS/VALUES ONLY")
    # episode START index per frame, for clamping the history window
    _starts = np.zeros(N, np.int64)
    _b = np.r_[0, np.flatnonzero(~ok) + 1]
    for _lo, _hi in zip(_b[:-1], np.r_[_b[1:], N]):
        _starts[_lo:_hi] = _lo
    RSTART = torch.from_numpy(_starts).to(dev)
    CELLWH = torch.tensor(np.asarray(grid.cell, np.float32), device=dev)
    _hpmax = max(sc.get("l_hp", 1.0), 1e-6)

    def gt(i):
        """One batch of tokens WITH A HISTORY AXIS (B, K, ...), newest frame last, plus the static environment
        patch. The window is clamped at the episode start rather than zero-padded: repeating the first frame is
        a truthful statement that nothing earlier exists, while a zero row is a state the game cannot be in."""
        K = a.hist
        w = window_index(i, K, RSTART[i]) if K > 1 else i[:, None]
        o = {k: (tt[k][w].long() if k in _LONG else tt[k][w].float()) for k in tt}
        # HP bins, from raw HP recovered with the fitted scale
        o["lhp_bin"] = (o["link"][:, :, L_HP].clamp(0, 1) * (a.hp_bins - 1)).round().long()
        o["ahp_bin"] = (o["actors"][:, :, :, A_HP].clamp(0, 1) * (a.hp_bins - 1)).round().long()
        # DUAL LOCAL FRAMES per history frame: fwd/right in Link's FACING frame and in the CAMERA frame.
        B = o["link"].shape[0]
        rel = (o["actor_rcell"].reshape(B * K, -1, 2).float() - REL_CLIP)
        o["frames"] = local_frames(rel, o["actor_off"].reshape(B * K, -1, 2),
                                   o["link_off"].reshape(B * K, 2), o["link"].reshape(B * K, -1)[:, 3:5],
                                   o["link"].reshape(B * K, -1)[:, 9:11], CELLWH,
                                   scale=float(max(grid.cell)) * 8.0).reshape(B, K, -1, 4)
        # static environment patch around the NEWEST frame's Link cell
        occ, _ = wall_patch(o["link_cell"][:, -1], WMAP, ENV_OFF)
        o["env_occ"] = occ
        # ctx_D is query -> env cell, so it must be measured from EACH entity, not just from Link. Link sits at
        # offset (0,0); actor r is at its relative cell. Broadcasting Link's distances would be wrong for actors.
        ent = torch.cat([torch.zeros(B, 1, 2, device=dev), o["cell_xy"][:, -1]], 1)        # (B, 1+R, 2)
        o["env_d"] = (ent[:, :, None, :] - ENV_OFF.float()[None, None]).norm(dim=-1)       # (B, 1+R, P)
        o["rel_xz"] = o["actors"][:, -1, :, :2]
        return o

    def yg(k, i):
        return Y[k][i].long() if k in _YLONG else Y[k][i].float()

    def ag(i):
        return F.one_hot(A[i].long(), NA).float()
    # An H-step autoregressive sample needs H consecutive in-episode transitions from its start.
    _ends = np.flatnonzero(~ok)
    _runlen = _ends[np.searchsorted(_ends, np.arange(N))] - np.arange(N)
    DEFAULT_MIX = "1:60,2:15,4:8,8:5,16:4,32:3,64:2,128:1.2,300:0.9,600:0.5,1200:0.25,1800:0.15"
    _mix = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in (a.h_mix or DEFAULT_MIX).split(",")] \
        if (a.h_mix or a.train_h > 1) else [(a.train_h, 1.0)]
    H_VALS = np.array([h for h, _ in _mix], np.int64)
    H_P = np.array([w for _, w in _mix], np.float64); H_P = H_P / H_P.sum()
    H_MAX = int(H_VALS.max())
    idx = lambda m: np.flatnonzero(m & ok & (_runlen >= 1))                   # noqa: E731
    tr, tu, te = idx(tr_m), idx(np.isin(slot, list(tune))), idx(np.isin(slot, list(test)))
    # Starts sorted by DESCENDING room-to-episode-end, so "every start with runlen >= H" is a prefix and can be
    # drawn in O(1) for any H. Rejection sampling would be hopeless at H=1800, where only the opening frames of
    # a full-length timeout episode qualify.
    _tro = tr[np.argsort(-_runlen[tr], kind="stable")]
    _trl = _runlen[_tro]

    def starts_for(H, n):
        n_ok = int(np.searchsorted(-_trl, -H, side="right"))
        return None if n_ok < n else _tro[rng.choice(n_ok, n, replace=False)]

    def step_weights(H):
        """0.5^((h-1)/half_life), renormalised so total gradient scale is independent of depth."""
        w = 0.5 ** (np.arange(H, dtype=np.float64) / max(a.h_half_life, 1e-9))
        return (w / w.sum()).tolist()
    if len(H_VALS) > 1:
        eh = float((H_VALS * H_P).sum())
        share = (H_VALS * H_P) / eh
        print("depth mixture: " + "  ".join(f"H{h}:{100*pp:.2f}%(cost {100*c:.0f}%)"
                                            for h, pp, c in zip(H_VALS, H_P, share)))
        print(f"  mean depth {eh:.1f} | step decay half-life {a.h_half_life:g} | bptt {a.bptt}")
        for h in (1, 8, 128, 1800):
            if h in H_VALS:
                w = step_weights(h)
                print(f"  H={h:<5} step weights: first {w[0]:.4f} last {w[-1]:.4f}"
                      + (f" mid {w[len(w)//2]:.4f}" if h > 2 else ""))

    # Rare-event class weights, sqrt-inverse-frequency. HP change is ~0.1% of steps, and unweighted the head
    # simply never predicts it (measured: hp loss-recall 0.000, accuracy 0.9987 = the base rate). This is the
    # same treatment that took HP damage F1 from 0.286 to 0.906; a RAW inverse gave the rare class weight
    # 0.0005 and it was never predicted either, hence sqrt.
    def cw(lab, k, m=None):
        v = lab.reshape(-1)[(m.reshape(-1) > 0.5) if m is not None else slice(None)]
        c = np.bincount(np.asarray(v).astype(np.int64), minlength=k).astype(np.float64)
        w = np.sqrt(c.sum() / np.maximum(c, 1.0))
        return torch.from_numpy((w / w.mean()).astype(np.float32)).to(dev)

    CW_HP = cw(hp_ev[tr], 3)
    CW_RHP = cw(r_hp_ev[tr], 3, pres_now[tr])
    # dcell is ~94% "same cell"; without a weight the head predicts only the centre class and never a move.
    CW_DC = cw(dcell_i[tr], DC) if a.dcell_weight else None
    # HP is now a bounded DELTA class, so it needs its own sqrt-inverse-frequency weights: "no change" is
    # ~99.8% of steps and unweighted the head would simply never predict damage (measured: recall 0.000).
    _lhp = np.asarray(P["link"][:, L_HP], np.float32) * sc.get("l_hp", 1.0)
    _lhpd = np.clip(np.rint((_lhp[np.minimum(np.arange(N) + 1, N - 1)] - _lhp) / HPD_STEP),
                    -HPD_K, HPD_K).astype(np.int64) + HPD_K
    CW_HPD = cw(_lhpd[tr], HPD_N)
    _ahp = np.asarray(P["actors"][:, :, A_HP], np.float32) * sc.get("a_hp", 1.0)
    _ahpd = np.clip(np.rint(_ahp[np.minimum(np.arange(N) + 1, N - 1)] - _ahp),
                    -AHPD_K, AHPD_K).astype(np.int64) + AHPD_K
    CW_RHPD = cw(_ahpd[tr], AHPD_N, pres_now[tr])
    print(f"hp delta classes: link {HPD_N} (step {HPD_STEP:g} raw), actor {AHPD_N} | "
          f"link no-change rate {(_lhpd[tr] == HPD_K).mean():.4f}")
    print(f"hp event rates train: {np.bincount(hp_ev[tr], minlength=3) / len(tr)} -> weights {CW_HP.cpu().numpy().round(2)}")

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

    net = TokenWM(NA, a.dim, a.heads, n_rows=a.rows, attn=a.attn, layers=a.layers,
                  hist=a.hist, env_p=int(ENV_OFF.shape[0]), hp_bins=a.hp_bins, grid_n=a.grid,
                  gate=a.gate).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = None
    if a.sched == "cosine":
        # Per-EPOCH, so its state is one integer and it survives a resume with the optimizer.
        def lr_at(e):
            if e < a.warmup:
                return (e + 1) / max(a.warmup, 1e-9)
            t = (e - a.warmup) / max(a.epochs - a.warmup, 1e-9)
            return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"{N:,} frames | train {len(tr):,} tune {len(tu):,} test {len(te):,} | link tok {LINK_DIM} "
          f"actor tok {ACTOR_DIM} | d {a.dim} heads {a.heads} | params {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    amp = bool(a.amp and dev.type == "cuda")
    if a.amp and dev.type != "cuda":
        print("--amp ignored: not on cuda")
    nb_ = len(SF1_KEEP)

    def split_link(o, i, cur=None):
        """cur = the state the logit residuals are relative to. In one-step training that is the TRUE state at
        i; in an autoregressive rollout it MUST be the model's own rolled state, or the residual leaks ground
        truth into the rollout."""
        cur = cur or {"bits": tt["link"][i].float()[:, 16:16 + nb_],
                      "scene": yg("scene_now", i),
                      "bg": tt["link"][i].float()[:, 16 + nb_:]}
        j = 0
        dcl, j = o[:, j:j + DC], j + DC
        dof, j = o[:, j:j + 2], j + 2
        dyv, j = o[:, j:j + 1], j + 1
        dyw, j = o[:, j:j + 2], j + 2
        hpd, j = o[:, j:j + HPD_N], j + HPD_N
        bits, j = o[:, j:j + nb_], j + nb_
        sc_, j = o[:, j:j + 2], j + 2
        dvel, j = o[:, j:j + 3], j + 3
        dcam, j = o[:, j:j + 2], j + 2
        dfl, j = o[:, j:j + 1], j + 1
        bg = o[:, j:j + len(BG_BITS)]
        # LOGIT RESIDUALS so argmax/sign at init equals the CURRENT value -> exact persistence
        dcl = dcl + LOGIT_K * F.one_hot(torch.full((o.shape[0],), CENTRE, device=dev), DC).float()
        hpd = hpd + LOGIT_K * F.one_hot(torch.full((o.shape[0],), HPD_K, device=dev), HPD_N).float()
        bits = bits + LOGIT_K * (2 * cur["bits"] - 1)
        bg = bg + LOGIT_K * (2 * cur["bg"] - 1)
        sc_ = sc_ + LOGIT_K * F.one_hot(cur["scene"], 2).float()
        return (dcl, dof), dyv, dyw, hpd, bits, sc_, dvel, dcam, dfl, bg

    def split_actor(o, i, cur=None):
        cur = cur or {"pres": yg("pres_now", i), "st": yg("st_now", i)}
        j = 0
        rdcl, j = o[..., j:j + DC], j + DC
        rdof, j = o[..., j:j + 2], j + 2
        hpd, j = o[..., j:j + AHPD_N], j + AHPD_N
        pr, j = o[..., j:j + 1], j + 1
        st, j = o[..., j:j + N_STATES], j + N_STATES
        drely, j = o[..., j:j + 1], j + 1
        dyaw, j = o[..., j:j + 2], j + 2
        dshape, j = o[..., j:j + 2], j + 2
        dtimer, j = o[..., j:j + 1], j + 1
        dvel = o[..., j:j + 3]
        hpd = hpd + LOGIT_K * F.one_hot(torch.full(o.shape[:2], AHPD_K, device=dev), AHPD_N).float()
        pr = pr + LOGIT_K * (2 * cur["pres"][..., None] - 1)
        st = st + LOGIT_K * F.one_hot(cur["st"], N_STATES).float()
        rdcl = rdcl + LOGIT_K * F.one_hot(torch.full(o.shape[:2], CENTRE, device=dev), DC).float()
        return (rdcl, rdof), hpd, pr, st, drely, dyaw, dshape, dtimer, dvel

    # Column offsets inside the packed link/actor token vectors, derived from LINK_FIELDS / ACTOR_FIELDS order
    # (rel_y, hp, hp_frac, yaw sin/cos, speed, vel x/z/y, cam_yaw sin/cos, heading sin/cos, floor_dy,
    #  n_present, n_baba, then the sf1 bits, then the bg bits).
    from zelda_rl.bc import BABA_ID  # noqa: E402
    # Median NON-ZERO change in raw Link HP, used to turn the 3-class HP event into a magnitude during a
    # rollout. The head predicts same/down/up, not how much, so a rollout needs some step size; the data's own
    # median is the least arbitrary choice. Taken over a CONTIGUOUS slice so the diffs are consecutive frames.

    # ---- AUTOREGRESSIVE STATE ADVANCE -------------------------------------------------------------------
    # Rebuild the token state at t+1 from the model's OWN predictions, so training can roll forward instead of
    # being teacher-forced at every step. This is what makes training match deployment: the dream feeds the
    # model its own output, so the model should be trained on its own output's distribution.
    #
    # FULL CLOSURE: every field the model consumes now has a head or an exact derivation, so NOTHING is
    # carried stale through a rollout. Predicted: Link cell+offset, rel_y, yaw, HP (delta class, so magnitude
    # not just direction), sf1 bits, scene, velocity, camera yaw, floor_dy, the 9 bg contact bits; and per
    # actor cell+offset, HP, presence, behaviour state, rel_y, yaw, shape_yaw, timer, velocity.
    # Derived exactly from those: speed = |vel_xz|, heading_rel_cam = yaw - cam_yaw, hp_frac = hp,
    # n_present / n_baba from predicted presence, bearing and yaw_to_player from predicted positions.
    # Carried only: actor id and cat, which are identity rather than dynamics.
    #
    # STOCHASTIC: the discrete heads are SAMPLED, not argmaxed. A Baba has a hidden inner state we never
    # observe, so its next behaviour is genuinely random from our side; the head's distribution is the
    # marginal over that hidden state, and sampling from it is the statistically correct rollout. argmax would
    # commit to one fictional future and make every rollout deterministic.
    SC_RELY, SC_HP = sc.get("l_rel_y", 1.0), sc.get("l_hp", 1.0)


    def advance(tok, ol, oa, cur, sample=True):
        """tok -> tok at t+1 from the model's OWN predictions, SHIFTING the history window by one frame.

        With a K-frame window the newest frame is the current state; advancing drops the oldest and appends the
        prediction, which is what the model will actually be fed at deployment. Nothing is carried stale except
        actor identity. `sample=True` draws from the discrete heads rather than argmaxing, because a Baba's
        inner state is unobservable and its next behaviour is genuinely stochastic from our side.
        """
        (dcl, dof), dyv, dyw, hpd, bits, scn, dvel, dcam, dfl, bg = split_link(ol, None, cur["link"])
        (rdcl, rdof), rhpd, rpr, rst, a_drely, a_dyaw, a_dshape, a_dtimer, a_dvel = \
            split_actor(oa, None, cur["actor"])
        if a.gate:
            # THE HALF THAT MATTERS FOR A ROLLOUT: where the gate says nothing changed, the field advances by
            # EXACTLY zero rather than by a small wrong delta. 1800 steps of small wrong deltas on heads whose
            # target is zero 95-99.97% of the time is how wm_v3 ended up worse than doing nothing.
            gl = (ol[:, -len(GATE_L):] > 0).float()
            ga = (oa[..., -len(GATE_A):] > 0).float()
            dyw = dyw * gl[:, 0:1]
            rdof = rdof * ga[..., 0:1]
            a_dyaw = a_dyaw * ga[..., 1:2]
            a_dshape = a_dshape * ga[..., 2:3]
            a_dtimer = a_dtimer * ga[..., 3:4]

        def pick(logits):
            if not sample:
                return logits.argmax(-1)
            sh = logits.shape[:-1]
            return torch.multinomial(logits.reshape(-1, logits.shape[-1]).softmax(-1), 1).reshape(sh)

        def flip(logits):
            pr = logits.sigmoid()
            return (torch.rand_like(pr) < pr).float() if sample else (pr > 0.5).float()

        now = {k: (tok[k][:, -1] if (a.hist > 1 and k in HIST_FIELDS) else tok[k]) for k in tok}
        B = now["link"].shape[0]
        k = pick(dcl)
        dcell = torch.stack([torch.div(k, DW, rounding_mode="floor"), k % DW], -1) - DW // 2
        n_cell = (now["link_cell"] + dcell).clamp(0, grid.n - 1)
        n_off = (now["link_off"] + dof).clamp(-0.5, 0.5)
        Lo = now["link"].float()
        rely = (Lo[:, L_RELY:L_RELY + 1] + dyv).clamp(-1, 1)
        hp = (Lo[:, L_HP:L_HP + 1] + (pick(hpd).float() - HPD_K)[:, None] * HPD_STEP / max(SC_HP, 1e-6)).clamp(0, 1)
        yaw = (Lo[:, L_YAW] + dyw).clamp(-1, 1)
        vel = (Lo[:, L_VEL] + dvel).clamp(-1, 1)
        cam = (Lo[:, L_CAM] + dcam).clamp(-1, 1)
        floor = (Lo[:, L_FLOOR:L_FLOOR + 1] + dfl).clamp(-1, 1)
        nbits, nbg = flip(bits), flip(bg)
        speed = vel[:, :2].norm(dim=-1, keepdim=True).clamp(-1, 1)          # derived exactly
        head = (yaw - cam).clamp(-1, 1)                                     # derived exactly
        rk = pick(rdcl)
        rd = torch.stack([torch.div(rk, DW, rounding_mode="floor"), rk % DW], -1) - DW // 2
        abs_cell = (now["link_cell"][:, None, :] + now["actor_rcell"] - REL_CLIP) + rd
        rel = (abs_cell - n_cell[:, None, :]).clamp(-REL_CLIP, REL_CLIP)
        n_rcell = (rel + REL_CLIP).clamp(0, 2 * REL_CLIP)
        n_aoff = (now["actor_off"] + rdof).clamp(-0.5, 0.5)
        n_mask = flip(rpr.squeeze(-1))
        n_states = pick(rst)
        Ao = now["actors"].float()
        a_hp = (Ao[:, :, A_HP:A_HP + 1]
                + (pick(rhpd).float() - AHPD_K)[..., None] / max(sc.get("a_hp", 1.0), 1e-6)).clamp(0, 1)
        a_rely = (Ao[:, :, A_RELY:A_RELY + 1] + a_drely).clamp(-1, 1)
        a_yaw = (Ao[:, :, A_YAW] + a_dyaw).clamp(-1, 1)
        a_shape = (Ao[:, :, A_SHAPE] + a_dshape).clamp(-1, 1)
        a_timer = (Ao[:, :, A_TIMER:A_TIMER + 1] + a_dtimer).clamp(-1, 1)
        a_vel = (Ao[:, :, A_VEL] + a_dvel).clamp(-1, 1)
        dx = rel[..., 0].float() + n_aoff[..., 0] - n_off[:, None, 0]
        dz = rel[..., 1].float() + n_aoff[..., 1] - n_off[:, None, 1]
        ang = torch.atan2(dx * float(grid.cell[0]), dz * float(grid.cell[1]))
        yaw_ang = torch.atan2(yaw[:, 0], yaw[:, 1])[:, None]
        bear = torch.stack([ang.sin(), ang.cos()], -1)
        y2p = torch.stack([(ang - yaw_ang).sin(), (ang - yaw_ang).cos()], -1)
        n_actors = torch.cat([a_rely, bear, a_hp, a_yaw, a_shape, y2p, a_timer, a_vel],
                             -1) * n_mask[:, :, None]
        npres = (n_mask.sum(1, keepdim=True) / max(sc.get("l_n_present", 1.0), 1e-6)).clamp(-1, 1)
        nbaba = (((now["ids"] == BABA_ID).float() * n_mask).sum(1, keepdim=True)
                 / max(sc.get("l_n_baba", 1.0), 1e-6)).clamp(-1, 1)
        n_link = torch.cat([rely, hp, hp, yaw, speed, vel, cam, head, floor, npres, nbaba, nbits, nbg], -1)
        # derived inputs for the new frame, exactly as gt() builds them
        n_lhp = (n_link[:, L_HP].clamp(0, 1) * (a.hp_bins - 1)).round().long()
        n_ahp = (n_actors[:, :, A_HP].clamp(0, 1) * (a.hp_bins - 1)).round().long()
        n_frames = local_frames(rel.float(), n_aoff, n_off, n_link[:, 3:5], n_link[:, 9:11],
                                CELLWH, scale=float(max(grid.cell)) * 8.0)
        newf = {"link": n_link, "actors": n_actors, "ids": now["ids"], "cats": now["cats"],
                "states": n_states, "mask": n_mask, "link_cell": n_cell, "link_off": n_off,
                "actor_rcell": n_rcell, "actor_off": n_aoff, "cell_xy": rel.float(),
                "lhp_bin": n_lhp, "ahp_bin": n_ahp, "frames": n_frames}
        # SHIFT the window: drop the oldest frame, append the prediction
        nt = {}
        for kk, v in tok.items():
            if kk in HIST_FIELDS:
                nt[kk] = torch.cat([v[:, 1:], newf[kk][:, None].to(v.dtype)], 1) if v.shape[1] > 1 \
                    else newf[kk][:, None].to(v.dtype)
            else:
                nt[kk] = v
        # the environment patch follows Link, so it must be recomputed from his NEW cell
        occ, _ = wall_patch(n_cell, WMAP, ENV_OFF)
        nt["env_occ"] = occ
        ent = torch.cat([torch.zeros(B, 1, 2, device=dev), rel.float()], 1)
        nt["env_d"] = (ent[:, :, None, :] - ENV_OFF.float()[None, None]).norm(dim=-1)
        nt["rel_xz"] = n_actors[:, :, :2]
        nxt_cur = {"link": {"bits": nbits, "scene": pick(scn), "bg": nbg},
                   "actor": {"pres": n_mask, "st": n_states}}
        return nt, nxt_cur

    HIST_FIELDS = ("link", "actors", "ids", "cats", "states", "mask", "link_cell", "link_off",
                   "actor_rcell", "actor_off", "cell_xy", "lhp_bin", "ahp_bin", "frames")

    def head_losses(ol, oa, j, cur=None, tok=None):
        """Twelve-plus typed heads, with every target taken RELATIVE TO THE STATE THE MODEL IS ACTUALLY IN.

        In one-step training `tok` is the true state at j, so these reduce exactly to the old targets. In an
        autoregressive rollout `tok` is the model's own drifted state, and targeting
        (true state at j+1) - (rolled state at j) teaches DRIFT CORRECTION. Targeting the true delta from the
        true state instead would have the model faithfully reproduce the right increment from the wrong place,
        i.e. perpetuate its own error -- which is the compounding failure we keep measuring.
        """
        # the model's CURRENT state is the NEWEST history frame; defined first because several targets below
        # need it
        cur_of = (lambda k: tok[k][:, -1] if (a.hist > 1 and k in HIST_FIELDS) else tok[k])
        (dcl, dof), dyv, dyw, hpd, bits, scn, dvel, dcam, dfl, bg = \
            split_link(ol, j, cur["link"] if cur else None)
        (rdcl, rdof), rhpd, rpr, rst, a_drely, a_dyaw, a_dshape, a_dtimer, a_dvel = \
            split_actor(oa, j, cur["actor"] if cur else None)
        jn = (j + 1).clamp(max=N - 1)
        m = yg("pres_now", j) if tok is None else cur_of("mask")
        L_now = tt["link"][j].float() if tok is None else cur_of("link").float()
        L_nxt = tt["link"][jn].float()
        A_now = tt["actors"][j].float() if tok is None else cur_of("actors").float()
        A_nxt = tt["actors"][jn].float()
        c_now = tt["link_cell"][j].long() if tok is None else cur_of("link_cell").long()
        o_now = tt["link_off"][j].float() if tok is None else cur_of("link_off").float()
        # --- Link position: which of the 5x5 neighbouring cells, relative to where the model IS
        dcell_t = (tt["link_cell"][jn].long() - c_now).clamp(-(DW // 2), DW // 2) + DW // 2
        dcell_i = dcell_t[:, 0] * DW + dcell_t[:, 1]
        doff_t = tt["link_off"][jn].float() - o_now
        # --- HP as a bounded DELTA class in raw units, so a rollout gets a magnitude not just a direction
        hp_raw_now = L_now[:, L_HP] * SC_HP
        hpd_t = (((tt["link"][jn].float()[:, L_HP] * SC_HP - hp_raw_now) / HPD_STEP)
                 .round().clamp(-HPD_K, HPD_K) + HPD_K).long()
        # --- actors: absolute cell now (from the rolled rcell) -> delta to the true next absolute cell
        ac_now = (c_now[:, None, :] + (cur_of("actor_rcell").long() if tok is not None
                                      else tt["actor_rcell"][j].long()) - REL_CLIP)
        ac_nxt = (tt["link_cell"][jn].long()[:, None, :] + tt["actor_rcell"][jn].long() - REL_CLIP)
        rdc_t = (ac_nxt - ac_now).clamp(-(DW // 2), DW // 2) + DW // 2
        rdc_i = rdc_t[..., 0] * DW + rdc_t[..., 1]
        rdoff_t = tt["actor_off"][jn].float() - (cur_of("actor_off").float() if tok is not None
                                                else tt["actor_off"][j].float())
        ahp_t = (((A_nxt[:, :, A_HP] - A_now[:, :, A_HP]) * sc.get("a_hp", 1.0))
                 .round().clamp(-AHPD_K, AHPD_K) + AHPD_K).long()
        msum = m.sum().clamp(min=1)
        mrow = lambda x: (x * m).sum() / msum                                   # noqa: E731
        # ---- change gates on the sparse regression heads -------------------------------------------------
        # A zero-init regression head STARTS at persistence but nothing keeps it there: trained with L1 over
        # frames that are 95-99.97% zero it learns to emit small wrong values everywhere, and in a 1800-step
        # rollout that is how wm_v3 ended up worse than doing nothing. The gate supervises the value head only
        # where the field changed and, at rollout time, zeroes the delta where the gate says nothing happened.
        gl = ol[:, -len(GATE_L):] if a.gate else None
        ga = oa[..., -len(GATE_A):] if a.gate else None
        def gated(name, err, tgt, idx, row=False):
            """err: per-element |pred-target| (already mean-reduced over the field's width). tgt: the target,
            used only to decide what CHANGED. Returns the value loss; records the gate's BCE in GB."""
            if not a.gate:
                return mrow(err) if row else err.mean()
            lg = (ga[..., idx] if row else gl[:, idx])
            chg = (tgt.abs().amax(-1) if tgt.dim() > (2 if row else 1) else tgt.abs()) > 1e-6
            b = F.binary_cross_entropy_with_logits(lg, chg.float(), reduction="none")
            GB.append(mrow(b) if row else b.mean())
            w = chg.float()
            den = ((w * m).sum() if row else w.sum()).clamp(min=1)
            return (((err * w) * m).sum() if row else (err * w).sum()) / den
        GB = []
        L = {"l_dcell": F.cross_entropy(dcl, dcell_i, weight=CW_DC),
             "l_doff": F.l1_loss(dof, doff_t),
             "l_y": F.l1_loss(dyv.squeeze(-1), L_nxt[:, L_RELY] - L_now[:, L_RELY]),
             "l_yaw": gated("l_yaw", (dyw - (L_nxt[:, L_YAW] - L_now[:, L_YAW])).abs().mean(-1),
                             L_nxt[:, L_YAW] - L_now[:, L_YAW], 0),
             "l_hp": F.cross_entropy(hpd, hpd_t, weight=CW_HPD),
             "l_bits": F.binary_cross_entropy_with_logits(bits, L_nxt[:, 16:16 + nb_]),
             "l_scene": F.cross_entropy(scn, yg("scene_nxt", j)),
             # --- new closure heads: everything the model reads, it now also predicts
             "l_vel": F.l1_loss(dvel, L_nxt[:, L_VEL] - L_now[:, L_VEL]),
             "l_cam": F.l1_loss(dcam, L_nxt[:, L_CAM] - L_now[:, L_CAM]),
             "l_floor": F.l1_loss(dfl.squeeze(-1), L_nxt[:, L_FLOOR] - L_now[:, L_FLOOR]),
             "l_bg": F.binary_cross_entropy_with_logits(bg, L_nxt[:, 16 + nb_:]),
             "a_dcell": mrow(F.cross_entropy(rdcl.reshape(-1, DC), rdc_i.reshape(-1),
                                             reduction="none").reshape(m.shape)),
             "a_doff": gated("a_doff", (rdof - rdoff_t).abs().mean(-1), rdoff_t, 0, row=True),
             "a_hp": mrow(F.cross_entropy(rhpd.reshape(-1, AHPD_N), ahp_t.reshape(-1), weight=CW_RHPD,
                                          reduction="none").reshape(m.shape)),
             "a_pres": F.binary_cross_entropy_with_logits(rpr.squeeze(-1), yg("pres_nxt", j)),
             "a_state": mrow(F.cross_entropy(rst.reshape(-1, N_STATES), yg("st_nxt", j).reshape(-1),
                                             reduction="none").reshape(m.shape)),
             "a_rely": mrow((a_drely.squeeze(-1) - (A_nxt[:, :, A_RELY] - A_now[:, :, A_RELY])).abs()),
             "a_yaw": gated("a_yaw", (a_dyaw - (A_nxt[:, :, A_YAW] - A_now[:, :, A_YAW])).abs().mean(-1),
                             A_nxt[:, :, A_YAW] - A_now[:, :, A_YAW], 1, row=True),
             "a_shape": gated("a_shape", (a_dshape - (A_nxt[:, :, A_SHAPE] - A_now[:, :, A_SHAPE])).abs().mean(-1),
                               A_nxt[:, :, A_SHAPE] - A_now[:, :, A_SHAPE], 2, row=True),
             "a_timer": gated("a_timer", (a_dtimer.squeeze(-1) - (A_nxt[:, :, A_TIMER] - A_now[:, :, A_TIMER])).abs(),
                               A_nxt[:, :, A_TIMER] - A_now[:, :, A_TIMER], 3, row=True),
             "a_vel": mrow((a_dvel - (A_nxt[:, :, A_VEL] - A_now[:, :, A_VEL])).abs().mean(-1))}
        if a.gate and GB:
            L["gate_bce"] = GATE_W * sum(GB) / len(GB)   # x0.2: the yes/no loss crowds out the value heads
        return L, ((dcl, dof), hpd)

    def loss_fn(i):
        """One step, teacher-forced: the inputs are the TRUE state at i."""
        ol, oa = net(gt(i), ag(i))
        L, extra = head_losses(ol, oa, i, None)
        return sum(L.values()), L, extra

    def loss_ar(i, H, wts, do_backward=True):
        """AUTOREGRESSIVE: encode the true state once, then feed the model its OWN predicted state forward.

        Loss at every step against ground truth, weighted by `wts` (step decay, already summing to 1) so the
        gradient scale is independent of depth.

        BACKWARD IS DONE PER BPTT WINDOW, not once at the end. Detaching the token state alone does NOT bound
        memory: every step's loss tensor keeps its own subgraph alive, so summing 1800 of them retains 1800
        windows of activations and OOMs an 80 GB card at batch 8192. Calling backward at each window boundary
        and then detaching frees it, and because the windows are independent after the detach the accumulated
        gradient is the same.
        """
        tok = gt(i)
        cur = {"link": {"bits": tt["link"][i].float()[:, 16:16 + nb_], "scene": yg("scene_now", i),
                        "bg": tt["link"][i].float()[:, 16 + nb_:]},
               "actor": {"pres": yg("pres_now", i), "st": yg("st_now", i)}}
        parts, extra, win, total = {}, None, None, 0.0
        W = max(a.bptt, 1) if a.bptt else H
        for st_ in range(H):
            j = i + st_
            ol, oa = net(tok, ag(j))
            L, ex = head_losses(ol, oa, j, cur, tok)
            if st_ == 0:
                extra = ex
            step = sum(v * wts[st_] for v in L.values())
            win = step if win is None else win + step
            for k, v in L.items():
                parts[f"h{st_+1}/{k}"] = float(v.detach()) * wts[st_]
            last = (st_ == H - 1)
            if do_backward and ((st_ + 1) % W == 0 or last):
                win.backward()
                total += float(win.detach())
                win = None
                if not last:
                    tok = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in tok.items()}
                    cur = {g: {k: (v.detach() if torch.is_tensor(v) else v) for k, v in d.items()}
                           for g, d in cur.items()}
            elif not last and (st_ + 1) % W == 0:
                tok = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in tok.items()}
        return total, parts, extra

    def eval_pos(sel):
        """Held-out one-step position error in WORLD UNITS, decoded through the same grid codec as the inputs,
        against the persistence control. Shared by the LR sweep and the real run so the two are comparable."""
        iu = torch.from_numpy(sel).to(dev)
        _, parts, (dp, hp) = loss_fn(iu)
        pc = dp[0].argmax(-1).cpu().numpy()
        pcell = np.stack([pc // DW, pc % DW], 1) - DW // 2 + T["link_cell"][sel]
        pred_xz = grid.from_cell(pcell, (T["link_off"][sel] + dp[1].float().cpu().numpy()).clip(-0.5, 0.5))
        true_xz = lpos[nxt[sel]]
        werr = float(np.linalg.norm(pred_xz - true_xz, axis=1).mean())
        pers = float(np.linalg.norm(lpos[sel] - true_xz, axis=1).mean())
        dcell_acc = float((dp[0].argmax(-1) == yg("dcell", iu)).float().mean())
        # hp is a bounded DELTA class now; "damage" is any class below the no-change centre, and recall on
        # those is the number that matters because they are ~0.2% of steps.
        _lh = tt["link"][iu].float()[:, L_HP] * SC_HP
        _lhn = tt["link"][(iu + 1).clamp(max=N - 1)].float()[:, L_HP] * SC_HP
        hpv = (((_lhn - _lh) / HPD_STEP).round().clamp(-HPD_K, HPD_K) + HPD_K).long()
        hp_pred = hp.argmax(-1)
        hp_acc = float((hp_pred == hpv).float().mean())
        dmg = hpv < HPD_K
        hp_rec = float((hp_pred[dmg] < HPD_K).float().mean()) if dmg.any() else float("nan")
        return parts, werr, pers, dcell_acc, hp_acc, hp_rec

    def run_epochs(n_epochs, lr):
        """Train a FRESH model for n_epochs at `lr`. Used by --lr-sweep, which rebinds `net`/`opt` in this
        scope -- loss_fn closes over the NAMES, so it picks up the new model automatically."""
        nonlocal net, opt
        torch.manual_seed(a.seed)
        net = TokenWM(NA, a.dim, a.heads, n_rows=a.rows, attn=a.attn, layers=a.layers,
                  hist=a.hist, env_p=int(ENV_OFF.shape[0]), hp_bins=a.hp_bins, grid_n=a.grid,
                  gate=a.gate).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        r = {}
        for ep in range(n_epochs):
            net.train()
            perm = rng.permutation(tr)
            gn, nb, tot = 0.0, 0, 0.0
            for st in range(0, len(perm) - a.batch, a.batch):
                i = torch.from_numpy(perm[st:st + a.batch]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss, _, _ = loss_fn(i)
                opt.zero_grad(); loss.backward()
                g = float(nn.utils.clip_grad_norm_(net.parameters(), a.clip))
                opt.step()
                tot += float(loss.detach()); gn += g; nb += 1
                if not np.isfinite(g):
                    r["diverged"] = True
                    print(f"    !! non-finite gradient at step {nb}")
                    return r
            net.eval()
            with torch.no_grad():
                _, werr, pers, dcell, _, hp_rec = eval_pos(tu[:8192])
                roll = autoregressive(tu[:2048], a.rollout_h)
            sk = 1 - werr / max(pers, 1e-9)
            r = {"epoch": ep + 1, "loss": tot / max(nb, 1), "grad": gn / max(nb, 1), "skill": sk,
                 "dcell": dcell, "hp_rec": hp_rec, "steps": nb * (ep + 1),
                 **{k: v for k, v in roll.items() if k.endswith("_skill")}}
            print(f"    ep {ep+1}/{n_epochs} loss {tot/max(nb,1):.4f} grad {gn/max(nb,1):.1f} "
                  f"skill {sk:+.4f} dcell {dcell:.4f} hp_rec {hp_rec:.3f} "
                  + " ".join(f"H{h} {roll.get(f'roll/h{h}_skill', float('nan')):+.4f}" for h in a.rollout_h))
        return r

    def autoregressive(idx_np, horizons):
        """Roll the FULL state forward on the model's own predictions and score position against ground truth.

        This used to integrate only Link's cell + offset and hold every other token field at its step-0 value,
        which made it a lower bound and, after full closure, simply the wrong measurement: training rolls the
        whole state through `advance`, so the eval must too or the two are measuring different models.

        Deterministic here (sample=False): a metric wants the model's modal trajectory, not one draw from it.
        Training samples, because there the point is to see the distribution.
        """
        out = {}
        with torch.no_grad():
            _ii = torch.from_numpy(idx_np).to(dev)
            tok = gt(_ii)
            cur = {"link": {"bits": tt["link"][_ii].float()[:, 16:16 + nb_],
                            "scene": yg("scene_now", _ii),
                            "bg": tt["link"][_ii].float()[:, 16 + nb_:]},
                   "actor": {"pres": yg("pres_now", _ii), "st": yg("st_now", _ii)}}
            start_xz = lpos[idx_np]
            hset = set(horizons)
            for h in range(1, max(horizons) + 1):
                _ih = torch.from_numpy(np.minimum(idx_np + h - 1, N - 1)).to(dev)
                ol, oa = net(tok, ag(_ih))
                tok, cur = advance(tok, ol, oa, cur, sample=False)
                if h in hset:
                    # the token state carries a history axis; the CURRENT frame is the newest one
                    cell = tok["link_cell"][:, -1].cpu().numpy()
                    off = tok["link_off"][:, -1].float().cpu().numpy()
                    pred_xz = grid.from_cell(cell, off)
                    true_xz = lpos[np.minimum(idx_np + h, N - 1)]
                    err = float(np.linalg.norm(pred_xz - true_xz, axis=1).mean())
                    per = float(np.linalg.norm(start_xz - true_xz, axis=1).mean())
                    out[f"roll/h{h}_err"] = err
                    out[f"roll/h{h}_persistence"] = per
                    out[f"roll/h{h}_skill"] = 1 - err / max(per, 1e-9)
        return out

    if a.lr_sweep:
        # ONE process, ONE load. Five separate processes each paid ~3 minutes to load and tokenise from the
        # pod's network filesystem and spent 18 seconds training, which burned a 20-minute budget on four
        # redundant loads. The model is re-initialised from the same seed for every LR, so the only thing
        # that differs between rows is the learning rate.
        lrs = [float(x) for x in a.lr_sweep.split(",")]
        print(f"\nLR SWEEP: {lrs} | {a.epochs} epochs each | batch {a.batch} | dim {a.dim} | "
              f"{len(tr):,} train frames ({len(tr) // a.batch} steps/epoch)")
        table = []
        for lr in lrs:
            print(f"  --- lr={lr:g} ---")
            t0 = time.time()
            r = run_epochs(a.epochs, lr)
            r["lr"] = lr; r["secs"] = time.time() - t0
            table.append(r)
        hh = a.rollout_h
        print("\n" + "=" * 100)
        print(f"{'lr':>8} {'loss':>8} {'grad':>7} {'1step':>9} {'dcell':>7} {'hp_rec':>7} "
              + " ".join(f"{'H'+str(h):>9}" for h in hh))
        for r in table:
            if r.get("diverged"):
                print(f"{r['lr']:>8.1e}   DIVERGED (non-finite gradient)")
                continue
            print(f"{r['lr']:>8.1e} {r['loss']:>8.4f} {r['grad']:>7.1f} {r['skill']:>+9.4f} "
                  f"{r['dcell']:>7.4f} {r['hp_rec']:>7.3f} "
                  + " ".join(f"{r.get(f'roll/h{h}_skill', float('nan')):>+9.4f}" for h in hh))
        ok = [r for r in table if not r.get("diverged")]
        if ok:
            b = max(ok, key=lambda r: r.get(f"roll/h{hh[-1]}_skill", -9e9))
            print(f"\nBEST on H{hh[-1]} rollout skill: lr={b['lr']:g} "
                  f"(H{hh[-1]} {b.get(f'roll/h{hh[-1]}_skill', float('nan')):+.4f}, 1-step {b['skill']:+.4f})")
            print("Rollout skill is the criterion, NOT loss: val loss fell 1.8x while the rollout got worse "
                  "in the previous run (AGENTS.md 8).")
        print("=" * 100)
        return

    wb = None
    if a.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, group=a.wandb_group, name=a.run_name, config=vars(a),
                            reinit=True)
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled: {e}")

    cdir = Path(a.ckpt_dir) if a.ckpt_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    hist, best, start_ep = [], {"werr": float("inf")}, 0
    if a.resume and Path(a.resume).exists():
        rk = torch.load(a.resume, weights_only=False, map_location="cpu")
        # FULL CLOSURE changed the head widths (new velocity / camera / floor / bg-bit outputs, and HP became a
        # bounded delta class), so a pre-closure checkpoint cannot supply them. Load every tensor whose shape
        # still matches -- the whole trunk -- and re-initialise the heads. Silently skipping a shape mismatch
        # would be the kind of thing that looks like a resume and is not, so it is reported.
        _have = net.state_dict()
        _ok = {k: v for k, v in rk["model"].items() if k in _have and _have[k].shape == v.shape}
        _skip = [k for k in rk["model"] if k not in _ok]
        net.load_state_dict(_ok, strict=False)
        if _skip:
            print(f"  partial resume: loaded {len(_ok)}/{len(rk['model'])} tensors; RE-INITIALISED {_skip} "
                  f"(shape changed by full closure). Optimizer state is therefore NOT reused.")
        start_ep = int(rk.get("epoch", 0))
        # A fresh Adam on a converged model destroys it: the first steps run with zero second moments, so the
        # effective step is ~lr on every parameter regardless of curvature. This is exactly what happened to
        # the 100-epoch run -- dpos went 1.048 -> 10.572 across a resume at epoch 5 and the "degrades with
        # training" conclusion drawn from it was an artefact of the resume, not of the model. So it is a hard
        # error, not a warning: a warning in a fleet launch scrolls past 32 ranks of output unread.
        if "opt" in rk and not _skip:
            opt.load_state_dict(rk["opt"])
            if sched is not None and rk.get("sched") is not None:
                sched.load_state_dict(rk["sched"])
            elif sched is not None:
                sys.exit(f"--sched {a.sched} but {a.resume} carries no scheduler state: resuming would restart "
                         f"the schedule from warmup on a trained model. Same class of bug as the missing "
                         f"optimizer state.")
            hist = list(rk.get("hist", []))
            best = rk.get("best", best)
            print(f"resumed from {a.resume} at epoch {start_ep} (optimizer state restored, "
                  f"{len(hist)} epochs of history, best werr {best.get('werr', float('inf')):.3f})")
        elif _skip or a.resume_fresh_opt:
            print(f"resumed from {a.resume} at epoch {start_ep} WITHOUT optimizer state (--resume-fresh-opt): "
                  f"expect the first epochs to move backwards; do not compare them to the pre-resume curve")
        else:
            sys.exit(f"{a.resume} has no optimizer state, so continuing from it would start a fresh Adam on a "
                     f"trained model and wreck it (this produced the bogus 1.048 -> 10.572 jump in the "
                     f"100-epoch run). Either train from scratch, or pass --resume-fresh-opt if you accept "
                     f"that the run is discontinuous at epoch {start_ep}.")
    for ep in range(start_ep, a.epochs):
        net.train()
        perm = rng.permutation(tr)
        t0, tot, gn, gmax, nb = time.time(), 0.0, 0.0, 0.0, 0
        sums, dsums = {}, {}
        nsteps = max(1, (len(perm) - a.batch) // a.batch)
        if a.steps_per_epoch:
            nsteps = min(nsteps, a.steps_per_epoch)
        for _s in range(nsteps):
            H = int(rng.choice(H_VALS, p=H_P)) if len(H_VALS) > 1 else int(H_VALS[0])
            sel = starts_for(H, a.batch) if H > 1 else perm[(_s * a.batch) % max(len(perm) - a.batch, 1):][:a.batch]
            if sel is None or len(sel) < a.batch:
                continue
            i = torch.from_numpy(np.asarray(sel)).to(dev)
            opt.zero_grad()
            if H > 1:
                # loss_ar backwards internally, once per BPTT window, so memory stays bounded at any depth
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    lossv, parts, _ = loss_ar(i, H, step_weights(H))
                loss = torch.as_tensor(lossv, device=dev)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss, parts, _ = loss_fn(i)
                loss.backward()
            dsums[f"d{H}/n"] = dsums.get(f"d{H}/n", 0) + 1
            g = float(nn.utils.clip_grad_norm_(net.parameters(), a.clip))
            opt.step()
            tot += float(loss.detach()); gn += g; gmax = max(gmax, g); nb += 1
            for _k, _v in parts.items():
                _hd = _k.split("/")[-1]
                _g = "link" if _hd.startswith("l_") else "actor" if _hd.startswith("a_") else "other"
                # LOG-SPACED STEP BINS. Keying by raw step number gives up to 1800 columns per head, which
                # bloats metrics.jsonl into six figures of keys and makes any table unreadable. 12 bins carry
                # the same shape.
                _st = int(_k.split("/")[0][1:]) if "/" in _k and _k[0] == "h" else 1
                _bin = 1 << max(0, (_st - 1).bit_length() - 1) if _st > 1 else 1
                _f = float(_v.detach()) if torch.is_tensor(_v) else float(_v)
                _kk = f"s{_bin}/{_hd}"
                sums[_kk] = sums.get(_kk, 0.0) + _f
                dsums[f"d{H}/{_g}"] = dsums.get(f"d{H}/{_g}", 0.0) + _f
            if not np.isfinite(g):
                print(f"  !! non-finite gradient at step {nb}"); break
        net.eval()
        with torch.no_grad():
            parts, werr, pers, dcell_acc, hp_acc, hp_rec = eval_pos(tu[:8192])
        # PER-ENTITY and PER-STEP aggregates, so the table can show where the loss actually lives:
        # link heads are l_*, actor heads are a_*, and with --train-h > 1 each rollout step h has its own set.
        agg = {k: v / max(nb, 1) for k, v in sums.items()}
        ent = {}
        for k, v in agg.items():
            step, _, head = k.partition("/") if "/" in k else ("h1", "/", k)
            grp = "link" if head.startswith("l_") else "actor" if head.startswith("a_") else "other"
            ent[f"{step}/{grp}"] = ent.get(f"{step}/{grp}", 0.0) + v
            ent[f"tot/{grp}"] = ent.get(f"tot/{grp}", 0.0) + v
        dep = {}
        for k, v in dsums.items():
            if k.endswith("/n"):
                dep[k] = v
                continue
            n_d = dsums.get(k.split("/")[0] + "/n", 1)
            dep[k] = v / max(n_d, 1)
        row = {"epoch": ep + 1, "lr": opt.param_groups[0]["lr"], **agg, **ent, **dep,
               "train": tot / nb, "grad_mean": gn / nb, "grad_max": gmax,
               "dpos_world": werr, "persistence_world": pers, "skill": 1 - werr / max(pers, 1e-9),
               "hp_acc": hp_acc, "hp_loss_recall": hp_rec, "dcell_acc": dcell_acc,
               **{k: float(v) for k, v in parts.items()}}
        with torch.no_grad():                       # held-out VALIDATION loss on the test slots
            iv = torch.from_numpy(te[:8192]).to(dev)
            vloss, vparts, _ = loss_fn(iv)
        row["val_loss"] = float(vloss)
        row.update({f"val/{k}": float(v) for k, v in vparts.items()})
        if (ep + 1) % a.rollout_every == 0 or ep == a.epochs - 1:
            _deep = [h for h in a.rollout_h if h <= int(_runlen[tu].max())]
            _pool = tu[_runlen[tu] >= max(_deep)][:2048] if _deep else tu[:2048]
            _tr0 = time.time()
            if len(_pool) >= 256:
                row.update(autoregressive(_pool, _deep))
            # The AR eval is the expensive MEASUREMENT (H=1800 is 1800 sequential forwards that cannot be
            # batched over h), so its wall cost is logged rather than hidden inside the epoch time: it is the
            # number that decides --rollout-every.
            row["roll/secs"] = time.time() - _tr0
            print(f"    rollout ({row['roll/secs']:.0f}s, n={len(_pool)}): " + "  ".join(
                f"H{h} {row[f'roll/h{h}_err']:.2f} vs persistence {row[f'roll/h{h}_persistence']:.2f} "
                f"(skill {row[f'roll/h{h}_skill']:+.3f})" for h in a.rollout_h if f"roll/h{h}_err" in row))
        if wb is not None:
            wb.log({"epoch": ep + 1, "train_loss": row["train"], "val_loss": row["val_loss"],
                    "dpos_world": werr, "persistence_world": pers, "skill": row["skill"],
                    "dcell_acc": dcell_acc, "hp_loss_recall": hp_rec, "grad_mean": row["grad_mean"],
                    **{k: v for k, v in row.items() if k.startswith(("roll/", "val/"))}})
        # BEST-EPOCH SELECTION on held-out position error. The loss falls smoothly while dpos/dcell oscillate
        # between a good state (dcell 0.97, skill ~-0.1) and a bad one (0.76, -10), so the LAST epoch is a coin
        # flip -- and reading the last epoch is how several "degrades with training" conclusions got made this
        # session. Keep the best weights, and report best alongside last.
        if werr < best["werr"]:
            best = {**row, "werr": werr, "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
        hist.append(row)
        if cdir:
            with open(cdir / "metrics.jsonl", "a") as fh:
                fh.write(json.dumps({k: v for k, v in row.items()}) + "\n")
            if (ep + 1) % a.ckpt_every == 0:
                # `opt` is NOT optional. Adam's per-parameter second moments ARE the training state; a
                # checkpoint without them cannot be continued, only restarted. See --resume below.
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict() if sched is not None else None, "epoch": ep + 1,
                            "metrics": row, "hist": hist, "best": best,
                            "scales": sc, "grid": grid.state_dict(), "n_actions": NA, "config": vars(a)},
                           cdir / f"ck_{ep+1:04d}.pt")
        if link is not None:
            # FedAvg once per epoch: each rank has trained on its own shard, so this is data parallelism over
            # the whole fleet's dataset. Straggler tolerant -- a slow rank never holds the others back.
            info = link.average_weights(net, ep + 1, aggregator=a.fleet_aggregator)
            link.heartbeat({"epoch": ep + 1, "dpos_world": werr, "skill": row["skill"],
                            "dcell_acc": dcell_acc})
            print(f"    fleet: averaged over {info.get('n', '?')} ranks (from update {info.get('src', '?')})")
        print(f"  ep {ep+1:>3}/{a.epochs} loss {tot/nb:.4f} | grad mean {gn/nb:.3f} max {gmax:.3f} | "
              f"dpos {werr:.3f} vs persistence {pers:.3f} world units (skill {1-werr/max(pers,1e-9):+.3f}) | "
              f"dcell {dcell_acc:.4f} | hp rec {hp_rec:.3f}  ({time.time()-t0:.0f}s)")
        print(f"    train_loss {row['train']:.4f}  val_loss {row['val_loss']:.4f}"
              + (f"  lr {opt.param_groups[0]['lr']:.2e}" if sched is not None else ""))
        if sched is not None:
            sched.step()
    if "epoch" not in best:
        sys.exit(f"no epoch ran: --epochs {a.epochs} but the resume started at {start_ep}. Raise --epochs.")
    print(f"\nBEST epoch {best['epoch']}: dpos {best['werr']:.3f} vs persistence {best['persistence_world']:.3f} "
          f"(skill {best['skill']:+.3f}) dcell {best['dcell_acc']:.4f} | LAST epoch {hist[-1]['epoch']}: "
          f"dpos {hist[-1]['dpos_world']:.3f} (skill {hist[-1]['skill']:+.3f}) dcell {hist[-1]['dcell_acc']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best.pop("state", net.state_dict()), "last_model": net.state_dict(),
                "best": {k: v for k, v in best.items() if k != "state"}, "scales": sc, "split": split, "n_actions": NA,
                "config": vars(a), "history": hist}, a.out)
    # ---- PER-FIELD EVAL, IDENTICAL IN BOTH ARMS -------------------------------------------------------
    # The training losses of gated heads are averaged over CHANGED elements only, so they are not comparable
    # across a --gate and a no-gate run. This scores the eight gated fields the same way either way: MAE of the
    # predicted delta against the true delta, with the gate APPLIED exactly as it would be at deployment, next
    # to a do-nothing control (predict zero). Reported over all frames and over changed frames separately,
    # because a gate is supposed to win on the first and could lose on the second.
    @torch.no_grad()
    def eval_heads(sel, chunk=2048):
        net.eval()
        FL = [("l_y", 1), ("l_yaw", 2), ("l_floor", 1)]
        FA = [("a_doff", 2), ("a_yaw", 2), ("a_shape", 2), ("a_timer", 1), ("a_vel", 3)]
        acc = {n: np.zeros(4) for n, _ in FL + FA}      # err_all, pers_all, err_chg, n_chg  (+ n_all in [4])
        nall = {n: 0.0 for n, _ in FL + FA}
        for st in range(0, len(sel), chunk):
            i = torch.from_numpy(np.asarray(sel[st:st + chunk])).to(dev)
            if len(i) < 8:
                continue
            T = gt(i)
            ol, oa = net(T, ag(i))
            (dcl, dof), dyv, dyw, hpd, bits, scn, dvel, dcam, dfl, bg = split_link(ol, i, None)
            (rdcl, rdof), rhpd, rpr, rst, ardy, adyw, adsh, adti, advl = split_actor(oa, i, None)
            if a.gate:
                gl = (ol[:, -len(GATE_L):] > 0).float(); gaa = (oa[..., -len(GATE_A):] > 0).float()
                dyw = dyw * gl[:, 0:1]
                rdof, adyw = rdof * gaa[..., 0:1], adyw * gaa[..., 1:2]
                adsh, adti = adsh * gaa[..., 2:3], adti * gaa[..., 3:4]
            j, jn = i, (i + 1).clamp(max=N - 1)
            Ln, Lx = tt["link"][j].float(), tt["link"][jn].float()
            An, Ax = tt["actors"][j].float(), tt["actors"][jn].float()
            m = yg("pres_now", j)
            aoff_t = tt["actor_off"][jn].float() - tt["actor_off"][j].float()
            pred = {"l_y": dyv.squeeze(-1)[:, None], "l_yaw": dyw, "l_floor": dfl.squeeze(-1)[:, None],
                    "a_doff": rdof, "a_yaw": adyw, "a_shape": adsh,
                    "a_timer": adti, "a_vel": advl}
            tgt = {"l_y": (Lx[:, L_RELY] - Ln[:, L_RELY])[:, None], "l_yaw": Lx[:, L_YAW] - Ln[:, L_YAW],
                   "l_floor": (Lx[:, L_FLOOR] - Ln[:, L_FLOOR])[:, None], "a_doff": aoff_t,
                   "a_yaw": Ax[:, :, A_YAW] - An[:, :, A_YAW], "a_shape": Ax[:, :, A_SHAPE] - An[:, :, A_SHAPE],
                   "a_timer": (Ax[:, :, A_TIMER] - An[:, :, A_TIMER])[..., None],
                   "a_vel": Ax[:, :, A_VEL] - An[:, :, A_VEL]}
            for n, _ in FL + FA:
                pr, tg = pred[n], tgt[n]
                if tg.dim() == 3:                        # actor field: mask absent rows, as training does
                    e = (pr - tg).abs().mean(-1) * m; p0 = tg.abs().mean(-1) * m
                    ch = (tg.abs().amax(-1) > 1e-6).float() * m
                    cnt = m.sum()
                else:
                    e = (pr - tg).abs().mean(-1); p0 = tg.abs().mean(-1)
                    ch = (tg.abs().amax(-1) > 1e-6).float(); cnt = torch.tensor(float(len(i)), device=dev)
                acc[n] += np.array([float(e.sum()), float(p0.sum()), float((e * ch).sum()), float(ch.sum())])
                nall[n] += float(cnt)
        net.train()
        print(f"\nPER-FIELD EVAL (identical metric in both arms, gate {'ON' if a.gate else 'off'}), "
              f"{len(sel):,} held-out frames")
        print(f"  {'field':<10}{'changes':>9}{'MAE all':>10}{'do-nothing':>12}{'skill':>8}{'MAE|changed':>13}")
        rows = {}
        for n, _ in FL + FA:
            ea, pa, ec, nc = acc[n]; na = max(nall[n], 1)
            sk = 1 - (ea / max(pa, 1e-12))
            rows[n] = dict(chg=nc / na, mae=ea / na, pers=pa / na, skill=sk, mae_chg=ec / max(nc, 1))
            print(f"  {n:<10}{nc/na*100:>8.2f}%{ea/na:>10.5f}{pa/na:>12.5f}{sk:>+8.3f}{ec/max(nc,1):>13.5f}")
        print("  GATED-FIELD MEAN SKILL: " f"{np.mean([r['skill'] for r in rows.values()]):+.4f}")
        return rows

    eval_heads(tu[:16384])

    print(f"\nper-head final losses: " + json.dumps({k: round(hist[-1][k], 5) for k in
          ("l_dcell", "l_doff", "l_y", "l_yaw", "l_hp", "l_bits", "l_scene", "a_dcell", "a_doff", "a_hp",
           "a_pres", "a_state")}))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
