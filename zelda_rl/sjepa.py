"""Structured-state JEPA: V-JEPA's masked-prediction pretext, on the 363-d game state.

WHY THIS EXISTS, and what is different about it.

`LATENT_WORLD_MODEL.md` §1-§30, `JEPA_PLAN.md` and `WM_EXPERIMENT_REGIME.md` are four attempts at a latent
world model and all four optimise the SAME pretext: predict z_{t+1} from (z_t, a_t). Every one of them lost to
the identity, and the mechanism is recorded: the latent moves 0.023-0.05 of a random pair per step, so
"copy z_t" is a near-optimal answer. The VQ arm proved it outright -- skill +0.000 with tune error
bit-identical to the identity, its predictor having provably learned to copy.

The JEPA papers do not pose that task. I-JEPA (2301.08243) and V-JEPA (2404.08471) mask part of the INPUT and
predict the target encoder's representation of the masked part from the visible remainder. There is no
persistence shortcut, because there is no previous latent to carry forward. That pretext was never run here.

PHASE 1 (this module + scripts/train_sjepa.py) is that pretext and nothing else: no action, no temporal
prediction, no reward. PHASE 2 adds action conditioning, and is not started until phase 1 clears its controls.

WHAT IT IS MASKING. The substrate is `zelda_rl.state_layout`'s flat state -- Link's 43 raw engine fields plus
16 actor rows x 20 -- NOT the 483/1033-d PPO `obs`. That is deliberate. `obs` is a featuriser's output and is
full of derived redundancy (`goal_delta` = goal_pos - link_pos, `nearest_target_dist` = |rel_pos|, and
`prev_action_*`, the §5 control echo). Masking a derived field while its inputs stay visible scores arithmetic,
which is exactly the free-signal failure `AGENTS.md` §1 was written about. The structured state has its own
derivable pairs -- `speed` = |vel|, pos_{t} from pos_{t-1} + vel -- so `analytic_floor` below measures them
rather than trusting that they do not matter.

THE HONEST RISK, stated before any number exists. `scripts/train_jepa_pure.py` trained an encoder by a
predictive objective alone and the result was that it is WORSE THAN RANDOM INITIALISATION on 7 of 10 reward
fields (link_pos 35.1 world units against an untrained encoder's 29.0, where reconstruction gets 9-11).
Prediction alone has no reason to keep quantities that are hard to predict. A masked pretext differs in that
the thing being predicted IS the state rather than its future, but that is an argument, not a measurement, and
the falsification bar is therefore the same one: if the frozen probes do not beat the untrained-encoder
control, this is `pure.pt` again and phase 1 stops.

NOTHING HERE TRAINS THROUGH A PROBE. The probes live in scripts/probe_sjepa.py, are fitted on a frozen
encoder, and never return a gradient to it (DINO-WM 2411.04983 reports decoder gradients hurt, and keeping the
readout post-hoc is what makes "the objective discarded this" a statement about the objective).
"""
from __future__ import annotations

import numpy as np

from zelda_rl.state_layout import (ENEMY_FIELDS, N_ROWS, ROW_DIM, STATE_DIM_V3, link_fields)

# ---- field groups ------------------------------------------------------------------------------------
# The masking axis "mask one KIND of thing everywhere" needs semantic groups. They are declared by NAME and
# the union is ASSERTED against state_layout's own lists below, so a renamed or added field fails loudly here
# instead of silently dropping out of the mask vocabulary. AGENTS §3: derive it from the definition, assert it.
GROUPS: dict[str, tuple[str, ...]] = {
    "pos":       ("pos_x", "pos_y", "pos_z"),
    "vel":       ("vel_x", "vel_y", "vel_z", "speed"),
    "orient":    ("yaw", "shape_yaw", "parallel_yaw", "wall_yaw", "yaw_to_player",
                  "cam_yaw", "cam_pitch"),
    "health":    ("hp", "hp_cap"),
    "actstate":  ("action_fn_rel", "melee_state", "melee_anim", "fn_rel", "state_idx", "knockback_type"),
    "contact":   ("bg_flags", "floor_height", "dist_interact_wall", "y_dist_to_ledge",
                  "fall_distance", "floor_pitch"),
    "identity":  ("present", "id", "cat", "params"),
    "timers":    ("timer", "ztarget_timer", "door_timer"),
    "targeting": ("targeted_row", "arrow_row"),
    "scene":     ("scene", "room"),
    "flags":     ("sf1", "sf2", "sf3"),
    "camera":    ("cam_eye_x", "cam_eye_y", "cam_eye_z"),
    "inventory": ("held_item", "seeds", "nuts", "grass"),
    "focus":     ("focus_x", "focus_y", "focus_z"),
    "door":      ("door_type", "door_direction"),
}

# Fields read as DISCRETE (embedding table on the input side, cross-entropy on the output side). Everything
# else is continuous. `bg_flags`/`sf*` are bitfields: discrete as an id, and the probe reads single bits.
DISCRETE = frozenset((
    "scene", "room", "sf1", "sf2", "sf3", "targeted_row", "arrow_row", "held_item", "bg_flags",
    "action_fn_rel", "melee_state", "melee_anim", "door_type", "door_direction", "knockback_type",
    "present", "id", "fn_rel", "state_idx", "cat", "params",
))


def _assert_groups(width: int) -> None:
    """Every field of this state version belongs to exactly one group. Fails loudly, by construction.

    Two directions, and they are deliberately asymmetric. For the width actually in use, EVERY field must be
    grouped -- an ungrouped field would be silently unmaskable and invisible to the probe suite. In the other
    direction GROUPS is checked against the WIDEST known version (v3), not against `width`, because v1 and v2
    states legitimately lack the body/collision/camera blocks; checking against `width` would make a v1 load
    fail on 24 fields that exist, just not there. A typo still fails, because it would be absent from v3 too.
    """
    names = set(link_fields(width)) | set(ENEMY_FIELDS)
    widest = set(link_fields(STATE_DIM_V3)) | set(ENEMY_FIELDS)
    seen: dict[str, str] = {}
    for g, fs in GROUPS.items():
        for f in fs:
            if f in seen:
                raise AssertionError(f"field {f!r} is in two groups: {seen[f]} and {g}")
            seen[f] = g
    missing = names - set(seen)
    if missing:
        raise AssertionError(f"state width {width}: {len(missing)} field(s) in no GROUP: {sorted(missing)}")
    typos = set(seen) - widest
    if typos:
        raise AssertionError(f"GROUPS names that are in NO state version: {sorted(typos)}")
    bad = DISCRETE - widest
    if bad:
        raise AssertionError(f"DISCRETE names that are in NO state version: {sorted(bad)}")


class Schema:
    """The (entity, field) grid a state vector unpacks into, with every index derived, not written down.

    Entities are 1 Link + N_ROWS actors. The grid is RAGGED -- Link has 43 fields, an actor row has 20 -- so
    it is stored as a per-entity list of (field name, column in the flat state) and a boolean validity grid,
    rather than as a dense (entity, field) array that would silently treat Link's 44th column as an actor's.
    """

    def __init__(self, width: int):
        _assert_groups(width)
        self.width = width
        self.lf = list(link_fields(width))
        self.n_link = len(self.lf)
        self.n_ent = 1 + N_ROWS
        # field vocabulary: every DISTINCT field name across Link and the rows, in a stable order
        self.fields = list(dict.fromkeys(self.lf + list(ENEMY_FIELDS)))
        self.fidx = {f: i for i, f in enumerate(self.fields)}
        self.n_field = len(self.fields)
        # cols[e][f] = column in the flat state, or -1 where entity e has no field f
        self.cols = np.full((self.n_ent, self.n_field), -1, dtype=np.int64)
        for j, f in enumerate(self.lf):
            self.cols[0, self.fidx[f]] = j
        for r in range(N_ROWS):
            base = self.n_link + r * ROW_DIM
            for j, f in enumerate(ENEMY_FIELDS):
                self.cols[1 + r, self.fidx[f]] = base + j
        self.valid = self.cols >= 0
        assert int(self.valid.sum()) == width, (
            f"grid covers {int(self.valid.sum())} cells for a {width}-wide state -- the layout and the "
            f"grid disagree, which is the drift state_layout.py exists to prevent")
        self.group_of = np.array([self._group(f) for f in self.fields])
        self.is_discrete = np.array([f in DISCRETE for f in self.fields])
        self.groups = sorted(GROUPS)

    @staticmethod
    def _group(f: str) -> str:
        for g, fs in GROUPS.items():
            if f in fs:
                return g
        raise AssertionError(f"ungrouped field {f!r}")        # unreachable: _assert_groups ran

    def group_cells(self, g: str) -> np.ndarray:
        """(n_ent, n_field) boolean: the valid cells belonging to group `g`."""
        return self.valid & (self.group_of == g)[None, :]

    def unpack(self, state: np.ndarray) -> np.ndarray:
        """(..., width) -> (..., n_ent, n_field), zero where a cell is invalid."""
        out = np.zeros(state.shape[:-1] + (self.n_ent, self.n_field), dtype=state.dtype)
        e, f = np.nonzero(self.valid)
        out[..., e, f] = state[..., self.cols[e, f]]
        return out


# ---- masking -----------------------------------------------------------------------------------------
# V-JEPA's multi-block mask is the method, not a detail: it samples several blocks per clip and masks their
# UNION across the whole time axis ("tube" masking), so the model cannot solve a target by copying the same
# patch from a neighbouring frame. The structured analogue of a spatio-temporal patch grid is
# (time x entity x field), and the four block shapes below are the four ways that grid can be cut:
#
#   entity_tube  one whole actor (or Link) vanishes for the clip     -> "where is Baba 3, from the rest"
#   group_tube   one KIND of field vanishes for every entity         -> "reconstruct all velocities"
#   time_block   a contiguous span of frames vanishes entirely       -> interpolation / extrapolation
#   cell         scattered individual cells                          -> the easy case, kept as a floor
#
# `time_block` is the only one of the four that a neighbouring frame can help with, and that is the point of
# keeping the other three as tubes. A per-frame random mask -- the obvious first implementation -- is trivially
# solvable by temporal interpolation on a 60 Hz game where most fields barely move between frames, and it
# would have produced an excellent-looking loss that measured nothing.
BLOCKS = ("entity_tube", "group_tube", "time_block", "cell")


class MaskSampler:
    """Samples (T, n_ent, n_field) target masks over a Schema's valid cells.

    `ratio` is the TARGET fraction of valid cells, and it is approached by adding blocks until it is reached
    rather than by solving for block sizes, because the blocks overlap. The achieved ratio is returned so the
    trainer can log it: a requested ratio and an achieved ratio are different numbers and only one of them is
    evidence (AGENTS §36 -- every number a program prints about its own output is a claim).

    Both sides are asserted non-empty. A mask with no targets makes the loss `nan` over an empty reduction; a
    mask with no context makes the encoder read nothing. Neither announces itself, and the n==0 edge case is
    exactly what cost 17 of 22 sweep arms in §36 row 1, so it is an assertion and a test, not a comment.
    """

    def __init__(self, schema: Schema, *, ratio: float = 0.75, ratio_max: float | None = None,
                 weights: dict[str, float] | None = None, rng: np.random.Generator | None = None):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0,1), got {ratio}")
        self.s = schema
        self.ratio = ratio
        # Blocks OVERLAP and a single block can be large (one time_block at T=1 covers the whole grid), so the
        # loop that adds blocks until it reaches `ratio` can overshoot badly. Unclamped, this sampler produced
        # draws with ONE context cell -- nominally a 0.997 mask, actually noise -- and the non-empty repair
        # below would have hidden them behind a passing assertion. Surplus target cells are released at random
        # above `ratio_max`; that punches holes in a tube, which is why the cap is loose rather than exact.
        self.ratio_max = min(0.95, ratio + 0.10) if ratio_max is None else ratio_max
        if not ratio <= self.ratio_max < 1.0:
            raise ValueError(f"need ratio <= ratio_max < 1, got {ratio} / {self.ratio_max}")
        # A PARTIAL weights dict is rejected rather than merged over the defaults. Merging was the first
        # implementation and it made the "covers exactly BLOCKS" check unreachable -- the merged dict always
        # covered all four, so `weights={"entity_tube": 1}` silently kept 0.2/0.2/0.3 of the others and ran a
        # different experiment than the one asked for. A check that cannot fire is decoration (AGENTS 36).
        w = ({"entity_tube": 0.3, "group_tube": 0.3, "time_block": 0.2, "cell": 0.2}
             if weights is None else dict(weights))
        if set(w) != set(BLOCKS):
            raise ValueError(f"weights must name exactly {BLOCKS}, got {sorted(w)} -- pass all four "
                             f"(use 0.0 to disable one) so an arm's mask mixture is explicit in its config")
        tot = sum(w.values())
        if tot <= 0:
            raise ValueError("mask block weights sum to zero")
        self.w = np.array([w[b] / tot for b in BLOCKS])
        self.rng = rng or np.random.default_rng(0)
        # per-entity valid cells, precomputed: row 0 is Link (43 cells), rows 1.. are actors (20 each)
        self._ent_cells = schema.valid
        self._n_valid = int(schema.valid.sum())

    def sample(self, T: int) -> tuple[np.ndarray, dict]:
        """-> (target mask (T, n_ent, n_field) bool, info). Context is `schema.valid & ~mask`."""
        s = self.s
        valid_t = np.broadcast_to(s.valid, (T, s.n_ent, s.n_field))
        tgt = np.zeros((T, s.n_ent, s.n_field), dtype=bool)
        want = self.ratio * self._n_valid * T
        used: list[str] = []
        for _ in range(512):                                  # bounded: never spin on an unreachable ratio
            if tgt.sum() >= want:
                break
            b = BLOCKS[self.rng.choice(len(BLOCKS), p=self.w)]
            used.append(b)
            if b == "entity_tube":
                e = self.rng.integers(s.n_ent)
                tgt[:, e, :] |= s.valid[e]
            elif b == "group_tube":
                g = s.groups[self.rng.integers(len(s.groups))]
                tgt |= s.group_cells(g)[None]
            elif b == "time_block":
                span = max(1, int(self.rng.integers(1, max(2, T // 2 + 1))))
                t0 = int(self.rng.integers(0, max(1, T - span + 1)))
                tgt[t0:t0 + span] |= s.valid[None]
            else:                                             # cell
                k = max(1, int(0.05 * self._n_valid * T))
                flat = np.flatnonzero(valid_t.reshape(-1))
                pick = self.rng.choice(flat, size=min(k, flat.size), replace=False)
                f = tgt.reshape(-1)
                f[pick] = True
        tgt &= valid_t                                        # never target a cell that does not exist
        n_valid = int(valid_t.sum())
        surplus = int(tgt.sum()) - int(self.ratio_max * n_valid)
        if surplus > 0:
            on = np.flatnonzero(tgt.reshape(-1))
            tgt.reshape(-1)[self.rng.choice(on, size=surplus, replace=False)] = False
        ctx = valid_t & ~tgt
        # Both sides non-empty. A full-coverage draw (e.g. one time_block spanning T at T==1) is possible and
        # is corrected by freeing one random target cell back to the context, which keeps the encoder fed.
        if not ctx.any():
            idx = self.rng.choice(np.flatnonzero(tgt.reshape(-1)))
            tgt.reshape(-1)[idx] = False
            ctx = valid_t & ~tgt
        if not tgt.any():
            idx = self.rng.choice(np.flatnonzero(ctx.reshape(-1)))
            tgt.reshape(-1)[idx] = True
            ctx = valid_t & ~tgt
        assert ctx.any() and tgt.any(), "mask left one side empty after repair"
        assert not (ctx & tgt).any(), "context and target overlap"
        assert ((ctx | tgt) == valid_t).all(), "context + target does not cover the valid cells"
        n = n_valid
        return tgt, {"ratio_achieved": int(tgt.sum()) / n, "ratio_requested": self.ratio,
                     "blocks": used, "n_target": int(tgt.sum()), "n_context": int(ctx.sum()),
                     "n_valid": n}


# ---- preprocessing -----------------------------------------------------------------------------------
class Prep:
    """Per-field normalisation and discrete vocabularies, fitted ONCE on the train split.

    Two reasons this is a fitted artefact written to disk rather than computed per run:

    1. `MAP.md` records that fitting per-pod under FedAvg averages weights across ranks whose inputs mean
       different things -- silently wrong, and it presents as a training failure. `train_wm_tokens.py` refuses
       to run without `--prep` for exactly this reason and so does the sjepa trainer.
    2. A DISCRETE field is embedded by value, so its vocabulary has to be stable across runs or checkpoint
       `id`-embedding row 7 means a different actor in the probe than it did in training.

    Discrete fields get an embedding table, not an affine on the scalar: actor `id` 5 is not "between" 4 and 6,
    and `state_idx` is a behaviour state machine. Treating those as continuous is what the semantic-recognition
    probe would then fail to read, and the failure would look like the representation's fault.
    """

    MAX_VOCAB = 512         # a field with more distinct values than this is treated as continuous, and said so

    def __init__(self, schema: Schema, mean, std, vocab, cont_fields, disc_fields, notes=None):
        self.s = schema
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.vocab = {int(k): {float(v): i for i, v in enumerate(vs)} for k, vs in vocab.items()}
        self.cont = np.asarray(cont_fields, bool)
        self.disc = np.asarray(disc_fields, bool)
        self.notes = list(notes or [])
        assert self.mean.shape == self.std.shape == (schema.n_field,)
        assert not (self.cont & self.disc).any(), "a field cannot be both continuous and discrete"
        assert (self.cont | self.disc).all(), "every field must be one or the other"

    @classmethod
    def fit(cls, schema: Schema, states: np.ndarray, *, verbose: bool = True) -> "Prep":
        """states: (N, width) raw. Statistics are over VALID cells only, per field, pooled over entities."""
        g = schema.unpack(states)                                  # (N, E, F)
        mean = np.zeros(schema.n_field, np.float32)
        std = np.ones(schema.n_field, np.float32)
        vocab: dict[int, list[float]] = {}
        cont = np.zeros(schema.n_field, bool)
        disc = np.zeros(schema.n_field, bool)
        notes: list[str] = []
        for f in range(schema.n_field):
            ents = np.flatnonzero(schema.valid[:, f])
            v = g[:, ents, f].reshape(-1)
            name = schema.fields[f]
            if schema.is_discrete[f]:
                u = np.unique(v)
                if u.size <= cls.MAX_VOCAB:
                    vocab[f] = [float(x) for x in u]
                    disc[f] = True
                    continue
                notes.append(f"{name}: {u.size} distinct values > MAX_VOCAB {cls.MAX_VOCAB}, "
                             f"treated as CONTINUOUS")
            cont[f] = True
            mean[f] = float(v.mean())
            s = float(v.std())
            std[f] = s if s > 1e-6 else 1.0
            if s <= 1e-6:
                notes.append(f"{name}: std {s:.2e} over {v.size} cells -- constant in this split")
        p = cls(schema, mean, std, {str(k): v for k, v in vocab.items()}, cont, disc, notes)
        if verbose:
            print(f"prep: {int(cont.sum())} continuous + {int(disc.sum())} discrete fields "
                  f"({sum(len(x) for x in vocab.values())} vocab entries total)", flush=True)
            for n in notes:
                print(f"  note: {n}", flush=True)
        return p

    def vocab_sizes(self) -> np.ndarray:
        return np.array([len(self.vocab.get(f, ())) for f in range(self.s.n_field)], dtype=np.int64)

    def encode(self, states: np.ndarray):
        """raw (..., width) -> (cont values z-scored, discrete value INDICES), both (..., E, F).

        INVALID cells are not out-of-vocabulary. The first version counted them: `unpack` writes 0.0 into a
        cell the entity does not have (Link has no `id`), `encode` then looked 0.0 up in `id`'s vocabulary,
        missed, and reported 192 OOV values on the very data the vocabulary was fitted from. An instrument
        that reports a fault on correct input gets ignored within a day (AGENTS 36), so the OOV counter is
        restricted to valid cells -- where it is now exactly 0 on the fit split, which is what makes a
        non-zero count on a new split evidence.

        Vectorised with searchsorted over the sorted vocabulary rather than a per-element dict lookup: the
        loop version was ~2.6 M dict lookups per 1000 frames and dominated the data path.
        """
        g = self.s.unpack(states).astype(np.float32)
        x = (g - self.mean) / self.std
        x[..., ~self.cont] = 0.0
        idx = np.zeros(g.shape, dtype=np.int64)
        oov = 0
        for f in np.flatnonzero(self.disc):
            tbl = self.vocab[int(f)]
            keys = np.array(sorted(tbl), dtype=np.float32)          # vocabulary values, ascending
            order = np.array([tbl[k] for k in sorted(tbl)], dtype=np.int64)
            col = g[..., f]
            pos = np.searchsorted(keys, col)
            pos_c = np.clip(pos, 0, keys.size - 1)
            hit = keys[pos_c] == col
            idx[..., f] = np.where(hit, order[pos_c], 0)
            ents = self.s.valid[:, f]                                # only real cells can be OOV
            oov += int((~hit)[..., ents].sum())
        self.last_oov = oov
        return x, idx

    def to_json(self) -> dict:
        return {"width": self.s.width, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "vocab": {str(k): [float(x) for x in sorted(v, key=v.get)] for k, v in self.vocab.items()},
                "cont": self.cont.tolist(), "disc": self.disc.tolist(), "notes": self.notes,
                "fields": self.s.fields}

    @classmethod
    def from_json(cls, d: dict) -> "Prep":
        s = Schema(int(d["width"]))
        if list(d["fields"]) != s.fields:
            raise ValueError("prep was fitted on a different field order than this Schema produces")
        return cls(s, d["mean"], d["std"], d["vocab"], d["cont"], d["disc"], d.get("notes"))


# ---- model -------------------------------------------------------------------------------------------
# Deliberately NOT reusing `scripts/train_wm_tokens.Attn`. That block is shared by the token WM and the latent
# JEPA, carries a per-head distance bias and an asymmetric query/context split, and MAP.md records that a
# change to it must be pushed to both consumers. The sjepa grid is (time x entity) with no distance matrix and
# no context pool, so it needs a plain block; editing Attn to serve a third shape is how the two existing
# models would acquire a silent change. A new block is cheaper than a shared one with three callers.
import torch                                                        # noqa: E402
import torch.nn as nn                                               # noqa: E402
import torch.nn.functional as F                                     # noqa: E402


class Block(nn.Module):
    """Pre-norm transformer block, full attention over the flattened (T*E) token grid."""

    def __init__(self, d: int, heads: int, mlp: int = 4, drop: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class StateTokeniser(nn.Module):
    """(cont, idx, cellmask) -> one token per (frame, entity).

    A token is the SUM of its field embeddings plus entity-type, slot and time embeddings. Summing rather
    than concatenating is what lets Link (43 fields) and an actor row (20) share one token width and one
    transformer, and it is what makes a per-field `[MASK]` vector a drop-in substitution: a masked cell
    contributes `mask_emb[f]` where it would otherwise contribute its value's embedding.

    TWO BUGS LIVED HERE AND BOTH WERE MEASURED, NOT SPOTTED. Recorded because the first one made the whole
    representation worthless while every health column looked fine.

    1. POSITION SWAMPED CONTENT 14x. `nn.Embedding` initialises to N(0,1), so type/slot/time each had L2
       norm ~sqrt(d) = 9.8 and summed to 17.2, while the field embeddings (`randn * 0.02`) contributed 1.20
       in total. Measured on a real batch: content/positional = 0.070 in norm, and the across-clip /
       across-position variance ratio was **0.013 -- identical to an UNTRAINED encoder's 0.016**. The
       representation was ~98.7% a function of token position, the masked-prediction loss fell to 0.030
       because a positional constant solves it, and predicting a DIFFERENT CLIP's target rep scored 0.0029,
       ten times better than the model. `collapse_stats` could not see any of it: it pools variance over
       (clip, frame, entity), so positional variance reads as informativeness (eff_rank 17.9/96,
       min_std 0.657 -- both healthy, both meaningless). Fixed by initialising the positional tables at the
       same 0.02 scale and LayerNorm-ing the token, so content and position enter on comparable footing.

    2. THE FIELD SUM WAS DIVIDED BY n, NOT sqrt(n). The previous version of this docstring claimed dividing
       by `n_valid_per_ent` keeps Link's token from being larger in norm than an actor's. That is wrong: a
       sum of n independent terms grows as sqrt(n), so dividing by n OVER-corrects and attenuates Link
       (43 fields) by sqrt(43)/43 = 1/6.6 against an actor's 1/4.5 -- Link's content entered 1.5x weaker
       than an actor's. This is exactly the asymmetry that later reads as "Link decodes worse than actors"
       and gets blamed on the objective.
    """

    def __init__(self, schema: Schema, prep: Prep, d: int = 128):
        super().__init__()
        self.s, self.p, self.d = schema, prep, d
        F_ = schema.n_field
        self.register_buffer("valid", torch.from_numpy(schema.valid))                 # (E, F)
        self.register_buffer("cont", torch.from_numpy(prep.cont))                     # (F,)
        self.register_buffer("disc", torch.from_numpy(prep.disc))
        self.cont_w = nn.Parameter(torch.randn(F_, d) * 0.02)       # per-field affine: value -> d
        self.cont_b = nn.Parameter(torch.zeros(F_, d))
        vs = prep.vocab_sizes()
        self.vocab_sizes = vs
        self.disc_emb = nn.ModuleList([nn.Embedding(max(1, int(n)), d) for n in vs])
        self.mask_emb = nn.Parameter(torch.randn(F_, d) * 0.02)     # per-FIELD mask token
        self.type_emb = nn.Embedding(2, d)                          # 0 = Link, 1 = actor
        self.slot_emb = nn.Embedding(schema.n_ent, d)
        self.time_emb = nn.Embedding(512, d)
        for t in (self.type_emb, self.slot_emb, self.time_emb):     # see bug 1 in the class docstring
            nn.init.normal_(t.weight, std=0.02)
        self.out_norm = nn.LayerNorm(d)                             # content and position on one footing
        tp = torch.zeros(schema.n_ent, dtype=torch.long); tp[1:] = 1
        self.register_buffer("types", tp)
        # sqrt(n), not n: see bug 2 in the class docstring.
        self.register_buffer("n_valid_per_ent",
                             torch.from_numpy(schema.valid.sum(1)).float().clamp(min=1).sqrt())

    def forward(self, cont, idx, cell_masked):
        """cont/idx (B,T,E,F); cell_masked (B,T,E,F) bool -> tokens (B,T,E,d)."""
        B, T, E, Fn = cont.shape
        v = self.valid[None, None]                                   # (1,1,E,F)
        keep = v & ~cell_masked                                      # cells whose VALUE is visible
        e = cont.unsqueeze(-1) * self.cont_w + self.cont_b           # (B,T,E,F,d) continuous
        e = e * self.cont[None, None, None, :, None]
        for f in torch.nonzero(self.disc, as_tuple=True)[0].tolist():
            e[..., f, :] = self.disc_emb[f](idx[..., f])
        e = e * keep.unsqueeze(-1)                                   # visible values only
        m = (v & cell_masked).unsqueeze(-1) * self.mask_emb[None, None, None]
        tok = (e + m).sum(-2) / self.n_valid_per_ent[None, None, :, None]
        t = torch.arange(T, device=cont.device)
        tok = tok + self.type_emb(self.types)[None, None] + self.slot_emb.weight[None, None] \
            + self.time_emb(t)[None, :, None]
        return self.out_norm(tok)


class SJepa(nn.Module):
    """V-JEPA, structured-state: context encoder, EMA target encoder, predictor over masked tokens.

    The loss is L1 between the predictor's output and the TARGET ENCODER's representation of the same tokens,
    both layer-normalised, with a stop-gradient on the target (I-JEPA 2301.08243 App. A; V-JEPA 2404.08471).
    The target encoder sees the FULL unmasked grid, which is the asymmetry that makes the task non-trivial and
    -- together with the EMA and the stop-gradient -- is what the papers name as the anti-collapse mechanism.

    This repo has already measured that this mechanism is sufficient HERE: `train_jepa_pure.py` reached
    effective rank 10.2-10.7 of 128 with no dead dimensions and a dead-stable norm, the least collapsed latent
    in the project. So collapse is monitored, not defended against with an extra term -- and `collapse_stats`
    prints it every epoch, because a collapse discovered in the final table has already cost the run.

    NOTE on what is NOT here: no action, no z_{t+1}, no reward. Phase 1 is the masked pretext alone. Phase 2
    adds action conditioning and is a separate decision (AGENTS 13: one authorisation, one experiment).
    """

    def __init__(self, schema: Schema, prep: Prep, *, d: int = 128, depth: int = 4, heads: int = 4,
                 pred_depth: int = 2, ema: float = 0.996):
        super().__init__()
        self.s, self.d, self.ema = schema, d, ema
        self.tok = StateTokeniser(schema, prep, d)
        self.enc = nn.Sequential(*[Block(d, heads) for _ in range(depth)], nn.LayerNorm(d))
        self.pred_in = nn.Linear(d, d)
        self.pred = nn.Sequential(*[Block(d, heads) for _ in range(pred_depth)], nn.LayerNorm(d))
        self.pred_out = nn.Linear(d, d)
        self.query = nn.Parameter(torch.randn(d) * 0.02)             # the "predict me" marker
        self.tok_t = None                                            # EMA target, built on first use
        self.enc_t = None

    # -- EMA target ------------------------------------------------------------------------------------
    def _ensure_target(self):
        if self.enc_t is None:
            import copy
            self.tok_t = copy.deepcopy(self.tok).requires_grad_(False)
            self.enc_t = copy.deepcopy(self.enc).requires_grad_(False)

    @torch.no_grad()
    def update_target(self, m: float | None = None):
        self._ensure_target()
        m = self.ema if m is None else m
        for tgt, src in ((self.tok_t, self.tok), (self.enc_t, self.enc)):
            for a, b in zip(tgt.parameters(), src.parameters()):
                a.mul_(m).add_(b.detach(), alpha=1 - m)
            for a, b in zip(tgt.buffers(), src.buffers()):
                a.copy_(b)

    # -- forward ---------------------------------------------------------------------------------------
    def encode(self, cont, idx, cell_masked=None):
        """Representation of a grid. `cell_masked=None` means the full unmasked state -- what a PROBE reads."""
        if cell_masked is None:
            cell_masked = torch.zeros_like(cont, dtype=torch.bool)
        B, T, E, _ = cont.shape
        x = self.tok(cont, idx, cell_masked).reshape(B, T * E, self.d)
        return self.enc(x).reshape(B, T, E, self.d)

    @torch.no_grad()
    def encode_target(self, cont, idx):
        self._ensure_target()
        B, T, E, _ = cont.shape
        zeros = torch.zeros_like(cont, dtype=torch.bool)
        x = self.tok_t(cont, idx, zeros).reshape(B, T * E, self.d)
        return self.enc_t(x).reshape(B, T, E, self.d)

    def forward(self, cont, idx, cell_masked, tok_target):
        """-> (loss, stats). `tok_target` (B,T,E) bool: tokens whose representation is predicted."""
        B, T, E, _ = cont.shape
        z = self.encode(cont, idx, cell_masked)                              # context view
        h = self.pred_in(z.reshape(B, T * E, self.d))
        q = tok_target.reshape(B, T * E, 1).float() * self.query[None, None]
        p = self.pred_out(self.pred(h + q)).reshape(B, T, E, self.d)
        with torch.no_grad():
            t = self.encode_target(cont, idx)
        sel = tok_target
        if not bool(sel.any()):
            raise RuntimeError("no target tokens in this batch -- the mask sampler guarantees at least one, "
                               "so this is a batching bug, not an empty draw")
        loss = F.l1_loss(_ln(p[sel]), _ln(t[sel].detach()))
        return loss, {"n_target_tok": int(sel.sum()), "n_tok": B * T * E}


def _ln(x):
    """Layer-norm over the feature axis, as in I-JEPA/V-JEPA's loss. See the scale note in train_jepa_pure."""
    return F.layer_norm(x, (x.shape[-1],))


@torch.no_grad()
def collapse_stats(z: torch.Tensor) -> dict:
    """AGENTS 2: predictability is never reported without informativeness.

    `eff_rank` is the participation ratio of the covariance eigenvalues -- the number this project has used
    since §25 -- and `min_std` catches dead dimensions that a mean std hides. A constant encoder is the global
    optimum of any pure prediction objective, so these columns decide whether a good loss means anything.
    """
    x = z.reshape(-1, z.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    sd = x.std(0)
    c = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = torch.linalg.eigvalsh(c.double()).clamp(min=0)
    eff = float((ev.sum() ** 2 / (ev ** 2).sum()).item()) if float(ev.sum()) > 0 else 0.0
    return {"std": float(sd.mean()), "min_std": float(sd.min()), "eff_rank": eff,
            "norm": float(z.reshape(-1, z.shape[-1]).norm(dim=-1).mean())}


def tok_targets(schema: Schema, cell_masked: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Which (frame, entity) tokens are PREDICTION targets: those with >= `frac` of their cells masked.

    This threshold is the whole reason the function exists, and the first version did not have it. Masking is
    at CELL granularity but the loss is at TOKEN granularity, so "any masked cell makes a target" marked
    544 of 544 tokens -- every token was simultaneously its own context and its own target, and the L1 would
    have been dominated by tokens the encoder could see almost all of. That is not V-JEPA's task; it is BERT
    with 2% corruption, and it would have trained to a low loss that meant nothing.

    RETRACTED, in the same minute it was written: this docstring first claimed that `group_tube` alone
    produces no targets, because one group is ~1-3 of an actor row's 20 cells. Measured over 100 draws at
    ratio 0.75 it produces a token-target fraction of **0.997**, the highest of the four -- the sampler keeps
    adding blocks until it reaches the ratio, so "group_tube only" masks ~11 of 15 groups, not one. The
    reasoning was about a single block and the sampler does not draw a single block. Measured, per 100 draws
    at ratio 0.75, T=8, frac 0.5:

        default 0.771 | entity_tube 0.778 | group_tube 0.997 | time_block 0.796 | cell 0.996

    So at the high ratios this pretext is meant to run at, nearly every token is a target and the CONTEXT
    lives in the surviving cells inside partially-masked tokens. That is the V-JEPA regime, where a 90% mask
    leaves ~10% of patches as context -- it is not the degenerate case the threshold was added to catch. The
    threshold still matters at LOW ratios, where it stops a token with one masked cell of twenty from being
    scored as if it were hidden. The trainer asserts a non-empty target set per batch and prints the achieved
    fraction every epoch, so a mixture that stops producing targets is visible rather than inferred.
    """
    v = schema.valid[None]                                        # (1,E,F)
    n = schema.valid.sum(1).clip(min=1)[None]                     # (1,E)
    hit = (cell_masked & v).sum(-1)
    return hit >= np.ceil(frac * n)
