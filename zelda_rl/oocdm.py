"""Object-Oriented Causal Dynamics Model: value codecs, class-shared encoders, per-field heads.

Stage 1 of docs/CAUSAL_WORLD_MODEL.md. Follows Yu, Ruan & Xing (ICML 2024) and the reference implementation
in ../oocdm, with the causal graph present but held at the FULL graph G1 -- no causal discovery yet. Stage 1
is isolating one hypothesis: does predicting each field in its own units, with class-shared parameters and
inter-object attention, beat the do-nothing floor that every latent model in LATENT_WORLD_MODEL.md failed to
beat? Causal discovery is Stage 2 and changes only the masks.

THE ONE INVARIANT EVERYTHING ELSE IS BUILT AROUND
-------------------------------------------------
**An untrained model is EXACTLY do-nothing.** Every head is a residual from the current value with a
zero-initialised output layer:

  real / count   v' = v + 0                      (delta head, zero-init)
  angle          v' = v + atan2(0 + (0,1))       (delta head with a fixed (cos,sin)=(1,0) bias)
  categorical    logits = IDENTITY_LOGIT * onehot(v) + 0
  boolean/bits   logit  = IDENTITY_LOGIT * (2v - 1) + 0

So skill 0 is the floor by construction, not by convention (AGENTS.md section 1), and `assert_identity_init`
checks it numerically rather than trusting the argument.

This cuts both ways and the risk is named up front: section 30 of the latent trail found that a residual
formulation makes "copy the input" the global optimum of a next-step objective when the thing being predicted
barely moves. That failure is why every metric here is reported BOTH aggregate and restricted to the steps
where the field actually changed (`id|chg` in scripts/eval_field_identity.py). A model that learns to copy
scores 0 on the restricted metric and is caught immediately.

NOTHING IN THIS FILE MAY DEVIATE FROM THE PAPER WITHOUT AN EXPLICIT, RECORDED DECISION. See AGENTS.md
section 13. A change gate, per-field `pos_weight` oversampling, binned direction/magnitude heads and
threshold calibration were all added here once, none of them are in Yu et al., all were removed, and the
session that added them never ran the paper's causal discovery at all.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from zelda_rl.oomdp import (SCHEMA_V3, TURNS, Field, ObjClass, ObjectGroup, Schema, err,
                            prep, unit, unpack_bits, wrap)

IDENTITY_LOGIT = 8.0        # logit margin that makes argmax == the current value at init

UNKNOWN = -1                # categorical index for a value never seen while fitting


# --------------------------------------------------------------------------------------------------------
# Codecs: raw column values <-> network input, and raw pairs <-> per-field training target.
#
# The paper's VType.raw2input plus its PType, split apart because our fields are messier than
# "position is an int in [-99, 99]": angles wrap, bitfields pack ten booleans into one float, and half the
# enums have a cardinality nobody has written down. Every constant here is MEASURED by `fit`, never declared
# (AGENTS.md section 3).
# --------------------------------------------------------------------------------------------------------
@dataclass
class FieldCodec:
    cls: str
    field: Field
    mean: np.ndarray            # (size,) for real/count
    std: np.ndarray             # (size,)
    dstd: np.ndarray            # (size,) std of the ONE-STEP DELTA -- what the head actually regresses
    values: np.ndarray          # (n,) observed values, for categorical/boolean

    @property
    def kind(self) -> str:
        return self.field.vtype.kind

    @property
    def name(self) -> str:
        return f"{self.cls}.{self.field.name}"

    @property
    def input_dim(self) -> int:
        k, sz = self.kind, self.field.vtype.size
        if k in ("real", "count"):
            return sz
        if k == "angle":
            return 2
        if k == "bits":
            return self.field.vtype.n
        if k in ("categorical", "boolean"):
            return len(self.values) + 1          # +1 for UNKNOWN
        raise ValueError(k)


    @property
    def n_out(self) -> int:
        """Width of the head's output for this field."""
        k = self.kind
        if k in ("real", "count"):
            return self.field.vtype.size
        if k == "angle":
            return 2                              # (cos, sin) of the DELTA
        if k == "bits":
            return self.field.vtype.n
        return len(self.values) + 1

    # -- input side ---------------------------------------------------------------------------------------
    def encode_input(self, v: np.ndarray) -> np.ndarray:
        """(..., size) raw -> (..., input_dim) float32 network input."""
        k = self.kind
        if k in ("real", "count"):
            return ((v - self.mean) / self.std).astype(np.float32)
        if k == "angle":
            a = self._rad(v)[..., 0]
            return np.stack([np.cos(a), np.sin(a)], -1).astype(np.float32)
        if k == "bits":
            return unpack_bits(v, self.field.vtype.n)
        idx = self.index(v)
        out = np.zeros((*idx.shape, len(self.values) + 1), np.float32)
        np.put_along_axis(out, np.where(idx < 0, len(self.values), idx)[..., None], 1.0, -1)
        return out

    def index(self, v: np.ndarray) -> np.ndarray:
        """(..., 1) raw -> (...) index into `values`, or UNKNOWN."""
        flat = v[..., 0]
        i = np.searchsorted(self.values, flat)
        i = np.clip(i, 0, len(self.values) - 1)
        return np.where(self.values[i] == flat, i, UNKNOWN)

    def _rad(self, v: np.ndarray) -> np.ndarray:
        return v * (2 * np.pi / TURNS) if self.field.vtype.unit == "binang" else v

    # -- target side --------------------------------------------------------------------------------------
    def target(self, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """(cur, nxt) raw -> the head's training target, in the head's own parameterisation.

        Binned vector fields return [sector, magbin, dy/dstd_y] as float; the loss casts the first two back
        to indices. Packing them together keeps one target array per field, so nothing downstream needs a
        second code path for the shape.
        """
        k = self.kind
        if k in ("real", "count"):
            return ((nxt - cur) / self.dstd).astype(np.float32)       # standardised delta
        if k == "angle":
            d = wrap(self._rad(nxt) - self._rad(cur))[..., 0]
            return np.stack([np.cos(d), np.sin(d)], -1).astype(np.float32)
        if k == "bits":
            return unpack_bits(nxt, self.field.vtype.n)
        idx = self.index(nxt)
        return np.where(idx < 0, len(self.values), idx).astype(np.int64)[..., None]

    def apply(self, cur: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Head output + current value -> the predicted RAW value. The inverse of `target`, used by rollouts
        and by every metric, so predictions are always scored in the field's own units."""
        k = self.kind
        if k in ("real", "count"):
            return cur + out * self.dstd
        if k == "angle":
            d = np.arctan2(out[..., 1], out[..., 0])[..., None]
            r = wrap(self._rad(cur) + d)
            return r * (TURNS / (2 * np.pi)) if self.field.vtype.unit == "binang" else r
        if k == "bits":
            return pack_bits(out > 0)
        idx = out.argmax(-1)
        safe = np.clip(idx, 0, len(self.values) - 1)
        return np.where(idx[..., None] >= len(self.values), cur, self.values[safe][..., None])


def pack_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.int64) * (1 << np.arange(b.shape[-1]))).sum(-1, keepdims=True).astype(np.float32)


def fit_codecs(groups: Iterable[dict[str, ObjectGroup]], schema: Schema = SCHEMA_V3,
               changed_only: bool = False) -> dict[str, dict[str, FieldCodec]]:
    """Measure every codec's constants from decomposed episodes. Returns {class: {field: codec}}.

    Scales come from the data the model will see, restricted to steps where the object EXISTS -- a despawned
    actor's zeroed row must not drag a position mean toward the origin.

    `changed_only` sets WHICH steps `dstd` is measured over and must match how the value head is trained.
    The paper's head trains on every transition, so the default is all steps. Kept as a switch only because
    the two subsets differ by 3.4x-50x (Link.pos 2.70 -> 9.11, y_dist_to_ledge 7.9 -> 400) and a mismatch
    silently produces Gaussian-NLL targets in the hundreds.

    The scale is `max(std, mean|delta|)` rather than std alone: a field that always changes by the same
    amount (Link.grass, always +/-5) has std 0 on changed steps, and dividing by the 1e-3 floor would
    reintroduce the same blow-up from the other direction.
    """
    acc: dict[str, dict[str, list]] = {}
    for g in groups:
        for cname, grp in g.items():
            for f in grp.cls.fields:
                if f.role == "action" or f.name not in grp.attrs:
                    continue
                v = grp.attrs[f.name]
                both = grp.mask[:-1] & grp.mask[1:]
                cur, nxt = v[:-1][both], v[1:][both]
                a = acc.setdefault(cname, {}).setdefault(f.name, [[], [], [], []])
                if len(a[0]) < 64:                                     # 64 episodes is plenty for a scale
                    a[0].append(v[grp.mask])
                    if f.vtype.kind in ("real", "count"):
                        dl = nxt - cur
                        a[1].append(dl)
                        if len(dl):
                            # Changed deltas are pooled ACROSS episodes before any fallback. Falling back
                            # per episode was the bug: a field that changes on 0.08% of steps has no changed
                            # step in most episodes, so the fallback filled the pool with zeros and the
                            # scale came out as the all-steps one after all (y_dist_to_ledge 7.8, not 400).
                            a[3].append(dl[err(f, prep(f, nxt), prep(f, cur)) > 0])
                    elif f.vtype.kind in ("categorical", "boolean"):
                        a[2].append(np.unique(v[grp.mask]))

    out: dict[str, dict[str, FieldCodec]] = {}
    for cname, fields in acc.items():
        c = next(k for k in schema.classes if k.name == cname)
        out[cname] = {}
        for fname, (vals, deltas, seen, deltas_chg) in fields.items():
            f = c.field(fname)
            v = np.concatenate(vals) if vals else np.zeros((1, f.vtype.size), np.float32)
            mean, std = v.mean(0), v.std(0)
            dc = np.concatenate(deltas_chg) if deltas_chg else np.zeros((0, f.vtype.size), np.float32)
            d = dc if (changed_only and len(dc)) else (
                np.concatenate(deltas) if deltas else np.zeros((1, f.vtype.size), np.float32))
            dstd = np.maximum(d.std(0), np.abs(d).mean(0)) if len(d) else np.zeros(f.vtype.size)
            # A field with no changed example in the fitting subset (Link.grass changes on 0.026% of steps)
            # would otherwise land on the 1e-3 floor, and any prediction it made would be enormous. Fall back
            # to the field's own spread, which is a sane bound on "how much could this change".
            dstd = np.maximum(dstd, 0.01 * std)
            values = np.unique(np.concatenate(seen)) if seen else np.zeros(0, np.float32)
            out[cname][fname] = FieldCodec(cname, f, mean.astype(np.float32),
                                           np.maximum(std, 1e-3).astype(np.float32),
                                           np.maximum(dstd, 1e-3).astype(np.float32),
                                           values.astype(np.float32))
    return out


def describe_codecs(codecs: dict[str, dict[str, FieldCodec]]) -> str:
    lines = []
    for cname, fs in codecs.items():
        lines.append(f"  {cname}")
        for fname, c in fs.items():
            extra = (f"values={len(c.values)}" if c.kind in ("categorical", "boolean")
                     else (f"dstd={np.array2string(c.dstd, precision=3)}" if c.kind in ("real", "count") else ""))
            lines.append(f"    {fname:<20} {c.kind:<12} in={c.input_dim:<4} out={c.n_out:<4} {extra}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------
# Packing episodes into per-class tensors.
#
# The row -> class assignment is NOT global: over the four v3 collections there are 5 distinct slot->actor-id
# maps, because an episode starts with one, two or three Deku Babas (and in two episodes a Baba is allocated
# a slot after the Skulltulas). So objects are grouped per episode by `oomdp.decompose` and padded to a fixed
# per-class capacity with mask=False. Nothing here may assume a fixed layout.
# --------------------------------------------------------------------------------------------------------
@dataclass
class Packed:
    """All transitions of a dataset, grouped by class.

    attrs[cls][field]: (N_frames, n_obj_cls, size)     mask[cls]: (N_frames, n_obj_cls)
    pairs: (N_pairs,) frame indices t such that t and t+1 are in the SAME episode.
    """
    attrs: dict[str, dict[str, np.ndarray]]
    mask: dict[str, np.ndarray]
    pairs: np.ndarray
    episode: np.ndarray
    capacity: dict[str, int]

    def __len__(self) -> int:
        return len(self.pairs)

    def pad_to(self, capacity: dict[str, int]) -> "Packed":
        """Grow the object axis to a shared capacity.

        Splits are packed independently, and a split can simply contain fewer Deku Babas than another --
        the tune slots did. The model is built for ONE capacity, so every split must be padded to it before
        a batch is made, or the self-attention mask indexes past the end of the object axis.
        """
        for c, n in capacity.items():
            have = self.mask[c].shape[1]
            assert n >= have, f"{c}: capacity {n} < packed {have}"
            if n == have:
                continue
            self.mask[c] = np.pad(self.mask[c], ((0, 0), (0, n - have)))
            for f, a in self.attrs[c].items():
                self.attrs[c][f] = np.pad(a, ((0, 0), (0, n - have), (0, 0)))
            self.capacity[c] = n
        return self


def pack(episodes: Sequence[tuple[np.ndarray, np.ndarray]], schema: Schema = SCHEMA_V3) -> Packed:
    """[(state (T, width), action (T,))] -> Packed. Capacity per class is measured, then asserted."""
    decomposed = []
    cap: dict[str, int] = {c.name: 0 for c in schema.classes}
    from zelda_rl.oomdp import decompose
    for st, act in episodes:
        g = decompose(st, act)
        decomposed.append(g)
        for cname, grp in g.items():
            cap[cname] = max(cap[cname], grp.n_obj)

    attrs: dict[str, dict[str, list]] = {c.name: {} for c in schema.classes}
    masks: dict[str, list] = {c.name: [] for c in schema.classes}
    pairs, eps, off = [], [], 0
    for ep, g in enumerate(decomposed):
        T = len(next(iter(g.values())).mask)
        pairs.append(off + np.arange(T - 1))
        eps.append(np.full(T, ep, np.int64))
        for c in schema.classes:
            n = cap[c.name]
            grp = g.get(c.name)
            m = np.zeros((T, n), bool)
            if grp is not None:
                m[:, : grp.n_obj] = grp.mask
            masks[c.name].append(m)
            for f in c.fields:
                if f.role == "action" and c.name != "Link":
                    continue
                size = 2 if f.name == "stick" else f.vtype.size
                a = np.zeros((T, n, size), np.float32)
                if grp is not None and f.name in grp.attrs:
                    a[:, : grp.n_obj] = grp.attrs[f.name]
                attrs[c.name].setdefault(f.name, []).append(a)
        off += T

    out_attrs = {c: {f: np.concatenate(v) for f, v in fs.items()} for c, fs in attrs.items()}
    out_mask = {c: np.concatenate(v) for c, v in masks.items()}
    p = Packed(out_attrs, out_mask, np.concatenate(pairs), np.concatenate(eps), cap)
    for c in schema.classes:                                   # capacity is measured, then asserted
        assert out_mask[c.name].shape[1] == cap[c.name], c.name
        assert not out_mask[c.name][:, cap[c.name]:].any() if cap[c.name] else True
    return p


# --------------------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------------------
import torch                                                   # noqa: E402
import torch.nn as nn                                          # noqa: E402
import torch.nn.functional as Fn                               # noqa: E402



# --------------------------------------------------------------------------------------------------------
# The Object-Oriented Causal Graph (paper section 4.1, ../oocdm/core/causal_graph.py).
#
# Two kinds of edge and nothing else:
#   local_matrices[C][j, i]  field i of an object affects field j of THAT SAME object   (C.U -> V')
#   global_matrix[J, I]      field I of ANY OTHER object affects field J of any object  (Cl.U -> Ck.V')
#
# Indices are class-level, so the graph does not grow when an actor spawns. `edge_budget()` in oomdp.py
# measures 1,741 local + 6,545 global = 8,286 candidates against 132,132 for a flat CDL over the same state.
# --------------------------------------------------------------------------------------------------------
class OOCGraph:
    def __init__(self, order: Sequence[ObjClass], field_names: dict, target_names: dict, full: bool = True):
        self.order = list(order)
        self.field_names, self.target_names = field_names, target_names
        # global index space: every (class, field) flattened in class order
        self.gin, self.gout = {}, {}
        for c in self.order:
            for f in self.field_names[c.name]:
                self.gin[(c.name, f)] = len(self.gin)
            for t in self.target_names[c.name]:
                self.gout[(c.name, t)] = len(self.gout)
        self.n_in, self.n_out = len(self.gin), len(self.gout)
        v = bool(full)
        self.global_matrix = np.full((self.n_out, self.n_in), v)
        self.local = {c.name: np.full((len(self.target_names[c.name]), len(self.field_names[c.name])), v)
                      for c in self.order}

    def n_edges(self) -> int:
        return int(self.global_matrix.sum() + sum(m.sum() for m in self.local.values()))

    def set_local(self, cls: str, target: str, parents: Sequence[str]) -> None:
        j = self.target_names[cls].index(target)
        self.local[cls][j, :] = False
        for pnt in parents:
            self.local[cls][j, self.field_names[cls].index(pnt)] = True

    def set_global(self, cls: str, target: str, parents: Sequence[tuple[str, str]] | None) -> None:
        J = self.gout[(cls, target)]
        self.global_matrix[J, :] = False
        for cn, fn in (parents or ()):
            self.global_matrix[J, self.gin[(cn, fn)]] = True

    def local_mask(self, cls: str, device) -> torch.Tensor:
        return torch.as_tensor(self.local[cls], dtype=torch.bool, device=device)

    def global_mask(self, cls: str, device) -> torch.Tensor:
        """(n_out_total, n_field_of_this_class) -- the columns of the global matrix this class supplies."""
        cols = [self.gin[(cls, f)] for f in self.field_names[cls]]
        return torch.as_tensor(self.global_matrix[:, cols], dtype=torch.bool, device=device)

    def state_dict(self) -> dict:
        """Serialisable form, so a run can resume with the graph it had (paper: train_ooc.py save/load)."""
        return {"global": self.global_matrix.tolist(),
                "local": {k: v.tolist() for k, v in self.local.items()}}

    def load_state_dict(self, d: dict) -> "OOCGraph":
        self.global_matrix = np.asarray(d["global"], bool)
        self.local = {k: np.asarray(v, bool) for k, v in d["local"].items()}
        return self

    def edge_set(self) -> set:
        """Edges as printable names, for diffing one iteration's graph against the last."""
        out = set()
        for (tc, tf), J in self.gout.items():
            for (sc, sf), I in self.gin.items():
                if self.global_matrix[J, I]:
                    out.add(f"G {sc}.{sf} -> {tc}.{tf}'")
        for c, m in self.local.items():
            for j, t in enumerate(self.target_names[c]):
                for i, f in enumerate(self.field_names[c]):
                    if m[j, i]:
                        out.add(f"L {c}.{f} -> {t}'")
        return out

    def describe(self, top: int = 12) -> str:
        lines = [f"OOCG: {self.n_edges()} of {self.n_out * self.n_in + sum(m.size for m in self.local.values())}"
                 f" candidate edges active"]
        for c in self.order:
            for j, t in enumerate(self.target_names[c.name]):
                par = [self.field_names[c.name][i] for i in np.nonzero(self.local[c.name][j])[0]]
                if len(par) < len(self.field_names[c.name]):
                    lines.append(f"  local  {c.name}.{t}' <- {par}")
        return "\n".join(lines[: top + 1])



# --------------------------------------------------------------------------------------------------------
# Mask generators (paper section 4.4, ../oocdm/alg/model/mask_generator.py).
#
# Training optimises the average instance log-likelihood under THREE masks at once:
#
#     loss = -(ll_graph + ll_full + ll_random) / 3        (train_ooc.py:90)
#
# ll_full     keeps the model good at the full graph, which is what CMI is measured against.
# ll_random   each edge kept with probability p_edge, RESAMPLED EVERY BATCH. This is the term that teaches
#             the model to answer "what would you predict WITHOUT this parent?" -- the question causal
#             discovery asks it. Without it, CMI interrogates a model that has never seen a masked input.
# ll_graph    the current graph estimate, empty at init, sharpening the model on what it currently believes.
# --------------------------------------------------------------------------------------------------------
class MaskGen:
    def local(self, cls: str, batch: int, device) -> torch.Tensor: raise NotImplementedError
    def glob(self, cls: str, batch: int, device) -> torch.Tensor: raise NotImplementedError


class FullMask(MaskGen):
    def __init__(self, graph: "OOCGraph"):
        self.g = graph

    def local(self, cls, batch, device):
        return torch.ones(len(self.g.target_names[cls]), len(self.g.field_names[cls]),
                          dtype=torch.bool, device=device)

    def glob(self, cls, batch, device):
        return torch.ones(self.g.n_out, len(self.g.field_names[cls]), dtype=torch.bool, device=device)


class GraphMask(MaskGen):
    def __init__(self, graph: "OOCGraph"):
        self.g = graph

    def local(self, cls, batch, device):
        return self.g.local_mask(cls, device)

    def glob(self, cls, batch, device):
        return self.g.global_mask(cls, device)


class RandomMask(MaskGen):
    """Each edge kept independently with probability `p_edge`, resampled per batch AND per batch element."""

    def __init__(self, graph: "OOCGraph", p_edge: float = 0.9):
        self.g, self.p = graph, p_edge

    def local(self, cls, batch, device):
        return torch.rand(batch, len(self.g.target_names[cls]), len(self.g.field_names[cls]),
                          device=device) <= self.p

    def glob(self, cls, batch, device):
        return torch.rand(batch, self.g.n_out, len(self.g.field_names[cls]), device=device) <= self.p



class DroppedMask(MaskGen):
    """The full graph with ONE input field removed -- the counterfactual CMI is measured against.

    Batch-independent, so it takes the cheap folded path. `drop_global` removes a (class, field) column from
    the global matrix for every target; `drop_local` removes a field from one class's local matrix.
    """

    def __init__(self, graph: "OOCGraph", drop_global: tuple[str, str] | None = None,
                 drop_local: tuple[str, str] | None = None):
        self.g, self.dg, self.dl = graph, drop_global, drop_local

    def local(self, cls, batch, device):
        m = torch.ones(len(self.g.target_names[cls]), len(self.g.field_names[cls]),
                       dtype=torch.bool, device=device)
        if self.dl is not None and self.dl[0] == cls:
            m[:, self.g.field_names[cls].index(self.dl[1])] = False
        return m

    def glob(self, cls, batch, device):
        m = torch.ones(self.g.n_out, len(self.g.field_names[cls]), dtype=torch.bool, device=device)
        if self.dg is not None and self.dg[0] == cls:
            m[:, self.g.field_names[cls].index(self.dg[1])] = False
        return m


class MultiLinear(nn.Module):
    """n independent Linear layers applied along a named axis. The paper's parameter sharing device: the
    weight is indexed by FIELD, and the object axis is a batch dim, so every instance of a class shares it."""

    def __init__(self, n: int, d_in: int, d_out: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(n, d_out))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (..., n, d_in) -> (..., n, d_out)
        return torch.einsum("...ni,nio->...no", x, self.w) + self.b


class HeteroEmbed(nn.Module):
    """One Linear per field, because fields have different input widths (the paper's HeterogenousLinear)."""

    def __init__(self, dims: Sequence[int], d: int):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(n, d) for n in dims])

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack([l(x) for l, x in zip(self.lins, xs)], -2)       # (B, n_obj, n_field, d)


class MaskedFieldLinear(nn.Module):
    """The paper's masked flatten+linear, with the mask FOLDED INTO THE WEIGHT.

    `masked_retain(x, attrmask)` zeroes the embeddings of non-parent fields, flattens (n_field, d) and applies
    a per-target-field Linear. Done literally that needs a (B, n_obj, n_target, n_field, d) intermediate,
    which for our 77 global state fields is hundreds of MB. Since the mask is the same for every batch
    element (it comes from the graph, not from the data), folding it into the weight is algebraically
    identical and allocates nothing:

        out[b,o,n,e] = sum_{f,k} x[b,o,f,k] * mask[n,f] * W[n,f,k,e]

    A per-batch-element mask -- which the paper's RandomMaskGenerator produces -- cannot be folded, because
    the weight would have to differ per batch element. Those take the expanded path, which materialises
    (B, n_obj, n_target, n_field, d). That is why P1 trains at a smaller batch.
    """

    def __init__(self, n_target: int, n_field: int, d: int, d_out: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_target, n_field, d, d_out))
        self.b = nn.Parameter(torch.zeros(n_target, d_out))
        nn.init.kaiming_uniform_(self.w.view(n_target, n_field * d, d_out), a=5 ** 0.5)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, n_obj, n_field, d); mask (n_target, n_field) or (B, n_target, n_field).

        -> (B, n_obj, n_target, d_out).   b=batch o=object f=field k=embed n=target e=out
        """
        if mask.dim() == 2:                                     # graph / full: fold into the weight
            return torch.einsum("bofk,nfke->bone", x, self.w * mask[:, :, None, None]) + self.b
        # random: the mask differs per batch element, so zero the embeddings instead (paper's masked_retain)
        xm = x.unsqueeze(2) * mask.unsqueeze(1).unsqueeze(-1)   # (B, n_obj, n_target, n_field, d)
        return torch.einsum("bonfk,nfke->bone", xm, self.w) + self.b


class ClassModule(nn.Module):
    """Everything owned by one class: field embeddings, the masked local and global encodings, and one head
    per state field. `f_global` produces an encoding of each object FOR EACH global state field, because
    masking is per target field -- Stage 1 collapsed this to one encoding per object and that simplification
    is incompatible with a graph."""

    def __init__(self, cls: ObjClass, codecs: dict[str, FieldCodec], fields: Sequence[str],
                 targets: Sequence[str], n_out_total: int, d: int, d_h: int, d_g: int,
                 d_k: int, d_v: int, d_dec: int):
        super().__init__()
        self.cls_name, self.fields, self.targets = cls.name, list(fields), list(targets)
        self.embed = HeteroEmbed([codecs[f].input_dim if f in codecs else 2 if f == "stick" else 1
                                  for f in self.fields], d)
        n_f, n_t = len(self.fields), len(self.targets)
        self.f_local = MaskedFieldLinear(n_t, n_f, d, d_h)
        self.f_global = MaskedFieldLinear(n_out_total, n_f, d, d_g)
        self.fq, self.fk, self.fv = MultiLinear(n_t, d_h, d_k), MultiLinear(n_t, d_g, d_k), \
            MultiLinear(n_t, d_g, d_v)
        self.dec1, self.dec2 = MultiLinear(n_t, d_h + d_v, d_dec), MultiLinear(n_t, d_dec, d_dec)
        self.heads = nn.ModuleList()
        for t in self.targets:                                  # zero-init: an untrained head is do-nothing
            h = nn.Linear(d_dec, codecs[t].n_out)
            nn.init.zeros_(h.weight), nn.init.zeros_(h.bias)
            self.heads.append(h)

    def encode(self, xs: Sequence[torch.Tensor], local_mask: torch.Tensor, global_mask: torch.Tensor):
        x = self.embed(xs)                                              # (B, n_obj, n_field, d)
        return (Fn.leaky_relu(self.f_local(x, local_mask)),             # (B, n_obj, n_t, d_h)
                Fn.leaky_relu(self.f_global(x, global_mask)))           # (B, n_obj, n_out_total, d_g)


class OOCDM(nn.Module):
    """p(S' | S, A) factored over (class, field), class-shared parameters, inter-object attention, and a
    causal graph that masks non-parents (paper sections 4.1-4.2).

    The graph enters exactly where the paper puts it: as a boolean broadcast over the FIELD axis before the
    flatten+linear, once for the local encoding (own fields -> own next field) and once for the global
    encoding (this object's fields, as seen by every target field of every class). Attention masks self out,
    which is the paper's `j != i`, so the ONLY route from an object's own field to its own next field is the
    local path -- that is the local causality C.U -> V'.
    """

    def __init__(self, codecs: dict[str, dict[str, FieldCodec]], capacity: dict[str, int],
                 schema: Schema = SCHEMA_V3, targets: dict[str, list[str]] | None = None,
                 d: int = 16, d_h: int = 64, d_g: int = 32, d_k: int = 32, d_v: int = 64, d_dec: int = 128,
                 graph: "OOCGraph | None" = None):
        super().__init__()
        self.schema, self.codecs, self.capacity = schema, codecs, capacity
        self.order = [c for c in schema.classes if capacity.get(c.name, 0) > 0]
        self.field_names, self.target_names = {}, {}
        for c in self.order:
            self.field_names[c.name] = [f.name for f in c.fields
                                        if f.name in codecs[c.name] or f.role == "action"]
            self.target_names[c.name] = (list(targets[c.name]) if targets else
                                         [f.name for f in c.state_fields() if f.name in codecs[c.name]])
        self.graph = graph or OOCGraph(self.order, self.field_names, self.target_names, full=True)
        n_out_total = self.graph.n_out
        self.mods = nn.ModuleDict({
            c.name: ClassModule(c, codecs[c.name], self.field_names[c.name], self.target_names[c.name],
                                n_out_total, d, d_h, d_g, d_k, d_v, d_dec) for c in self.order})
        self.log_scale = nn.ParameterDict({
            f"{c.name}|{t}": nn.Parameter(torch.zeros(codecs[c.name][t].n_out))
            for c in self.order for t in self.target_names[c.name]
            if codecs[c.name][t].kind in ("real", "count")})

    def set_graph(self, g: "OOCGraph") -> None:
        self.graph = g

    def forward(self, batch: dict, maskgen: "MaskGen | None" = None) -> dict[str, torch.Tensor]:
        """batch['x'][cls][field] (B, n_obj, in_dim); 'mask'[cls] (B, n_obj); 'raw'[cls][field] for the
        identity residual. `maskgen` selects which graph the forward is run under (paper section 4.4);
        the default is the current graph estimate. Returns {'Cls|field': (B, n_obj, n_out)}."""
        dev = batch["mask"][self.order[0].name].device
        mg = maskgen or GraphMask(self.graph)
        bsz = batch["mask"][self.order[0].name].shape[0]
        local, glob, obj_mask = {}, [], []
        for c in self.order:
            m = self.mods[c.name]
            l, gl = m.encode([batch["x"][c.name][f] for f in self.field_names[c.name]],
                             mg.local(c.name, bsz, dev), mg.glob(c.name, bsz, dev))
            local[c.name] = l
            glob.append(gl)                                    # (B, n_obj_c, n_out_total, d_g)
            obj_mask.append(batch["mask"][c.name])
        G = torch.cat(glob, 1)                                 # (B, n_obj_all, n_out_total, d_g)
        M = torch.cat(obj_mask, 1)                             # (B, n_obj_all)

        out, base = {}, 0
        for c in self.order:
            m, n_obj = self.mods[c.name], self.capacity[c.name]
            cols = [self.graph.gout[(c.name, t)] for t in self.target_names[c.name]]
            Gc = G[:, :, cols, :]                              # (B, n_obj_all, n_t, d_g)
            q = m.fq(local[c.name])                            # (B, n_obj_c, n_t, d_k)
            k, v = m.fk(Gc), m.fv(Gc)                          # (B, n_obj_all, n_t, d_k/d_v)
            att = torch.einsum("botk,bjtk->botj", q, k) / q.shape[-1] ** 0.5
            idx = torch.arange(n_obj, device=att.device)
            self_edge = torch.zeros(n_obj, M.shape[1], dtype=torch.bool, device=att.device)
            self_edge[idx, base + idx] = True                  # no attention to itself (paper's j != i)
            vis = (M[:, None, None, :] & ~self_edge[None, :, None, :]).expand_as(att)
            att = att.masked_fill(~vis, float("-inf"))
            att = torch.where(vis.any(-1, keepdim=True), att.softmax(-1), torch.zeros_like(att))
            h = torch.cat([local[c.name], torch.einsum("botj,bjtv->botv", att, v)], -1)
            h = Fn.leaky_relu(m.dec2(Fn.leaky_relu(m.dec1(h))))
            for i, t in enumerate(self.target_names[c.name]):
                out[f"{c.name}|{t}"] = self._identity(c.name, t, m.heads[i](h[:, :, i]),
                                                      batch["raw"][c.name][t])
            base += n_obj
        return out

    def _identity(self, cname: str, t: str, raw_out: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """Residual-from-identity offset, so a zero-init head predicts EXACTLY no change."""
        cod = self.codecs[cname][t]
        k = cod.kind
        if k in ("real", "count"):
            return raw_out
        if k == "angle":
            bias = torch.zeros_like(raw_out)
            bias[..., 0] = 1.0
            return raw_out + bias
        if k == "bits":
            b = ((cur.long() >> torch.arange(cod.field.vtype.n, device=cur.device)) & 1).float()
            return raw_out + IDENTITY_LOGIT * (2 * b - 1)
        vals = torch.as_tensor(cod.values, device=cur.device)
        eq = (cur == vals).float()
        unknown = (1.0 - eq.sum(-1, keepdim=True)).clamp(0.0, 1.0)
        return raw_out + IDENTITY_LOGIT * torch.cat([eq, unknown], -1)


def make_batch(p: Packed, codecs: dict[str, dict[str, FieldCodec]], model: "OOCDM",
               idx: np.ndarray, device) -> dict:
    """Frames t -> the tensors `OOCDM.forward` and `field_losses` consume. Targets are read at t+1.

    Validity is per field, not global: every field needs the object to exist at BOTH t and t+1, except
    `present` itself, whose despawn (1 -> 0) is precisely the thing to predict.
    """
    t = np.asarray(idx)
    x, raw, mask, y, valid = {}, {}, {}, {}, {}
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)   # noqa: E731
    for c in model.order:
        cn = c.name
        at, m = p.attrs[cn], p.mask[cn]
        here, there = m[t], m[t + 1]
        mask[cn] = T(here, torch.bool)
        x[cn], raw[cn], y[cn], valid[cn] = {}, {}, {}, {}
        for f in model.field_names[cn]:
            cod = codecs[cn].get(f)
            v = at[f][t]
            x[cn][f] = T(cod.encode_input(v) if cod is not None else v)
        for tgt in model.target_names[cn]:
            cod, fld = codecs[cn][tgt], c.field(tgt)
            cur, nxt = at[tgt][t], at[tgt][t + 1]
            raw[cn][tgt] = T(cur)
            tv = cod.target(cur, nxt)
            y[cn][tgt] = T(tv, torch.int64 if cod.kind in ("categorical", "boolean") else torch.float32)
            valid[cn][tgt] = T(here if fld.is_mask else (here & there), torch.bool)
    return {"x": x, "raw": raw, "mask": mask, "y": y, "valid": valid, "t": t}


def field_losses(model: "OOCDM", out: dict, batch: dict) -> dict[str, torch.Tensor]:
    """One scalar per (class, field), each already averaged per instance -- the paper's AILL, per field.

    Averaging PER INSTANCE rather than per batch element is what makes the objective invariant to how many
    actors are alive, which is the whole point of the object-oriented formulation.
    """
    losses = {}
    for c in model.order:
        for t in model.target_names[c.name]:
            key = f"{c.name}|{t}"
            cod, pred, tgt = model.codecs[c.name][t], out[key], batch["y"][c.name][t]
            v = batch["valid"][c.name][t]
            if not bool(v.any()):
                continue
            k = cod.kind
            if k in ("real", "count"):
                sc = model.log_scale[f"{c.name}|{t}"]
                e = 0.5 * ((pred - tgt) / sc.exp()) ** 2 + sc               # Gaussian NLL, learned scale
                l = e.mean(-1)[v].mean()
            elif k == "angle":
                u = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                l = (1 - (u * tgt).sum(-1))[v].mean()                       # 1 - cos(pred - true)
            elif k == "bits":
                l = Fn.binary_cross_entropy_with_logits(pred, tgt, reduction="none").mean(-1)[v].mean()
            else:
                l = Fn.cross_entropy(pred[v], tgt[..., 0][v], reduction="mean")
            losses[key] = l
    return losses


# --------------------------------------------------------------------------------------------------------
# Trivial baselines. AGENTS.md section 1 demands a free-signal control, and section 28 names this exact one:
# "with velocity in z, next position is linear extrapolation". It was not run before Stage 1's value heads
# were built, and four architecture experiments were scored against a floor that was too low as a result.
#
# Measured on all four v3 collections, changed steps, XZ plane: do-nothing MAE 6.533, pos + 1.67*vel MAE
# 4.855 -> val|chg +0.257, against the trained value head's +0.064. Any value head that does not beat this
# has not earned its parameters.
# --------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_raw(model: "OOCDM", out: dict, batch: dict) -> dict[str, np.ndarray]:
    """Head outputs -> predicted RAW field values, so every metric is in the field's own units."""
    res = {}
    for c in model.order:
        for t in model.target_names[c.name]:
            key = f"{c.name}|{t}"
            res[key] = model.codecs[c.name][t].apply(batch["raw"][c.name][t].cpu().numpy(),
                                                     out[key].float().cpu().numpy())
    return res


@torch.no_grad()
def assert_identity_init(model: "OOCDM", batch: dict, tol: float = 1e-3) -> None:
    """An untrained model must be EXACTLY do-nothing (AGENTS.md section 1: a zero-init residual predictor
    IS the identity, so skill 0 is the floor). Asserted numerically, not argued."""
    pred = predict_raw(model, model(batch), batch)
    bad = []
    for c in model.order:
        for t in model.target_names[c.name]:
            key, cod = f"{c.name}|{t}", model.codecs[c.name][t]
            cur = batch["raw"][c.name][t].cpu().numpy()
            v = batch["valid"][c.name][t].cpu().numpy()
            if not v.any():
                continue
            d = pred[key][v] - cur[v]
            if cod.kind == "angle" and cod.field.vtype.unit == "binang":
                d = (d + TURNS / 2) % TURNS - TURNS / 2
            e = float(np.abs(d).max())
            rel = e / float(max(np.abs(cur[v]).max(), 1.0))
            if rel > tol:
                bad.append((key, e, rel))
    if bad:
        raise AssertionError("zero-init model is not the identity: "
                             + ", ".join(f"{k} max|delta|={e:.4g} (rel {r:.2g})" for k, e, r in bad[:8]))


def velocity_baseline(p: Packed, cls: str = "Link", pos: str = "pos", vel: str = "vel",
                      plane=(0, 2)) -> dict[str, float]:
    """`delta = k * vel[t]` with k fitted by least squares. Returns the control every value head is held to.

    Scored in the PLANE by default: vel_y has a median of -4.0 on every step (gravity pulling against the
    ground) while pos_y barely moves, so a naive 3-component version of this control scored -0.057 and hid
    the fact that the baseline beats the model.
    """
    t = p.pairs
    a, v, m = p.attrs[cls][pos], p.attrs[cls][vel], p.mask[cls]
    ok = m[t] & m[t + 1]
    d = (a[t + 1] - a[t])[ok][:, list(plane)]
    u = v[t][ok][:, list(plane)]
    moved = np.linalg.norm(d, axis=1) > 0
    d, u = d[moved], u[moved]
    if not len(d):
        return {}
    k = float((d * u).sum() / max((u * u).sum(), 1e-9))
    e_id, e_v = np.abs(d).mean(1), np.abs(d - k * u).mean(1)
    return {"k": k, "n": len(d), "identity_mae": float(e_id.mean()), "baseline_mae": float(e_v.mean()),
            "val_chg": float(1 - e_v.mean() / e_id.mean())}


# --------------------------------------------------------------------------------------------------------
# Causal discovery (paper section 4.3, ../oocdm/scripts_/train_ooc.py:get_causal_graph).
#
#   CMI(edge) = E[ log f(V' | full graph) ] - E[ log f(V' | full graph minus that edge) ]
#
# and the edge is kept when CMI > causal_threshold. No second model is trained: the SAME predictor is asked
# both questions, which is only meaningful because the random-mask term of P1 taught it to answer with
# parents missing.
#
# KNOWN APPROXIMATION, recorded rather than hidden: the paper's heads all emit proper distributions, so its
# log f is a log-probability. Ours does too for real/count (Gaussian), categorical and boolean (softmax) and
# bits (Bernoulli) -- but the ANGLE head emits a direction, not a distribution, and its `1 - cos` loss is a
# monotone surrogate rather than a log-probability. Differences are still signed correctly for a given field,
# but the nats scale is wrong for angles, so `causal_threshold` does not mean the same thing for them.
# --------------------------------------------------------------------------------------------------------
def field_logprob(model: "OOCDM", out: dict, batch: dict) -> dict[str, np.ndarray]:
    """Per-(object, step) log f(V' | parents) for every field, masked to valid entries."""
    res = {}
    for c in model.order:
        for t in model.target_names[c.name]:
            key = f"{c.name}|{t}"
            cod, pred, tgt = model.codecs[c.name][t], out[key], batch["y"][c.name][t]
            v = batch["valid"][c.name][t]
            if not bool(v.any()):
                continue
            k = cod.kind
            if k in ("real", "count"):
                sc = model.log_scale[key]
                lp = (-0.5 * ((pred - tgt) / sc.exp()) ** 2 - sc - 0.5 * np.log(2 * np.pi)).sum(-1)
            elif k == "angle":
                u = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                lp = (u * tgt).sum(-1) - 1.0                        # surrogate, see the note above
            elif k == "bits":
                lp = -Fn.binary_cross_entropy_with_logits(pred, tgt, reduction="none").sum(-1)
            else:
                lp = -Fn.cross_entropy(pred.flatten(0, 1), tgt[..., 0].flatten(0, 1),
                                       reduction="none").view(tgt.shape[:-1])
            res[key] = lp[v].detach().cpu().numpy()
    return res


@torch.no_grad()
def discover_graph(model: "OOCDM", p: Packed, codecs, device, threshold: float = 0.1,
                   n_batches: int = 4, batch_size: int = 128, seed: int = 0,
                   verbose: bool = False) -> tuple["OOCGraph", dict]:
    """One pass of the paper's causal discovery. Returns (graph, {edge: cmi})."""
    model.eval()
    idx = np.random.default_rng(seed).permutation(p.pairs)[: n_batches * batch_size]
    chunks = [idx[i: i + batch_size] for i in range(0, len(idx), batch_size)]
    batches = [make_batch(p, codecs, model, ch, device) for ch in chunks]

    def mean_lp(maskgen):
        acc: dict[str, list] = {}
        for b in batches:
            for k, v in field_logprob(model, model(b, maskgen), b).items():
                acc.setdefault(k, []).append(v)
        return {k: float(np.concatenate(v).mean()) for k, v in acc.items()}

    g = model.graph
    full = mean_lp(FullMask(g))
    out = OOCGraph(model.order, model.field_names, model.target_names, full=False)
    cmis: dict[str, float] = {}

    for c in model.order:                                            # GLOBAL edges: Cl.U -> Ck.V'
        for f in model.field_names[c.name]:
            dropped = mean_lp(DroppedMask(g, drop_global=(c.name, f)))
            for key, lp in full.items():
                cmi = lp - dropped.get(key, lp)
                cmis[f"G {c.name}.{f} -> {key.replace('|', '.')}'"] = cmi
                if cmi > threshold:
                    tc, tf = key.split("|")
                    out.global_matrix[out.gout[(tc, tf)], out.gin[(c.name, f)]] = True

    for c in model.order:                                            # LOCAL edges: C.U -> V'
        for f in model.field_names[c.name]:
            dropped = mean_lp(DroppedMask(g, drop_local=(c.name, f)))
            for t in model.target_names[c.name]:
                key = f"{c.name}|{t}"
                if key not in full:
                    continue
                cmi = full[key] - dropped.get(key, full[key])
                cmis[f"L {c.name}.{f} -> {t}'"] = cmi
                if cmi > threshold:
                    out.local[c.name][model.target_names[c.name].index(t),
                                      model.field_names[c.name].index(f)] = True
    model.train()
    return out, cmis


# --------------------------------------------------------------------------------------------------------
# Event oversampling (docs/LATENT_WORLD_MODEL.md section 23).
#
# NOT from Yu et al. -- the paper's environments have no rare events. `mouse.hunger` decreases by 2.0 on
# EVERY step and its reward is a function of `delta_hunger` and `delta_health`, so there is nothing to miss.
# Ours is the opposite: enemy HP changes on 0.02% of scored steps, a despawn on 0.02%, the room is left on
# 0.03%, and those are what CombatTracker.step() pays out on. An objective drawn uniformly is computed almost
# entirely on frames whose correct answer is "nothing changed".
#
# This repo has already measured the fix. Section 23 took hit-window event detection from 2% to 55% and Baba
# HP removed from 0.04 to 1.15 (real 1.25) with `1 + K*rare + sqrt(K)*move` applied to the BATCH SAMPLER.
#
# It is a SAMPLING weight, not a loss weight, and that distinction is the whole reason to expect it to work
# where the change gate did not: section 13 weighted a gate's BCE by pos_weight up to 200x and backpropagated
# it through the shared trunk, which collapsed every value head (28/47 -> 14/47). Changing which frames the
# model sees does not deform the loss geometry.
# --------------------------------------------------------------------------------------------------------
def event_weights(p: Packed, model: "OOCDM", oversample: float,
                  rare_fields: Sequence[tuple[str, str]] = (("En_Dekubaba", "hp"),
                                                            ("En_Dekubaba", "present"),
                                                            ("En_Dekubaba", "state_idx"),
                                                            ("Link", "hp"),
                                                            ("Room", "scene")),
                  move_field: tuple[str, str] = ("Link", "pos")) -> tuple[np.ndarray, dict]:
    """Per-transition sampling weights `1 + K*rare + sqrt(K)*move`, and the batch composition they produce."""
    t = p.pairs
    rare = np.zeros(len(t), bool)
    for cls, fld in rare_fields:
        if cls not in p.attrs or fld not in p.attrs[cls]:
            continue
        c = next((x for x in model.order if x.name == cls), None)
        if c is None:
            continue
        f, a, m = c.field(fld), p.attrs[cls][fld], p.mask[cls]
        v = m[t] if f.is_mask else (m[t] & m[t + 1])
        e = err(f, prep(f, a[t + 1]), prep(f, a[t]))
        rare |= ((e > 0) & v).any(-1) if e.ndim > 1 else ((e > 0) & v)
    cls, fld = move_field
    c = next(x for x in model.order if x.name == cls)
    f, a, m = c.field(fld), p.attrs[cls][fld], p.mask[cls]
    v = m[t] & m[t + 1]
    e = err(f, prep(f, a[t + 1]), prep(f, a[t]))
    move = ((e > 0) & v).any(-1) if e.ndim > 1 else ((e > 0) & v)
    w = 1.0 + oversample * rare + np.sqrt(max(oversample, 0.0)) * move
    return w, {"rare_frac": float(rare.mean()), "move_frac": float(move.mean()),
               "rare_share": float(w[rare].sum() / w.sum()), "move_share": float(w[move].sum() / w.sum()),
               "n_rare": int(rare.sum())}
