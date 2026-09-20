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

THE CHANGE GATE (docs/CAUSAL_WORLD_MODEL.md section 12.1)
---------------------------------------------------------
Every field head is split in two:

  gate   1 logit    did this field change at all this step?
  value  n outputs  GIVEN that it changed, by how much / to what?

and the prediction is `gate < threshold ? v_t : apply(v_t, value)`.

This is not a refinement, it is the fix for the measured failure of the first Stage 1 run. Without it a head
regresses a delta directly, so a delta that is zero on ~90% of steps gets smeared across all of them: the
6-epoch run emitted **1.41 world units of spurious `Link.pos` motion on steps where Link did not move**, and
seven rare-event heads (`En_Dekubaba.hp` among them) sat at exactly 0.000 because "never changes" is the
regression optimum. Section 29 of the latent trail already had the evidence that this decomposition works --
`TransitionHead`'s `moved` logit reached AP 0.435, 6.8x base.

Three consequences, each deliberate:
  - the VALUE loss is computed on ALL steps by default. Restricting it to changed steps was the original
    design and `scripts/probe_gate.py` measured it as HARMFUL on its own terms: at matched epochs on one
    subset it took the fields whose value head beats do-nothing GIVEN A CHANGE from 28/47 down to 16/47, and
    flipped Link.pos from +0.066 to -0.106. The zeros are apparently teaching the head something about scale
    and direction that it does not recover from changed steps alone. `value_on_changed=True` restores the
    old behaviour for anyone who wants to re-test it;
  - the GATE loss is a per-field BCE with `pos_weight` taken from that field's MEASURED change rate, which is
    where rare-event supervision now lives (section 23 of the latent trail, applied per field);
  - at init the gate bias is GATE_INIT_BIAS, so p(changed) ~ 3e-4 and the untrained model is do-nothing
    EXACTLY -- more exactly than before, since the angle codec's float round-trip is bypassed entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from zelda_rl.oomdp import (SCHEMA_V3, TURNS, Field, ObjClass, ObjectGroup, Schema, err,
                            prep, unit, unpack_bits, wrap)

IDENTITY_LOGIT = 8.0        # logit margin that makes argmax == the current value at init
GATE_INIT_BIAS = -8.0       # change-gate bias at init: p(changed) ~ 3e-4, so the model starts as do-nothing
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
        """(cur, nxt) raw -> the head's training target, in the head's own parameterisation."""
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
               changed_only: bool = True) -> dict[str, dict[str, FieldCodec]]:
    """Measure every codec's constants from decomposed episodes. Returns {class: {field: codec}}.

    Scales come from the data the model will see, restricted to steps where the object EXISTS -- a despawned
    actor's zeroed row must not drag a position mean toward the origin.

    `changed_only` sets WHICH steps `dstd` is measured over, and it must match how the value head is trained.
    With the change gate on, the value head only ever sees steps where the field changed, and the delta std
    there is 3.4x to 50x larger than over all steps (Link.pos 2.70 -> 9.11, y_dist_to_ledge 7.9 -> 400).
    Measuring it over all steps -- which are ~90% zeros -- left targets reaching |152| and a Gaussian NLL in
    the thousands, which is exactly what the first gated run did before this was fixed. With the gate off the
    value head sees every step, so the all-steps scale is the right one.

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


class ClassModule(nn.Module):
    """Everything owned by one class: field embeddings, the object encoding, and one head per state field."""

    def __init__(self, cls: ObjClass, codecs: dict[str, FieldCodec], fields: Sequence[str],
                 targets: Sequence[str], d: int, d_h: int, d_k: int, d_v: int, d_dec: int):
        super().__init__()
        self.cls_name, self.fields, self.targets = cls.name, list(fields), list(targets)
        self.embed = HeteroEmbed([codecs[f].input_dim if f in codecs else 2 if f == "stick" else 1
                                  for f in self.fields], d)
        flat = len(self.fields) * d
        n_t = len(self.targets)
        self.f_local = MultiLinear(n_t, flat, d_h)              # per TARGET FIELD: the query's source
        self.f_obj = nn.Linear(flat, d_h)                       # per OBJECT: what other classes attend to
        self.fq, self.fk, self.fv = MultiLinear(n_t, d_h, d_k), nn.Linear(d_h, d_k), nn.Linear(d_h, d_v)
        self.dec = nn.Sequential()
        self.dec1, self.dec2 = MultiLinear(n_t, d_h + d_v, d_dec), MultiLinear(n_t, d_dec, d_dec)
        self.heads, self.gates = nn.ModuleList(), nn.ModuleList()
        for t in self.targets:                                  # zero-init: an untrained head is do-nothing
            h = nn.Linear(d_dec, codecs[t].n_out)
            nn.init.zeros_(h.weight), nn.init.zeros_(h.bias)
            self.heads.append(h)
            g = nn.Linear(d_dec, 1)                             # "did this field change at all this step?"
            nn.init.zeros_(g.weight), nn.init.constant_(g.bias, GATE_INIT_BIAS)
            self.gates.append(g)

    def encode(self, xs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.embed(xs).flatten(-2)                          # (B, n_obj, n_field*d)
        local = self.f_local(e.unsqueeze(-2).expand(*e.shape[:-1], len(self.targets), e.shape[-1]))
        return Fn.leaky_relu(local), Fn.leaky_relu(self.f_obj(e))


class OOCDM(nn.Module):
    """p(S' | S, A) factored over (class, field), with class-shared parameters and inter-object attention.

    Deviation from the paper, recorded deliberately: its `f_global` produces a separate encoding of each
    object FOR EACH of the 77 global state fields, so that masking can give every target field its own
    parents. At the full graph G1 there is no masking, so Stage 1 uses ONE encoding per object as the
    attention key/value and keeps the per-target-field projection only on the query side. That is ~77x less
    compute for the question Stage 1 is asking; Stage 2 needs the full form and will restore it.
    """

    def __init__(self, codecs: dict[str, dict[str, FieldCodec]], capacity: dict[str, int],
                 schema: Schema = SCHEMA_V3, targets: dict[str, list[str]] | None = None,
                 d: int = 16, d_h: int = 64, d_k: int = 32, d_v: int = 64, d_dec: int = 128,
                 use_gate: bool = True, value_on_changed: bool = False, gate_detach: bool = True):
        super().__init__()
        self.schema, self.codecs, self.capacity, self.use_gate = schema, codecs, capacity, use_gate
        self.value_on_changed, self.gate_detach = value_on_changed, gate_detach
        self.pos_weight: dict[str, torch.Tensor] = {}      # filled by set_change_rates()
        self.order = [c for c in schema.classes if capacity.get(c.name, 0) > 0]
        self.field_names, self.target_names = {}, {}
        mods = {}
        for c in self.order:
            fs = [f.name for f in c.fields if f.name in codecs[c.name] or f.role == "action"]
            ts = list(targets[c.name]) if targets else [f.name for f in c.state_fields()
                                                        if f.name in codecs[c.name]]
            self.field_names[c.name], self.target_names[c.name] = fs, ts
            mods[c.name] = ClassModule(c, codecs[c.name], fs, ts, d, d_h, d_k, d_v, d_dec)
        self.mods = nn.ModuleDict(mods)
        self.log_scale = nn.ParameterDict({
            f"{c.name}|{t}": nn.Parameter(torch.zeros(codecs[c.name][t].n_out))
            for c in self.order for t in self.target_names[c.name]
            if codecs[c.name][t].kind in ("real", "count")})

    def forward(self, batch: dict) -> dict[str, dict[str, torch.Tensor]]:
        """batch['x'][cls][field] (B, n_obj, in_dim); batch['mask'][cls] (B, n_obj);
        batch['raw'][cls][field] (B, n_obj, size) for the identity residual.

        Returns {'value': {'Cls|field': (B, n_obj, n_out)}, 'gate': {'Cls|field': (B, n_obj)}}.
        """
        local, keys, vals, obj_mask = {}, [], [], []
        for c in self.order:
            m = self.mods[c.name]
            l, o = m.encode([batch["x"][c.name][f] for f in self.field_names[c.name]])
            local[c.name] = l
            keys.append(m.fk(o)), vals.append(m.fv(o))
            obj_mask.append(batch["mask"][c.name])
        K, V = torch.cat(keys, 1), torch.cat(vals, 1)                     # (B, n_obj_all, d)
        M = torch.cat(obj_mask, 1)                                        # (B, n_obj_all)

        out: dict[str, torch.Tensor] = {}
        gate: dict[str, torch.Tensor] = {}
        base = 0
        for c in self.order:
            m, n_obj = self.mods[c.name], self.capacity[c.name]
            q = m.fq(local[c.name])                                       # (B, n_obj_c, n_t, d_k)
            att = torch.einsum("botk,bjk->botj", q, K) / q.shape[-1] ** 0.5
            idx = torch.arange(n_obj, device=att.device)
            self_edge = torch.zeros(n_obj, M.shape[1], dtype=torch.bool, device=att.device)
            self_edge[idx, base + idx] = True                             # object i is its own key
            vis = M[:, None, None, :] & ~self_edge[None, :, None, :]      # no attention to itself
            vis = vis.expand_as(att)
            att = att.masked_fill(~vis, float("-inf"))
            att = torch.where(vis.any(-1, keepdim=True), att.softmax(-1), torch.zeros_like(att))
            h = torch.cat([local[c.name], torch.einsum("botj,bjv->botv", att, V)], -1)
            h = Fn.leaky_relu(m.dec2(Fn.leaky_relu(m.dec1(h))))           # (B, n_obj_c, n_t, d_dec)
            for i, t in enumerate(self.target_names[c.name]):
                key = f"{c.name}|{t}"
                out[key] = self._identity(c.name, t, m.heads[i](h[:, :, i]), batch["raw"][c.name][t])
                # The gate READS the representation but must not SHAPE it. Its BCE carries pos_weight up
                # to 200x, and backpropagating that through the shared trunk collapsed the value head:
                # at matched epochs, adding the gate loss took value heads beating do-nothing from 28/47 to
                # 14/47 and Link.pos's predicted magnitude from 0.418 to 0.119 of the truth. Detaching keeps
                # the classifier (AP 0.839 on Link.pos) without letting it repurpose the trunk.
                gh = h[:, :, i].detach() if self.gate_detach else h[:, :, i]
                gate[key] = m.gates[i](gh)[..., 0]
            base += n_obj
        return {"value": out, "gate": gate} if self.use_gate else {"value": out}

    def set_change_rates(self, rates: dict[str, float], cap: float = 200.0) -> dict[str, float]:
        """`pos_weight` for each gate, from that field's MEASURED change rate on the training split.

        This is where per-field rare-event supervision lives (docs/CAUSAL_WORLD_MODEL.md 12.3). A field that
        changes on 0.057% of object-steps -- En_Dekubaba.hp -- gets its positives weighted ~1750x, which is
        section 23's oversampling expressed per field instead of as one shared batch policy. Capped, because
        an unbounded weight on a field that changed twice in the dataset is noise amplification.
        """
        self.pos_weight = {k: torch.tensor(min((1 - r) / r, cap), dtype=torch.float32)
                           for k, r in rates.items() if r > 0}
        return {k: float(v) for k, v in self.pos_weight.items()}

    def _identity(self, cname: str, t: str, raw_out: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """Add the residual-from-identity offset, so a zero-init head predicts EXACTLY no change."""
        cod = self.codecs[cname][t]
        k = cod.kind
        if k in ("real", "count"):
            return raw_out
        if k == "angle":
            bias = torch.zeros_like(raw_out)
            bias[..., 0] = 1.0                                            # (cos, sin) = (1, 0) => delta 0
            return raw_out + bias
        if k == "bits":
            b = ((cur.long() >> torch.arange(cod.field.vtype.n, device=cur.device)) & 1).float()
            return raw_out + IDENTITY_LOGIT * (2 * b - 1)
        vals = torch.as_tensor(cod.values, device=cur.device)
        eq = (cur == vals).float()                                        # (B, n_obj, n_values)
        # Codecs are fitted on TRAIN ONLY, so a value first seen at test time matches no slot. The identity
        # bias then has nowhere to go and the head falls back to class 0 -- which is how an untrained model
        # scored skill -33 on Link.hp. `FieldCodec.apply` already maps the UNKNOWN slot back to the current
        # value, so that slot is where the identity belongs whenever nothing else matches.
        unknown = (1.0 - eq.sum(-1, keepdim=True)).clamp(0.0, 1.0)
        return raw_out + IDENTITY_LOGIT * torch.cat([eq, unknown], -1)


# --------------------------------------------------------------------------------------------------------
# Batching and the per-field loss
# --------------------------------------------------------------------------------------------------------
def make_batch(p: Packed, codecs: dict[str, dict[str, FieldCodec]], model: "OOCDM",
               idx: np.ndarray, device) -> dict:
    """Frames t -> the tensors `OOCDM.forward` and `field_losses` consume. Targets are read at t+1.

    Validity is per field, not global: every field needs the object to exist at BOTH t and t+1, except
    `present` itself, whose despawn (1 -> 0) is precisely the thing to predict.
    """
    t = np.asarray(idx)
    x, raw, mask, y, valid, chg = {}, {}, {}, {}, {}, {}
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)   # noqa: E731
    for c in model.order:
        cn = c.name
        at, m = p.attrs[cn], p.mask[cn]
        here, there = m[t], m[t + 1]
        mask[cn] = T(here, torch.bool)
        x[cn], raw[cn], y[cn], valid[cn], chg[cn] = {}, {}, {}, {}, {}
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
            # The gate's target, in the SAME metric the evaluation uses (zelda_rl.oomdp.err), so "changed"
            # can never mean one thing to the loss and another to the report.
            chg[cn][tgt] = T(err(fld, prep(fld, nxt), prep(fld, cur)) > 0, torch.float32)
    return {"x": x, "raw": raw, "mask": mask, "y": y, "valid": valid, "chg": chg, "t": t}


def field_losses(model: "OOCDM", out: dict, batch: dict) -> dict[str, torch.Tensor]:
    """One scalar per (class, field), each already averaged per instance -- the paper's AILL, per field.

    Averaging PER INSTANCE rather than per batch element is what makes the objective invariant to how many
    actors are alive, which is the whole point of the object-oriented formulation.

    With the gate on, each field contributes TWO terms: a `gate:` BCE over "did this change", weighted by the
    field's measured change rate, and a value term computed ONLY on the steps where it truly changed.
    """
    losses = {}
    # `forward` always wraps in {"value": ...} and adds "gate" only when the gate is on. Keying the unwrap
    # off the PRESENCE OF "gate" rather than off "value" passed the whole wrapper through as the values dict
    # when the gate was off, and --no-gate died with KeyError on the first field.
    vals = out["value"] if (isinstance(out, dict) and "value" in out) else out
    gates = out.get("gate") if isinstance(out, dict) else None
    for c in model.order:
        for t in model.target_names[c.name]:
            key = f"{c.name}|{t}"
            cod, pred, tgt = model.codecs[c.name][t], vals[key], batch["y"][c.name][t]
            v = batch["valid"][c.name][t]
            if not bool(v.any()):
                continue
            if gates is not None:
                ch = batch["chg"][c.name][t]
                pw = model.pos_weight.get(key)
                losses[f"gate:{key}"] = Fn.binary_cross_entropy_with_logits(
                    gates[key][v], ch[v], pos_weight=pw.to(gates[key].device) if pw is not None else None)
                if model.value_on_changed:
                    v = v & (ch > 0.5)         # "how much, GIVEN it moved" -- MEASURED HARMFUL, see below
                    if not bool(v.any()):
                        continue
            k = cod.kind
            if k in ("real", "count"):
                s = model.log_scale[f"{c.name}|{t}"]
                e = 0.5 * ((pred - tgt) / s.exp()) ** 2 + s                   # Gaussian NLL, learned scale
                l = e.mean(-1)[v].mean()
            elif k == "angle":
                u = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                l = (1 - (u * tgt).sum(-1))[v].mean()                         # 1 - cos(pred - true)
            elif k == "bits":
                l = Fn.binary_cross_entropy_with_logits(pred, tgt, reduction="none").mean(-1)[v].mean()
            else:
                l = Fn.cross_entropy(pred[v], tgt[..., 0][v], reduction="mean")
            losses[key] = l
    return losses


@torch.no_grad()
def predict_raw(model: "OOCDM", out: dict, batch: dict,
                thresholds: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """Head outputs -> predicted RAW field values, so every metric is in the field's own units.

    Where the gate says the field did not change, the CURRENT value is returned unchanged -- the prediction
    is then exactly do-nothing for that (object, step), which is what stops a near-zero delta being smeared
    across the ~90% of steps on which nothing happens.
    """
    # `forward` always wraps in {"value": ...} and adds "gate" only when the gate is on. Keying the unwrap
    # off the PRESENCE OF "gate" rather than off "value" passed the whole wrapper through as the values dict
    # when the gate was off, and --no-gate died with KeyError on the first field.
    vals = out["value"] if (isinstance(out, dict) and "value" in out) else out
    gates = out.get("gate") if isinstance(out, dict) else None
    res = {}
    for c in model.order:
        for t in model.target_names[c.name]:
            key = f"{c.name}|{t}"
            cod = model.codecs[c.name][t]
            cur = batch["raw"][c.name][t].cpu().numpy()
            pred = cod.apply(cur, vals[key].float().cpu().numpy())
            if gates is not None:
                thr = (thresholds or {}).get(key, 0.5)
                fire = torch.sigmoid(gates[key]).cpu().numpy() >= thr
                pred = np.where(fire[..., None], pred, cur)
            res[key] = pred
    return res


@torch.no_grad()
def assert_identity_init(model: "OOCDM", batch: dict, tol: float = 1e-3) -> None:
    """An untrained model must be EXACTLY do-nothing (AGENTS.md section 1: the identity is the floor, and
    a zero-init residual predictor IS the identity). Asserted numerically, not argued."""
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


def change_rates(p: Packed, model: "OOCDM") -> dict[str, float]:
    """Fraction of scored (object, step) pairs on which each field changes, measured on a split."""
    t = p.pairs
    out = {}
    for c in model.order:
        for tgt in model.target_names[c.name]:
            f, a, m = c.field(tgt), p.attrs[c.name][tgt], p.mask[c.name]
            here, there = m[t], m[t + 1]
            v = here if f.is_mask else (here & there)
            if not v.any():
                continue
            cur, nxt = a[t][v], a[t + 1][v]
            out[f"{c.name}|{tgt}"] = float((err(f, prep(f, nxt), prep(f, cur)) > 0).mean())
    return out


@torch.no_grad()
def calibrate_thresholds(model: "OOCDM", p: Packed, codecs, device, batch_size: int = 512,
                         cap: int = 40000,
                         grid=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)) -> dict[str, float]:
    """Pick each gate's firing threshold on the TUNE split, minimising that field's own raw error.

    Not a hyper-parameter and not 0.5 by default: section 3 of the latent trail took damage F1 from 0.746 to
    0.906 purely by calibrating a threshold on a held-out split, and the decision here is the same one --
    firing costs spurious motion, not firing costs a missed event, and the exchange rate is per field.

    The grid's last entry, 1.01, means NEVER FIRE. It matters more than the others: it lets a field whose
    value head cannot beat do-nothing fall back to exact do-nothing, so after calibration the identity is a
    floor for every field AFTER training and not only at init. A field pinned at 1.01 is reported as skill
    0.000, which is the truth about it -- and a threshold sitting at either end of the grid is itself a
    signal that the gate carries no information for that field.
    """
    if not model.use_gate:
        return {}
    model.eval()
    idx = p.pairs if len(p.pairs) <= cap else p.pairs[:: max(1, len(p.pairs) // cap)]
    tot = {}
    for s0 in range(0, len(idx), batch_size):
        b = make_batch(p, codecs, model, idx[s0: s0 + batch_size], device)
        o = model(b)
        for c in model.order:
            for t in model.target_names[c.name]:
                key = f"{c.name}|{t}"
                f = c.field(t)
                v = b["valid"][c.name][t].cpu().numpy()
                if not v.any():
                    continue
                cur = b["raw"][c.name][t].cpu().numpy()
                nxt = p.attrs[c.name][t][b["t"] + 1]
                val = codecs[c.name][t].apply(cur, o["value"][key].float().cpu().numpy())
                pr = torch.sigmoid(o["gate"][key]).cpu().numpy()
                e_id = err(f, prep(f, nxt[v]), prep(f, cur[v]))
                e_md = err(f, prep(f, nxt[v]), prep(f, val[v]))
                acc = tot.setdefault(key, np.zeros(len(grid)))
                for i, g in enumerate(grid):
                    fire = pr[v] >= g
                    acc[i] += float(np.where(fire, e_md, e_id).sum())
    model.train()
    return {k: float(grid[int(np.argmin(a))]) for k, a in tot.items()}
