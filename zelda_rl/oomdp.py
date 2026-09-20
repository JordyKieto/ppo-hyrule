"""OOMDP schema over the v3 game state: classes, fields, value types, and the exact column mapping.

Stage 0 of the object-oriented causal dynamics model (Yu, Ruan & Xing, ICML 2024, "Learning Causal Dynamics
Models in Object-Oriented Environments"; reference implementation in `../oocdm`). Nothing here trains, and
nothing here is a neural network. This file only answers: *what are the objects, what are their fields, and
which column of the 363-d state vector is each one?*

WHY, IN ONE PARAGRAPH
---------------------
Every world model in this repo so far (docs/LATENT_WORLD_MODEL.md sections 16-30) flattened the state into one
128-d latent and regressed `z_{t+1} = M(z_t, a_t)`. Six architecture changes failed with the same signature:
the latent moves 0.023-0.044 of a random pair per step, the identity beats every trained model, and position
and events provably compete for the same dimensions (section 24 -- `w_pos` 200 gives trajectory 1.06x and
damage 0%, and detaching does not avoid it). An OOMDP has no bottleneck: each field is predicted in its own
units by its own head, with its own parents, so there is nothing for position and HP to compete over.

WHAT AN OOMDP IS (paper section 3.2)
------------------------------------
A set of objects O, each an instance of a class C. A class declares FIELDS; a field is either a `state` field
(predicted at t+1) or an `action` field (an input, never a target). All instances of a class share one
transition function `p_C(O.S' | O; other objects)`, so the model's parameters and its causal graph are
per-CLASS, not per-object, and neither grows when an actor spawns.

THREE RULES THIS FILE ENFORCES, EACH FROM A SPECIFIC PAST FAILURE
-----------------------------------------------------------------
1. **Every column is derived and asserted, never hand-counted** (AGENTS.md section 3). `assert_partition()`
   checks that the 363 columns of a v3 state partition exactly into (class, field) slots, object masks and
   class discriminators, with nothing dropped silently and nothing counted twice. The `obs` echo slice was
   once hand-written as `slice(23, 27)` and was wrong; that cannot happen here.

2. **The action is a role, not a feature** (AGENTS.md section 5). `role='action'` fields are inputs to other
   fields and are never prediction targets, so a control echo cannot be constructed by accident. Note this is
   about the ACTION (what the agent commands), not the ACTION-STATE (`action_fn_rel`, `melee_state`, which
   routine Link is in) -- those are outcomes and stay `role='state'`, per AGENTS.md section 6.

3. **No row indices survive.** Link's `targeted_row` and `arrow_row` are indices into the actor table, which
   violates the paper's causation symmetry (Assumption 4.1: two instances of a class must be interchangeable).
   They are promoted here to per-actor booleans `is_targeted` / `has_arrow`. This also deletes a bug class:
   today rows are kept meaningful only because the collector assigns a stable slot per actor pointer.

PROVENANCE
----------
Column names and order: `zelda_rl.state_layout` (the single definition, imported, never copied).
Actor id -> class name: `include/tables/actor_table.h` of the OoT decompilation, read from each entry's own
`/* 0xNNNN */` comment. Cross-check: the table gives 0x0055 = En_Dekubaba, matching `combat.DEKU_BABA`, which
was verified live and independently.
`bg_flags` bit names: `include/actor.h` lines 176-185 (`BGCHECKFLAG_*`), which agree with `combat.BG_BITS`.

Categorical cardinalities are NOT written down here. `fit_cardinalities()` measures them from a dataset and
asserts, for the same reason offsets are derived: a guessed enum size is a hand-computed number.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal, Sequence

import numpy as np

from zelda_rl.state_layout import ENEMY_FIELDS, LINK_FIELDS_V3, N_ROWS, ROW_DIM, link_dim, link_index

Role = Literal["state", "action"]
Kind = Literal["real", "angle", "count", "categorical", "bits", "boolean"]

TURNS = 65536.0          # OoT binary angle unit (s16 covers one full turn)


# --------------------------------------------------------------------------------------------------------
# Value types. These are the paper's `VType` (core/vtype.py in ../oocdm): they decide how a field is fed to
# the network AND which distribution its head emits. The kind is the reason a field is predicted the way it
# is, so it is the thing to argue about, not a hyper-parameter.
# --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VType:
    kind: Kind
    size: int = 1              # number of raw state columns this field occupies
    unit: str = ""
    n: int | None = None       # categorical cardinality / bit count; None => measure it
    note: str = ""

    def __str__(self) -> str:
        n = f"[{self.n}]" if self.n is not None else ("[?]" if self.kind == "categorical" else "")
        return f"{self.kind}{n}" + (f" {self.unit}" if self.unit else "") + (f" x{self.size}" if self.size > 1 else "")


def Real(size: int = 1, unit: str = "", note: str = "") -> VType:
    """Continuous. Predicted as a delta with a zero-init head, so an untrained model is EXACTLY do-nothing."""
    return VType("real", size, unit, None, note)


def Angle(unit: Literal["binang", "rad"] = "binang", note: str = "") -> VType:
    """A direction on the circle. Error is wrap-aware; a head emits sin/cos, never a raw scalar."""
    return VType("angle", 1, unit, None, note)


def Count(unit: str = "", note: str = "") -> VType:
    """Small non-negative integer (timers, item counts). Discrete, but ordered and unbounded-ish."""
    return VType("count", 1, unit, None, note)


def Categorical(note: str = "") -> VType:
    """An enum. Cardinality is measured by fit_cardinalities(), not written down here."""
    return VType("categorical", 1, "", None, note)


def Bits(n: int, note: str = "") -> VType:
    """A packed bitfield: n independent booleans in one column."""
    return VType("bits", 1, "", n, note)


def Boolean(note: str = "") -> VType:
    return VType("boolean", 1, "", 2, note)


# --------------------------------------------------------------------------------------------------------
# Fields and classes
# --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Field:
    name: str
    vtype: VType
    role: Role = "state"
    cols: tuple[str, ...] = ()     # source column names in the owning block; () => derived, see DERIVED_*
    is_mask: bool = False          # doubles as the object-presence mask (the paper's `alive`)
    note: str = ""


@dataclass(frozen=True)
class ObjClass:
    name: str
    block: Literal["link", "row"]  # which half of the flat state its columns live in
    fields: tuple[Field, ...]
    actor_ids: tuple[int, ...] = ()   # row classes only: which raw actor ids are instances of this class
    note: str = ""

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.name} has no field {name!r}")

    def state_fields(self) -> tuple[Field, ...]:
        return tuple(f for f in self.fields if f.role == "state")

    def action_fields(self) -> tuple[Field, ...]:
        return tuple(f for f in self.fields if f.role == "action")

    def n_field(self, role: Role | None = None) -> int:
        return len(self.fields if role is None else
                   (self.state_fields() if role == "state" else self.action_fields()))


# --------------------------------------------------------------------------------------------------------
# The v3 schema.
#
# Link's 43 columns split four ways: Room (where we are), Camera (a separate object with its own dynamics,
# driven by Link and by lock-on), two row-index columns promoted to per-actor flags, and Link itself.
# --------------------------------------------------------------------------------------------------------
ROOM = ObjClass("Room", "link", (
    Field("scene", Categorical("scene id; leaving the room is this field changing"), cols=("scene",)),
    Field("room", Categorical("room within the scene"), cols=("room",)),
), note="one instance. The terminal 'exit' event is a transition of Room.scene.")

CAMERA = ObjClass("Camera", "link", (
    Field("yaw", Angle("rad", "atan2(at - eye); the stick is interpreted in THIS frame"), cols=("cam_yaw",)),
    Field("pitch", Angle("rad"), cols=("cam_pitch",)),
    Field("eye", Real(3, "world", "camera position"), cols=("cam_eye_x", "cam_eye_y", "cam_eye_z")),
), note="one instance. Section 16 of the world-model log: without it the world direction of a walk is "
        "unknowable, which capped the transition head's direction accuracy at 0.532 over 16 sectors.")

LINK = ObjClass("Link", "link", (
    # --- outcome: where the body is -------------------------------------------------------------------
    Field("pos", Real(3, "world"), cols=("pos_x", "pos_y", "pos_z")),
    Field("vel", Real(3, "world/step"), cols=("vel_x", "vel_y", "vel_z")),
    Field("speed", Real(1, "world/step"), cols=("speed",)),
    Field("yaw", Angle("binang", "world rotation"), cols=("yaw",)),
    Field("shape_yaw", Angle("binang", "rendered body facing; differs from yaw mid-turn"), cols=("shape_yaw",)),
    # --- outcome: health and inventory ------------------------------------------------------------------
    Field("hp", Categorical("half-hearts x2; classified, not regressed -- damage F1 0.286 -> 0.458 when "
                            "this was made discrete (world-model log section 3)"), cols=("hp",)),
    Field("hp_cap", Categorical("episode constant"), cols=("hp_cap",)),
    Field("held_item", Categorical(), cols=("held_item",)),
    Field("seeds", Count("slingshot ammo"), cols=("seeds",)),
    Field("nuts", Count(), cols=("nuts",)),
    Field("grass", Count(), cols=("grass",)),
    # --- outcome: contact with the world ----------------------------------------------------------------
    Field("bg_flags", Bits(10, "BGCHECKFLAG_* , decomp include/actor.h:176-185"), cols=("bg_flags",)),
    Field("wall_yaw", Angle("binang", "normal of the wall being touched"), cols=("wall_yaw",)),
    Field("floor_height", Real(1, "world"), cols=("floor_height",)),
    Field("floor_pitch", Angle("binang"), cols=("floor_pitch",)),
    Field("dist_interact_wall", Real(1, "world", "section 29: MAE 2.8 against a 2.9 mean-predictor, i.e. "
                                    "the current continuous head has learned nothing here"),
          cols=("dist_interact_wall",)),
    Field("y_dist_to_ledge", Real(1, "world"), cols=("y_dist_to_ledge",)),
    Field("fall_distance", Real(1, "world"), cols=("fall_distance",)),
    # --- ACTION-STATE: which routine Link is in. An outcome of the action, never the action itself -------
    #     (AGENTS.md section 6). These are exactly the fields removed from the ENCODER INPUT as control
    #     echo (bc.CONTROL_ECHO_FEATS); as prediction TARGETS they are legitimate and wanted.
    Field("action_fn_rel", Categorical("Player actionFunc, relative to the overlay base"), cols=("action_fn_rel",)),
    Field("melee_state", Categorical(), cols=("melee_state",)),
    Field("melee_anim", Categorical(), cols=("melee_anim",)),
    Field("ztarget_timer", Count(), cols=("ztarget_timer",)),
    Field("parallel_yaw", Angle("binang", "lock-on facing"), cols=("parallel_yaw",)),
    Field("knockback_type", Categorical(), cols=("knockback_type",)),
    Field("sf1", Bits(32, "Player stateFlags1"), cols=("sf1",)),
    Field("sf2", Bits(32, "Player stateFlags2"), cols=("sf2",)),
    Field("sf3", Bits(8, "Player stateFlags3"), cols=("sf3",)),
    Field("door_type", Categorical(), cols=("door_type",)),
    Field("door_direction", Categorical(), cols=("door_direction",)),
    Field("door_timer", Count(), cols=("door_timer",)),
    # --- ACTION: what the agent commanded. Inputs only. Sourced from the action index, not from the state,
    #     so there is no column for them and no way for them to leak into a target.
    Field("stick", Real(2, "unit vector", "world-relative stick direction"), role="action"),
    Field("stick_none", Boolean("neutral stick"), role="action"),
    Field("btn_A", Boolean(), role="action"),
    Field("btn_B", Boolean(), role="action"),
    Field("btn_Z", Boolean(), role="action"),
    Field("btn_R", Boolean(), role="action"),
    Field("btn_slingshot", Boolean(), role="action"),
    Field("btn_nut", Boolean(), role="action"),
), note="one instance.")

# Row block. `id` and `cat` are the CLASS DISCRIMINATOR (which class an object belongs to), not fields:
# the paper takes class membership as given prior knowledge (section 3.2).
ACTOR_FIELDS: tuple[Field, ...] = (
    Field("present", Boolean("the paper's `alive`; also the object mask"), cols=("present",), is_mask=True),
    Field("pos", Real(3, "world"), cols=("pos_x", "pos_y", "pos_z")),
    Field("vel", Real(3, "world/step"), cols=("vel_x", "vel_y", "vel_z")),
    Field("focus", Real(3, "world", "focus point, used for head tracking and lock-on"),
          cols=("focus_x", "focus_y", "focus_z")),
    Field("hp", Categorical("actor health; En_Dekubaba is 0/1/2"), cols=("hp",)),
    Field("yaw", Angle("binang"), cols=("yaw",)),
    Field("shape_yaw", Angle("binang"), cols=("shape_yaw",)),
    Field("yaw_to_player", Angle("binang"), cols=("yaw_to_player",)),
    Field("fn_rel", Categorical("actionFunc relative to the overlay base"), cols=("fn_rel",)),
    Field("state_idx", Categorical("decoded behaviour state"), cols=("state_idx",)),
    Field("timer", Count(), cols=("timer",)),
    Field("params", Categorical("instance constant set at spawn"), cols=("params",)),
    # Promoted from Link's row-index columns. A row index is not permutation-invariant and so violates the
    # paper's Assumption 4.1; as a per-actor boolean it is.
    Field("is_targeted", Boolean("Link's targeted_row == this row"), cols=()),
    Field("has_arrow", Boolean("Link's arrow_row == this row (Navi's attention arrow)"), cols=()),
)

# Actor id -> class, read from the decompilation's own /* 0xNNNN */ comments in include/tables/actor_table.h.
ACTOR_CLASSES: tuple[ObjClass, ...] = (
    ObjClass("En_Dekubaba", "row", ACTOR_FIELDS, (0x0055,),
             "the combat target: 13 action functions, 9 behaviour states, dies and despawns"),
    ObjClass("En_St", "row", ACTOR_FIELDS, (0x0037,), "Skulltula"),
    ObjClass("En_Sw", "row", ACTOR_FIELDS, (0x0095,), "Skullwalltula"),
)
UNKNOWN_ACTOR_CLASS = ObjClass("Actor", "row", ACTOR_FIELDS, (),
                               "fallback for any actor id not yet given its own class")

# Link columns consumed by another class or promoted to a row flag, rather than being Link's own fields.
PROMOTED_TO_ROW_FLAG = {"targeted_row": "is_targeted", "arrow_row": "has_arrow"}
NO_ROW = -1.0                  # value of targeted_row / arrow_row when nothing is targeted (verified: the
                               # only values these columns take are -1 and a valid slot index)
ROW_DISCRIMINATOR = ("id", "cat")


@dataclass(frozen=True)
class Schema:
    link_classes: tuple[ObjClass, ...]
    actor_classes: tuple[ObjClass, ...]
    fallback: ObjClass = UNKNOWN_ACTOR_CLASS

    @property
    def classes(self) -> tuple[ObjClass, ...]:
        return self.link_classes + self.actor_classes

    def actor_class(self, actor_id: int) -> ObjClass:
        for c in self.actor_classes:
            if actor_id in c.actor_ids:
                return c
        return self.fallback

    def n_field(self, role: Role | None = None) -> int:
        """Total fields across all classes -- the width of the OOCG's global causality matrix."""
        return sum(c.n_field(role) for c in self.classes)

    def edge_budget(self) -> dict[str, int]:
        """Candidate causal edges, class-level. This is the number the paper's whole method is about."""
        local = sum(c.n_field("state") * c.n_field() for c in self.classes)
        glob = self.n_field("state") * self.n_field()
        return {"local": local, "global": glob, "total": local + glob}


SCHEMA_V3 = Schema((LINK, CAMERA, ROOM), ACTOR_CLASSES)


# --------------------------------------------------------------------------------------------------------
# The assertion that makes this file trustworthy: the schema partitions the flat state exactly.
# --------------------------------------------------------------------------------------------------------
def assert_partition(schema: Schema = SCHEMA_V3, width: int = None) -> dict[str, int]:
    """Every column of a v3 state is claimed exactly once. Raises on a gap or an overlap.

    AGENTS.md section 3: an index is derived from the definition and asserted, never hand-counted. The
    definition here is `zelda_rl.state_layout`, imported above.
    """
    from zelda_rl.state_layout import STATE_DIM_V3
    width = STATE_DIM_V3 if width is None else width
    dl = link_dim(width)
    link_names, row_names = LINK_FIELDS_V3[:dl], list(ENEMY_FIELDS)

    claimed_link: dict[str, str] = {}
    for c in schema.link_classes:
        for f in c.fields:
            for col in f.cols:
                if col in claimed_link:
                    raise AssertionError(f"link column {col!r} claimed twice: "
                                         f"{claimed_link[col]} and {c.name}.{f.name}")
                claimed_link[col] = f"{c.name}.{f.name}"
    for col, flag in PROMOTED_TO_ROW_FLAG.items():
        if col in claimed_link:
            raise AssertionError(f"link column {col!r} is both a field and promoted to {flag}")
        claimed_link[col] = f"-> Actor.{flag}"

    missing = [n for n in link_names if n not in claimed_link]
    extra = [n for n in claimed_link if n not in link_names]
    if missing or extra:
        raise AssertionError(f"link block mismatch: unclaimed={missing} unknown={extra}")

    claimed_row: dict[str, str] = {}
    for f in ACTOR_FIELDS:
        for col in f.cols:
            if col in claimed_row:
                raise AssertionError(f"row column {col!r} claimed twice")
            claimed_row[col] = f"Actor.{f.name}"
    for col in ROW_DISCRIMINATOR:
        claimed_row[col] = "class discriminator"
    missing = [n for n in row_names if n not in claimed_row]
    extra = [n for n in claimed_row if n not in row_names]
    if missing or extra:
        raise AssertionError(f"row block mismatch: unclaimed={missing} unknown={extra}")

    cols = len(link_names) + N_ROWS * len(row_names)
    assert cols == width, f"schema covers {cols} columns, state is {width}"
    return {"width": width, "link_cols": len(link_names), "row_cols": len(row_names), "n_rows": N_ROWS}


# --------------------------------------------------------------------------------------------------------
# Decomposition: flat state -> objects with attributes. This is the ONLY place the flat layout is indexed.
# --------------------------------------------------------------------------------------------------------
@dataclass
class ObjectGroup:
    """Every instance of one class, over a whole episode.

    attrs[field] has shape (T, n_obj, size); mask has shape (T, n_obj) and is False where the object does
    not exist at that step (the paper's varying instance count -- AILL averages per instance, so a despawned
    Baba simply stops contributing).
    """
    cls: ObjClass
    labels: list[str]
    attrs: dict[str, np.ndarray]
    mask: np.ndarray

    @property
    def n_obj(self) -> int:
        return len(self.labels)


def decompose(state: np.ndarray, actions: np.ndarray | None = None,
              schema: Schema = SCHEMA_V3) -> dict[str, ObjectGroup]:
    """(T, width) flat state [+ (T,) action indices] -> {class name: ObjectGroup}.

    Actions are attached to Link as `role='action'` fields via `dynamics.factor_actions`, so the stick
    direction is a unit vector and each button is its own flag -- a direction seen with one button
    generalises to the same direction with another (world-model log section 27).
    """
    state = np.asarray(state, np.float32)
    T, width = state.shape
    dl = link_dim(width)
    lk, rb = state[:, :dl], state[:, dl:].reshape(T, N_ROWS, ROW_DIM)
    LIdx, RIdx = link_index(width), {k: i for i, k in enumerate(ENEMY_FIELDS)}

    out: dict[str, ObjectGroup] = {}

    # --- single-instance classes out of the Link block ---------------------------------------------------
    for c in schema.link_classes:
        attrs = {}
        for f in c.fields:
            if f.role == "action" or not f.cols:
                continue
            attrs[f.name] = lk[:, [LIdx[col] for col in f.cols]][:, None, :]      # (T, 1, size)
        out[c.name] = ObjectGroup(c, [c.name.lower()], attrs, np.ones((T, 1), bool))

    if actions is not None:
        from zelda_rl.dynamics import FACTORED_NAMES, factor_actions
        from zelda_rl.env import N_COMBAT_ACTIONS_V5
        a = np.asarray(actions, np.int64)
        assert len(a) == T, f"{len(a)} actions for {T} states"
        fa = factor_actions(N_COMBAT_ACTIONS_V5)[a]                               # (T, FACTORED_DIM)
        col = {n: i for i, n in enumerate(FACTORED_NAMES)}
        g = out[LINK.name]
        for f in LINK.action_fields():
            names = ("stick_x", "stick_z") if f.name == "stick" else (f.name,)
            g.attrs[f.name] = fa[:, [col[n] for n in names]][:, None, :]

    # --- actor classes out of the row block --------------------------------------------------------------
    present = rb[:, :, RIdx["present"]] > 0.5
    ever = present.any(0)
    ids = np.zeros(N_ROWS, np.int64)
    for s in range(N_ROWS):
        if ever[s]:
            seen = np.unique(rb[present[:, s], s, RIdx["id"]].astype(np.int64))
            assert len(seen) == 1, f"row {s} holds more than one actor id {seen}: row identity is not stable"
            ids[s] = seen[0]

    tgt, arw = lk[:, LIdx["targeted_row"]].astype(np.int64), lk[:, LIdx["arrow_row"]].astype(np.int64)
    by_cls: dict[str, list[int]] = {}
    for s in range(N_ROWS):
        if ever[s]:
            by_cls.setdefault(schema.actor_class(int(ids[s])).name, []).append(s)

    for cname, slots in by_cls.items():
        c = next(k for k in schema.actor_classes + (schema.fallback,) if k.name == cname)
        sel = rb[:, slots, :]                                                     # (T, n_obj, ROW_DIM)
        attrs = {}
        for f in c.fields:
            if f.cols:
                attrs[f.name] = sel[:, :, [RIdx[col] for col in f.cols]]
        slot_arr = np.asarray(slots)[None, :]
        attrs["is_targeted"] = (tgt[:, None] == slot_arr).astype(np.float32)[:, :, None]
        attrs["has_arrow"] = (arw[:, None] == slot_arr).astype(np.float32)[:, :, None]
        out[cname] = ObjectGroup(c, [f"row{s:02d}" for s in slots], attrs, present[:, slots])
    return out


# --------------------------------------------------------------------------------------------------------
# Cardinalities are measured, not declared.
# --------------------------------------------------------------------------------------------------------
def fit_cardinalities(groups: Iterable[dict[str, ObjectGroup]],
                      schema: Schema = SCHEMA_V3) -> tuple[Schema, dict[str, list[float]]]:
    """Scan decomposed episodes and fill every Categorical/Count `n` with the observed value set.

    Returns the resolved schema and {'Class.field': sorted observed values}. A field whose observed set has
    one element is an episode or instance CONSTANT and is flagged as such by the caller -- it is trivially
    predictable and must never be averaged into an aggregate metric (AGENTS.md section 2).
    """
    seen: dict[str, set[float]] = {}
    for g in groups:
        for cname, grp in g.items():
            for f in grp.cls.fields:
                if f.vtype.kind not in ("categorical", "count", "boolean") or f.name not in grp.attrs:
                    continue
                v = grp.attrs[f.name][grp.mask]
                if v.size:
                    seen.setdefault(f"{cname}.{f.name}", set()).update(np.unique(v).tolist())

    def resolve(c: ObjClass) -> ObjClass:
        fs = []
        for f in c.fields:
            vals = seen.get(f"{c.name}.{f.name}")
            if vals is not None and f.vtype.n is None:
                f = replace(f, vtype=replace(f.vtype, n=len(vals)))
            fs.append(f)
        return replace(c, fields=tuple(fs))

    resolved = Schema(tuple(resolve(c) for c in schema.link_classes),
                      tuple(resolve(c) for c in schema.actor_classes), resolve(schema.fallback))
    return resolved, {k: sorted(v) for k, v in seen.items()}


def recompose(groups: dict[str, ObjectGroup], width: int, schema: Schema = SCHEMA_V3) -> np.ndarray:
    """Objects -> the flat (T, width) state. The inverse of `decompose`, and only exists to be asserted."""
    any_g = next(iter(groups.values()))
    T = len(any_g.mask)
    dl = link_dim(width)
    LIdx, RIdx = link_index(width), {k: i for i, k in enumerate(ENEMY_FIELDS)}
    lk, rb = np.zeros((T, dl), np.float32), np.zeros((T, N_ROWS, ROW_DIM), np.float32)

    for c in schema.link_classes:
        g = groups[c.name]
        for f in c.fields:
            if f.role == "action" or not f.cols:
                continue
            lk[:, [LIdx[col] for col in f.cols]] = g.attrs[f.name][:, 0, :]

    for col, flag in PROMOTED_TO_ROW_FLAG.items():
        idx = np.full(T, NO_ROW, np.float32)
        for cname, g in groups.items():
            if g.cls.block != "row":
                continue
            hit = g.attrs[flag][:, :, 0] > 0.5                       # (T, n_obj)
            for j, lab in enumerate(g.labels):
                idx[hit[:, j]] = float(int(lab[3:]))                 # labels are "rowNN"
        lk[:, LIdx[col]] = idx

    for cname, g in groups.items():
        if g.cls.block != "row":
            continue
        slots = [int(lab[3:]) for lab in g.labels]
        for f in g.cls.fields:
            for k, col in enumerate(f.cols):
                rb[:, slots, RIdx[col]] = g.attrs[f.name][:, :, k]
        for j, s_ in enumerate(slots):
            aid = float(g.cls.actor_ids[0] if g.cls.actor_ids else 0)
            rb[:, s_, RIdx["id"]] = np.where(g.mask[:, j], aid, 0.0)
    return np.concatenate([lk, rb.reshape(T, -1)], 1)


def assert_roundtrip(state: np.ndarray, actions: np.ndarray | None = None,
                     schema: Schema = SCHEMA_V3, ignore: Sequence[str] = ("cat",)) -> None:
    """decompose -> recompose returns the original state exactly.

    A column partition proves nothing was forgotten in the BOOKKEEPING; this proves nothing is lost in the
    DECOMPOSITION. `cat` is excluded because it is a class discriminator that `recompose` does not carry
    back (id alone determines the class), so it is checked separately for constancy.
    """
    state = np.asarray(state, np.float32)
    back = recompose(decompose(state, actions, schema), state.shape[-1], schema)
    dl = link_dim(state.shape[-1])
    RIdx = {k: i for i, k in enumerate(ENEMY_FIELDS)}
    skip = np.zeros(state.shape[-1], bool)
    for col in ignore:
        skip[dl + np.arange(N_ROWS) * ROW_DIM + RIdx[col]] = True
    bad = np.nonzero((state != back).any(0) & ~skip)[0]
    if len(bad):
        names = LINK_FIELDS_V3[: dl] + [f"row{(i - dl) // ROW_DIM:02d}.{ENEMY_FIELDS[(i - dl) % ROW_DIM]}"
                                        for i in range(dl, state.shape[-1])]
        raise AssertionError(f"decompose/recompose lost {len(bad)} columns: "
                             f"{[names[i] for i in bad[:12]]}{' ...' if len(bad) > 12 else ''}")


# --------------------------------------------------------------------------------------------------------
# Field metrics. These live HERE, beside the Field definitions, because both the do-nothing census and the
# model evaluation must score a field the same way. state_layout.py's docstring records that every loader bug
# in this project came from two copies of one definition drifting apart; a metric is a definition.
# --------------------------------------------------------------------------------------------------------
def wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def unpack_bits(v: np.ndarray, n: int) -> np.ndarray:
    return ((v.astype(np.int64) >> np.arange(n)) & 1).astype(np.float32)


def prep(f: Field, v: np.ndarray) -> np.ndarray:
    """Raw column values -> the representation `err` works on."""
    k = f.vtype.kind
    if k == "bits":
        return unpack_bits(v, f.vtype.n)
    if k == "angle":
        return v * (2 * np.pi / TURNS) if f.vtype.unit == "binang" else v.astype(np.float32)
    return v.astype(np.float32)


def err(f: Field, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample error between two PREPARED values, in the field's own units. Shape (n,)."""
    k = f.vtype.kind
    if k == "angle":
        return np.abs(wrap(a - b)).mean(-1) * 180.0 / np.pi          # degrees
    if k in ("real", "count"):
        return np.abs(a - b).mean(-1)                                 # world units / items
    return (a != b).astype(np.float32).mean(-1)                       # error rate (enum, boolean, bits)


def unit(f: Field) -> str:
    return {"angle": "deg", "real": f.vtype.unit or "abs", "count": "abs"}.get(f.vtype.kind, "rate")


def describe(schema: Schema = SCHEMA_V3) -> str:
    lines = [f"OOMDP schema: {len(schema.classes)} classes, {schema.n_field()} fields "
             f"({schema.n_field('state')} state, {schema.n_field('action')} action)"]
    b = schema.edge_budget()
    lines.append(f"class-level causal edges: {b['local']} local + {b['global']} global = {b['total']} "
                 f"(independent of how many actors exist)")
    for c in schema.classes:
        lines.append(f"\n  class {c.name}  [{c.n_field('state')} state, {c.n_field('action')} action]"
                     + (f"  ids={[hex(i) for i in c.actor_ids]}" if c.actor_ids else ""))
        for f in c.fields:
            tag = "ACT" if f.role == "action" else ("MASK" if f.is_mask else "   ")
            lines.append(f"    {tag} {f.name:<20} {str(f.vtype):<22} {f.vtype.note or f.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(assert_partition(), "\n")
    print(describe())
