"""Roll the model over windows that actually CONTAIN events, and check it reproduces them.

    .venv/bin/python scripts/eval_event_windows.py checkpoints/dynamics/joint_v15_exitev.pt

Why this exists: every earlier rollout evaluation averaged over whole held-out episodes, and those are
event-poor. On that set "assume nothing ever changes" scores 0.92-1.00 on behaviour, live-Baba count and Link
HP, because almost nothing happens -- so a model that predicts a frozen world matches the control and looks
fine. Across 12 held-out episodes the real totals were 0.33 damage and 0.25 kills; most contain no kill at all.

So select the windows where something happens and score only those. A window qualifies if, within H steps of
its start, the REAL trajectory contains the event being tested:

  hit       an actor's HP falls (Link lands a sword hit)
  link_hit  Link's HP falls
  walk      Link travels more than `--walk-units` world units
  kill      CombatTracker books a defeat: a target's HP reaches 0 while it is still present
  block     CombatTracker books a block or a dodge (a lunge->pull_back cycle survived)
  lockon    Link acquires a lock-on target (`targeted_row` goes from -1 to a valid row)
  jump      Link leaves the ground (bgCheckFlags GROUND clears)
  roll      Link enters the `roll` action state
  collision Link touches a wall (bgCheckFlags WALL, bit 3)

Each window is rolled from its own real start state with the real action sequence, so the comparison is
like-for-like with the game. Reported per event type and horizon:

  detect    did the dream produce the event at all -- and the SAME number for the real trajectory and for
            identity, computed by the SAME function, so a broken window definition is visible as a real
            column that is not ~1.00
  magnitude how much HP / distance / how many event steps, dream vs real
  identity  the same numbers for "nothing ever changes", which is the bar to beat

A model can only be said to model attacks or walking if it beats identity HERE, not on the full-episode mean.

Three rules this file exists to enforce, each of them a bug that once produced a wrong conclusion:

  1. ONE event definition per kind, applied three times -- to the dream, to the real trajectory and to
     identity (`event()` below). The old table printed the *Baba* detection column for every row, so the
     `link_hit` row read "0% real detection" on windows selected for a real Link HP drop; the Link number was
     in the JSON all along at 1.00. A metric whose real column is not ~1.00 is measuring the wrong thing.
  2. Every rollout is compared against ITS OWN start state. The decoded seed does not always match truth (a
     Baba at 1 HP can decode as 2; the airborne bit can decode wrong on the seed frame), and mixing the two
     once produced a fake "the dream heals enemies" result. For the boolean kinds this means a dream whose
     seed frame already has the flag set cannot register an onset -- so `seed_mismatch` is reported too,
     which says whether a miss is a rollout failure or a seed-decode failure.
  3. Windows whose first real step exceeds `MAX_STEP_UNITS` are dropped: that is faster than Link can move
     and means the savestate had not settled. One such episode once dragged a mean from 0.0 to 46.2.

The boolean kinds (lockon/jump/roll/collision) are selected as ONSETS -- the flag is off at the window start
and turns on inside the window -- so identity scores exactly 0 on them by construction, the same bar the
HP and travel kinds face.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zelda_rl.bc as bc  # noqa: E402
from zelda_rl.bc import enc_arch, BCNet, Norm, StateDecoder, UnifiedDecoder, decode_targets, featurise, wants_v2  # noqa: E402
from zelda_rl.combat import (BG_DECODE_BITS, BG_GROUND, BG_WALL, CombatConfig,  # noqa: E402
                             CombatTracker)
from zelda_rl.dynamics import TransitionHead, load_predictor  # noqa: E402
from zelda_rl.dream_state import (DreamTracker, apply_transition, decoded_states_from_raw,  # noqa: E402
                                  decoded_to_states)

MAX_STEP_UNITS = 28.0      # Link's top speed per 10 Hz env step; larger means the savestate had not settled

# Column of the decoder's `link_bg` head for each bgCheckFlags bit we read (the head only carries a subset).
BG_I_GROUND, BG_I_WALL = BG_DECODE_BITS.index(BG_GROUND), BG_DECODE_BITS.index(BG_WALL)
ROLL_IDX = bc.LINK_ACTION_NAMES.index("roll")     # verified by driving the controller; see bc.LINK_ACTION_VALUES
NO_LOCK = bc.N_ROWS                               # decode_targets/`targeted` use N_ROWS for "no lock-on"

# action_fn_rel -> class index, with the same merges the decoder targets use
_ACT_IDX = {v: i for i, v in enumerate(bc.LINK_ACTION_VALUES)}
_ACT_IDX.update({s: _ACT_IDX[d] for s, d in bc.LINK_ACTION_MERGE.items() if d in _ACT_IDX})

BOOL_KINDS = ("lockon", "jump", "roll", "collision")
BOOL_FLAG = {"lockon": "lock", "jump": "airborne", "roll": "roll", "collision": "wall"}
KINDS = ("hit", "link_hit", "walk", "kill", "block") + BOOL_KINDS


# ---------------------------------------------------------------- the boolean facts, read two ways
# These two functions must stay mirror images: the dream's flags come from the decoder HEADS (thresholded the
# way scripts/train_joint.py scores them) and the real flags from the decoder TARGETS for the same fields, so
# "did the dream reproduce it" compares a prediction with its own training target and nothing else.
def flags_from_decode(p) -> dict:
    """Decoder output (batched) -> per-step boolean facts."""
    bg = p["link_bg"].numpy() > 0
    return {"airborne": ~bg[:, BG_I_GROUND], "wall": bg[:, BG_I_WALL],
            "roll": p["link_action"].argmax(-1).numpy() == ROLL_IDX,
            "lock": p["targeted"].argmax(-1).numpy() != NO_LOCK}


def flags_from_raw(S) -> dict:
    """Raw state rows -> the same booleans, via the decoder's own targets."""
    Y = decode_targets(np.asarray(S, np.float32))
    bg = Y["link_bg"] > 0.5
    return {"airborne": ~bg[:, BG_I_GROUND], "wall": bg[:, BG_I_WALL],
            "roll": Y["link_action"] == ROLL_IDX, "lock": Y["targeted"] != NO_LOCK}


def _tracker_events(S):
    """Replay the REAL CombatTracker over a whole episode; return when it books kills and blocks/dodges.

    Window selection has to use the tracker itself, because a block is not a field in the state: it is a
    lunge->pull_back cycle that ended with the shield up, in range, without damage -- bookkeeping that only
    `CombatTracker.step` does. Returned with each block is the step its lunge cycle OPENED, so a window can be
    required to contain the whole cycle; a tracker reset in the middle of a cycle never opens it and would
    never book the block, which would put the real detection column below 1.00.
    """
    states = decoded_states_from_raw(S)
    trk = CombatTracker(CombatConfig())
    trk.reset(states[0], len(states))
    kills, cycles = [], []              # kills: (step, row); cycles: (lunge_open_step, block_step)
    open_at = {}
    for t in range(1, len(states)):
        before = set(trk.cycles)
        _, term, ev = trk.step(states[t - 1], states[t])
        for p in set(trk.cycles) - before:
            open_at[p] = t
        closed = before - set(trk.cycles)
        if ev.get("defeat"):
            for p, e in states[t].enemies.items():      # the row that just reached 0 HP
                if e.hp == 0 and states[t - 1].enemies.get(p) is not None and states[t - 1].enemies[p].hp > 0:
                    kills.append((t, p))
        if ev.get("block") or ev.get("dodge"):
            cycles.append((min((open_at.get(p, t) for p in closed), default=t), t))
        for p in closed:
            open_at.pop(p, None)
        if term:
            break
    return kills, cycles


def valid_starts(S, H):
    """Every window start that passes the settling filter, event or not -- the pool the control samples from."""
    px, pz = S[:, bc.LI["pos_x"]], S[:, bc.LI["pos_z"]]
    step = np.r_[0.0, np.hypot(np.diff(px), np.diff(pz))]
    return [t for t in range(len(S) - H - 1) if step[t + 1] <= MAX_STEP_UNITS]


def find_windows(S, H, walk_units, events=None):
    """Per-step real event flags, and the window starts that contain each kind within H steps.

    `events`, if given, is filled with the number of DISTINCT real occurrences per kind. It is not the same
    number as the window count and the difference is large: one block generates up to H qualifying starts, so
    58 real blocks in the held-out set become ~1000 "windows". The window count says how many rollouts were
    scored; only the event count says how much independent evidence exists behind them.
    """
    L = bc.LI
    hp = S[:, L["hp"]]
    rows = S[:, len(bc.LINK_FIELDS_V3):].reshape(len(S), bc.N_ROWS, bc.ROW_DIM)
    rhp = rows[:, :, bc.RI["hp"]]
    pres = rows[:, :, bc.RI["present"]] > 0.5
    px, pz = S[:, L["pos_x"]], S[:, L["pos_z"]]
    step = np.r_[0.0, np.hypot(np.diff(px), np.diff(pz))]
    hit = np.r_[False, ((rhp[1:] < rhp[:-1]) & pres[1:] & pres[:-1]).any(1)]
    lhit = np.r_[False, hp[1:] < hp[:-1]]
    out = {}
    ev = {} if events is None else events
    n = len(S) - H - 1
    if n <= 0:
        ev.update({k: 0 for k in KINDS})
        return {k: [] for k in KINDS}
    ok = lambda t: 0 <= t < n and step[t + 1] <= MAX_STEP_UNITS                     # noqa: E731
    cum = np.cumsum(step)
    for name, flag in (("hit", hit), ("link_hit", lhit)):
        c = np.cumsum(flag.astype(int))
        out[name] = [t for t in range(n) if c[t + H] - c[t] >= 1 and ok(t)]
        ev[name] = int(flag[1:n + H + 1].sum())
    out["walk"] = [t for t in range(n) if cum[t + H] - cum[t] >= walk_units and ok(t)]
    ev["walk"] = _disjoint(out["walk"], H)      # walking is not a discrete event: count separate bouts

    # --- boolean ONSETS: flag off at the window start, on somewhere inside it. Selecting on the onset (not
    # merely "the flag is set during the window") is what makes identity score 0: a window that starts with
    # Link already airborne is "detected" by a frozen world for free.
    f = flags_from_raw(S)
    for kind in BOOL_KINDS:
        v = f[BOOL_FLAG[kind]]
        c = np.cumsum(v.astype(int))
        out[kind] = [t for t in range(n) if not v[t] and c[t + H] - c[t] >= 1 and ok(t)]
        ev[kind] = int((v[1:n + H + 1] & ~v[:n + H]).sum())          # onsets: flag off -> on

    kills, cycles = _tracker_events(S)
    # kill: the defeat step inside the window, AND the row still alive and present at the window start --
    # otherwise a tracker reset at t marks it not-a-target (reset needs hp > 0) and can never book the kill.
    ks = set()
    for t_k, ptr in kills:
        row = (ptr - 0x80300000) // 0x100
        for t in range(max(0, t_k - H), t_k):
            if ok(t) and 0 <= row < bc.N_ROWS and pres[t, row] and rhp[t, row] > 0:
                ks.add(t)
    out["kill"], ev["kill"] = sorted(ks), len(kills)
    bs = set()
    for t_open, t_blk in cycles:
        for t in range(max(0, t_blk - H), t_open):      # t < t_open: the whole lunge cycle is inside
            if ok(t):
                bs.add(t)
    out["block"], ev["block"] = sorted(bs), len(cycles)
    return out


def _disjoint(starts, H):
    """How many non-overlapping windows the list covers -- the count of separate occurrences."""
    n, last = 0, -10 ** 9
    for t in starts:
        if t - last >= H:
            n, last = n + 1, t
    return n


# ---------------------------------------------------------------- one event definition, used three times
def count_events(states, H, dream=False):
    """Run the reward function's own tracker over a state sequence and count what it books.

    Identical code for the dream and for the real window, which is the only way the "did the dream reproduce
    it" number and the "is this window really an event" number can be compared at all.
    """
    trk = (DreamTracker if dream else CombatTracker)(CombatConfig())
    trk.reset(states[0], H)
    kills = despawn = blocks = lockons = 0
    at_zero = set()
    for t in range(1, len(states)):
        was = set(states[t - 1].enemies)
        for p, e in states[t - 1].enemies.items():
            if e.hp <= 0:
                at_zero.add(p)
        _, term, ev = trk.step(states[t - 1], states[t])
        kills += int(ev.get("defeat", 0))
        blocks += int(bool(ev.get("block"))) + int(bool(ev.get("dodge")))
        lockons += int(bool(ev.get("lockon")))
        despawn += len(at_zero & (was - set(states[t].enemies)))   # HP 0 while present, then removed
        if term:
            break
    return dict(kills=kills, kill_despawn=despawn, blocks=blocks, lockons=lockons)


def event(rec, kind, walk_units):
    """(did this rollout contain the event, its magnitude) -- the ONE definition, for dream / real / identity."""
    if kind == "hit":
        return rec["baba_hp_lost"] > 0, rec["baba_hp_lost"]
    if kind == "link_hit":
        return rec["link_hp_lost"] > 0, rec["link_hp_lost"] / 16.0          # hearts
    if kind == "walk":
        return rec["travel"] >= walk_units, rec["travel"]
    if kind == "kill":
        return rec["kills"] > 0, float(rec["kills"])
    if kind == "block":
        return rec["blocks"] > 0, float(rec["blocks"])
    v, s = rec["flags"][BOOL_FLAG[kind]], rec["start"][BOOL_FLAG[kind]]
    return (not s) and bool(v.any()), float(v.sum())                         # steps with the flag on


def first_step(rec, kind):
    """Step index (1-based) of the first flagged step, or None -- only defined for the boolean kinds."""
    if kind not in BOOL_KINDS:
        return None
    v = rec["flags"][BOOL_FLAG[kind]]
    return int(v.argmax()) + 1 if v.any() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--walk-units", type=float, default=120.0)
    ap.add_argument("--per-kind", type=int, default=40, help="windows sampled per event kind")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kinds", nargs="*", default=None, help="subset of %s" % (KINDS,))
    ap.add_argument("--control", type=int, default=40,
                    help="random windows (event or not) used as a FALSE-POSITIVE control: a detection rate "
                         "means nothing if the dream fires the same flag on every window it sees")
    ap.add_argument("--held-out", action="store_true", default=True)
    ap.add_argument("--out", default="results/event_windows.json")
    a = ap.parse_args()
    torch.set_num_threads(6)
    kinds = tuple(a.kinds) if a.kinds else KINDS

    ck = torch.load(a.model, weights_only=False, map_location="cpu")
    LAT, NA = ck["latent"], ck["n_actions"]
    enc = BCNet(link_dim=ck["link_dim"], row_dim=ck["row_dim"], n_actions=NA, latent=LAT, gru=0, **enc_arch(ck))
    enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = (UnifiedDecoder(latent=LAT) if ck.get("unified_decoder") else StateDecoder(latent=LAT))
    dec.load_state_dict(ck["decoder"]); dec.eval()
    pred = load_predictor(ck, LAT, NA)
    thead = None
    if ck.get("thead") is not None:
        thead = TransitionHead(LAT, NA); thead.load_state_dict(ck["thead"]); thead.eval()
    norm = Norm.load(ck["norm"]); scale = ck["decode_scale"]
    EC, ET = ck.get("event_calib") or {}, ck.get("exit_thresh", 6.38)
    held = set(ck["split"]["val_slots"])

    dirs = a.data or sorted(glob.glob("datasets/transitions/v3-*/*/"))
    pool, found, nev, ctrl = {k: [] for k in kinds}, {k: 0 for k in kinds}, {k: 0 for k in kinds}, []
    for d in dirs:
        mf = Path(d) / "manifest.jsonl"
        if not mf.exists():
            continue
        for m in map(json.loads, open(mf)):
            if a.held_out and m["slot"] not in held:
                continue
            z = np.load(Path(d) / m["file"])
            S, A = z["state"], z["action"]
            if len(A) < a.horizon + 2 or S.shape[1] != len(bc.LINK_FIELDS_V3) + bc.N_ROWS * bc.ROW_DIM:
                continue
            evc = {}
            for k, ts in find_windows(S[:len(A) + 1], a.horizon, a.walk_units, events=evc).items():
                if k not in pool:
                    continue
                found[k] += len(ts)
                nev[k] += evc.get(k, 0)
                for t in ts:
                    pool[k].append((S, A, t))
            ctrl += [(S, A, t) for t in valid_starts(S[:len(A) + 1], a.horizon)]
    rng0 = np.random.default_rng(0)

    def sample(ws, n):
        """Sample windows that do not overlap, so two rollouts cannot be scored on the SAME real event."""
        out, seen = [], {}
        for i in rng0.permutation(len(ws)):
            S, A, t = ws[i]
            got = seen.setdefault(id(S), [])
            if any(abs(t - u) < a.horizon for u in got):
                continue
            got.append(t)
            out.append(ws[i])
            if len(out) == n:
                break
        return out

    for k in pool:
        pool[k] = sample(pool[k], a.per_kind)
    ctrl = sample(ctrl, a.control)
    print(f"held-out slots {sorted(held)}, horizon {a.horizon}")
    print(f"distinct real events:            { {k: nev[k] for k in kinds} }")
    print(f"qualifying window starts:        { {k: found[k] for k in kinds} }")
    print(f"non-overlapping windows scored:  { {k: len(pool[k]) for k in kinds} }\n")

    def roll(S, A, t0):
        """Encode the real state at t0, roll H steps on the real actions, return one record per seed."""
        f = norm.apply(featurise(S[t0:t0 + 1], link_v2=wants_v2(ck["link_dim"])))
        outs = []
        for seed in range(a.seeds):
            timers, rng = {}, np.random.default_rng(seed)
            with torch.no_grad():
                z = enc.encode(*(torch.from_numpy(f[k]) for k in ("link", "rows", "ids", "mask")))
                p = dec(z)
                cur = decoded_to_states(p, scale)
                start = cur[0]                       # the dream's OWN start, not the truth's
                s_hp0 = start.hp
                s_bhp0 = sum(e.hp for e in start.enemies.values())
                start_flags = {k: bool(v[0]) for k, v in flags_from_decode(p).items()}
                trav = 0.0
                path, flags, seq = [], [], [start]
                for j in range(a.horizon):
                    oh = torch.nn.functional.one_hot(torch.tensor([int(A[t0 + j])]), NA).float()
                    q = thead(z, oh) if thead is not None else None
                    z = pred(torch.cat([z, oh], -1))
                    p = dec(z)
                    nxt = decoded_to_states(p, scale)
                    if q is not None:
                        nxt = apply_transition(cur, nxt, q, calib=EC, rng=rng, exit_thresh=ET,
                                               _timers=timers)
                    trav += math.hypot(nxt[0].pos[0] - cur[0].pos[0], nxt[0].pos[2] - cur[0].pos[2])
                    cur = nxt
                    path.append((cur[0].pos[0], cur[0].pos[2]))
                    flags.append({k: bool(v[0]) for k, v in flags_from_decode(p).items()})
                    seq.append(cur[0])
            rec = dict(start=start_flags,
                       flags={k: np.array([fl[k] for fl in flags]) for k in start_flags},
                       link_hp_lost=float(s_hp0 - seq[-1].hp),
                       baba_hp_lost=float(s_bhp0 - sum(e.hp for e in seq[-1].enemies.values())),
                       travel=trav, path=np.asarray(path, np.float32))
            rec.update(count_events(seq, a.horizon, dream=True))
            outs.append(rec)
        return outs

    def real_record(S, t0):
        """The same record, built from the real states of the same window (and from a frozen one)."""
        H = a.horizon
        states = decoded_states_from_raw(S[t0:t0 + H + 1])
        fr = flags_from_raw(S[t0:t0 + H + 1])
        xz = S[t0:t0 + H + 1][:, [bc.LI["pos_x"], bc.LI["pos_z"]]]
        rec = dict(start={k: bool(v[0]) for k, v in fr.items()},
                   flags={k: v[1:] for k, v in fr.items()},
                   link_hp_lost=float(states[0].hp - states[-1].hp),
                   baba_hp_lost=float(sum(e.hp for e in states[0].enemies.values())
                                      - sum(e.hp for e in states[-1].enemies.values())),
                   travel=float(np.sum(np.hypot(*np.diff(xz, axis=0).T))), path=xz[1:])
        rec.update(count_events(states, H, dream=False))
        idt = dict(start=dict(rec["start"]),
                   flags={k: np.repeat(v[0], H) for k, v in fr.items()},
                   link_hp_lost=0.0, baba_hp_lost=0.0, travel=0.0,
                   path=np.repeat(xz[:1], H, axis=0))
        idt.update(count_events([states[0]] * (H + 1), H, dream=False))
        return rec, idt

    # --- FALSE-POSITIVE CONTROL. A detection rate alone is worthless: a head that says "airborne" on every
    # frame scores 100% on jump windows. Roll RANDOM windows once and ask each kind's own event() how often
    # the dream fired where the real trajectory did not. High detect + high control = the flag is noise.
    fp = {k: [] for k in kinds}
    for i, (S, A, t0) in enumerate(ctrl):
        rr, _ = real_record(S, t0)
        rds = roll(S, A, t0)
        for k in kinds:
            if event(rr, k, a.walk_units)[0]:
                continue                                   # the event really happened: not a control case
            fp[k] += [event(rd, k, a.walk_units)[0] for rd in rds]
    print(f"control windows rolled: {len(ctrl)} "
          f"(negative for each kind: { {k: len(v) // max(a.seeds, 1) for k, v in fp.items()} })\n")

    res = {}
    for kind in kinds:
        wins = pool[kind]
        if not wins:
            continue
        acc = {w: [] for w in ("det_d", "det_r", "det_i", "mag_d", "mag_r", "mag_i", "any_d", "any_r",
                               "seed_mismatch", "step_d", "step_r", "traj", "traj_id", "end", "end_id",
                               "d_hp", "r_hp", "d_bhp", "r_bhp", "d_tr", "r_tr",
                               "kills_d", "kills_r", "desp_d", "desp_r", "blocks_d", "blocks_r",
                               "lock_d", "lock_r")}
        for S, A, t0 in wins:
            rr, ri = real_record(S, t0)
            dr_r, mg_r = event(rr, kind, a.walk_units)
            dr_i, mg_i = event(ri, kind, a.walk_units)
            real_xz = rr["path"]
            for rd in roll(S, A, t0):
                dr_d, mg_d = event(rd, kind, a.walk_units)
                acc["det_d"].append(dr_d);  acc["det_r"].append(dr_r);  acc["det_i"].append(dr_i)
                acc["mag_d"].append(mg_d);  acc["mag_r"].append(mg_r);  acc["mag_i"].append(mg_i)
                if kind in BOOL_KINDS:
                    fl = BOOL_FLAG[kind]
                    acc["any_d"].append(bool(rd["flags"][fl].any()))
                    acc["any_r"].append(bool(rr["flags"][fl].any()))
                    acc["seed_mismatch"].append(rd["start"][fl] != rr["start"][fl])
                    sd, sr = first_step(rd, kind), first_step(rr, kind)
                    if sd is not None and sr is not None:
                        acc["step_d"].append(sd); acc["step_r"].append(sr)
                n = min(len(rd["path"]), len(real_xz))
                err = np.linalg.norm(rd["path"][:n] - real_xz[:n], axis=1)
                err_id = np.linalg.norm(ri["path"][:n] - real_xz[:n], axis=1)
                acc["traj"].append(float(err.mean()));    acc["traj_id"].append(float(err_id.mean()))
                acc["end"].append(float(err[-1]));        acc["end_id"].append(float(err_id[-1]))
                # measure each trajectory against ITS OWN start: the decoded seed does not always match
                # truth (a Baba at 1 HP can decode as 2), and mixing the two produced fake "healing"
                acc["d_hp"].append(rd["link_hp_lost"]);   acc["r_hp"].append(rr["link_hp_lost"])
                acc["d_bhp"].append(rd["baba_hp_lost"]);  acc["r_bhp"].append(rr["baba_hp_lost"])
                acc["d_tr"].append(rd["travel"]);         acc["r_tr"].append(rr["travel"])
                for nm, key in (("kills", "kills"), ("desp", "kill_despawn"),
                                ("blocks", "blocks"), ("lock", "lockons")):
                    acc[nm + "_d"].append(rd[key]); acc[nm + "_r"].append(rr[key])
        mean = lambda k: float(np.mean(acc[k])) if acc[k] else float("nan")             # noqa: E731
        res[kind] = dict(
            n=len(wins), n_found=found[kind], n_events=nev[kind],
            detect_dream=mean("det_d"), detect_real=mean("det_r"), detect_identity=mean("det_i"),
            control_dream=float(np.mean(fp[kind])) if fp[kind] else float("nan"),
            control_n=len(fp[kind]) // max(a.seeds, 1),
            mag_dream=mean("mag_d"), mag_real=mean("mag_r"), mag_identity=mean("mag_i"),
            any_dream=mean("any_d"), any_real=mean("any_r"), seed_mismatch=mean("seed_mismatch"),
            step_dream=mean("step_d"), step_real=mean("step_r"),
            baba_hp_lost_dream=mean("d_bhp"), baba_hp_lost_real=mean("r_bhp"),
            link_hp_lost_dream=mean("d_hp") / 16, link_hp_lost_real=mean("r_hp") / 16,
            travel_dream=mean("d_tr"), travel_real=mean("r_tr"),
            kills_dream=mean("kills_d"), kills_real=mean("kills_r"),
            kill_despawn_dream=mean("desp_d"), kill_despawn_real=mean("desp_r"),
            blocks_dream=mean("blocks_d"), blocks_real=mean("blocks_r"),
            lockons_dream=mean("lock_d"), lockons_real=mean("lock_r"),
            detect_baba=float(np.mean(np.array(acc["d_bhp"]) > 0)),
            detect_baba_real=float(np.mean(np.array(acc["r_bhp"]) > 0)),
            detect_link=float(np.mean(np.array(acc["d_hp"]) > 0)),
            detect_link_real=float(np.mean(np.array(acc["r_hp"]) > 0)),
            traj_err=mean("traj"), traj_err_identity=mean("traj_id"),
            traj_end_err=mean("end"), traj_end_err_identity=mean("end_id"))

    print(f"EVENT-RICH WINDOWS, horizon {a.horizon}, {a.seeds} seeds. Each row's detect/magnitude is THAT "
          f"kind's own event,\ncomputed by one function for dream, real and identity. real must be ~100% or "
          f"the window logic is wrong.\n")
    print("'control' = the dream's rate on RANDOM windows where the event did NOT happen (false positives).\n")
    print(f"{'window kind':<10} {'n':>4} {'events':>6} | {'detect: dream':>13} {'real':>6} {'ident':>6} "
          f"{'control':>8} | {'magnitude: dream':>17} {'real':>8} {'ident':>7}")
    for k, v in res.items():
        c = "     -" if v["control_dream"] != v["control_dream"] else f"{v['control_dream']:>7.0%}"
        print(f"{k:<10} {v['n']:>4} {v['n_events']:>6} | {v['detect_dream']:>13.0%} {v['detect_real']:>6.0%} "
              f"{v['detect_identity']:>6.0%} {c:>8} | {v['mag_dream']:>17.2f} {v['mag_real']:>8.2f} "
              f"{v['mag_identity']:>7.2f}")
    print("\nSECONDARY -- the shared quantities, every row measured against its own start state.")
    print(f"{'window kind':<10} | {'baba HP d/r':>16} | {'hearts d/r':>13} | {'travel d/r':>14} | "
          f"{'kills d/r':>11} {'blocks d/r':>11} {'lockons d/r':>11}")
    for k, v in res.items():
        print(f"{k:<10} | {v['baba_hp_lost_dream']:>6.2f}/{v['baba_hp_lost_real']:<9.2f} | "
              f"{v['link_hp_lost_dream']:>5.2f}/{v['link_hp_lost_real']:<7.2f} | "
              f"{v['travel_dream']:>6.1f}/{v['travel_real']:<7.1f} | "
              f"{v['kills_dream']:>4.2f}/{v['kills_real']:<6.2f} "
              f"{v['blocks_dream']:>4.2f}/{v['blocks_real']:<6.2f} "
              f"{v['lockons_dream']:>4.2f}/{v['lockons_real']:<6.2f}")
    bools = [k for k in res if k in BOOL_KINDS]
    if bools:
        print("\nBOOLEAN KINDS -- 'any' ignores the seed frame (flag on at any step); 'seed wrong' is how often")
        print("  the dream's own start frame disagrees with the real one, which alone can kill an onset.")
        print(f"{'window kind':<10} {'any d/r':>13} {'seed wrong':>11} {'first step d/r':>16}")
        for k in bools:
            v = res[k]
            sd = "-" if v["step_dream"] != v["step_dream"] else f"{v['step_dream']:.1f}"
            sr = "-" if v["step_real"] != v["step_real"] else f"{v['step_real']:.1f}"
            print(f"{k:<10} {v['any_dream']:>6.0%}/{v['any_real']:<6.0%} {v['seed_mismatch']:>11.0%} "
                  f"{sd:>7}/{sr:<8}")
    print("\nTRAJECTORY -- position at each step vs the real position. Identity = Link frozen at the start.")
    print("  This is what 'walking' means; travel above is only path length and can match while being lost.")
    print(f"{'window kind':<10} {'mean err':>10} {'identity':>10} {'ratio':>7} | "
          f"{'end err':>9} {'identity':>10} {'ratio':>7}")
    for k, v in res.items():
        r1 = v["traj_err"] / max(v["traj_err_identity"], 1e-9)
        r2 = v["traj_end_err"] / max(v["traj_end_err_identity"], 1e-9)
        print(f"{k:<10} {v['traj_err']:>10.1f} {v['traj_err_identity']:>10.1f} {r1:>7.2f} | "
              f"{v['traj_end_err']:>9.1f} {v['traj_end_err_identity']:>10.1f} {r2:>7.2f}"
              f"{'   BETTER than identity' if r1 < 1 else ''}")
    for k, v in res.items():
        if v["detect_real"] < 0.95:
            print(f"WARNING {k}: real detection {v['detect_real']:.0%} < 95% -- the window definition and the "
                  f"endpoint test disagree, so this row is NOT evidence about the model.")
        if v["n"] < 10:
            print(f"WARNING {k}: only {v['n']} windows -- not evidence.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "horizon": a.horizon, "res": res}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
