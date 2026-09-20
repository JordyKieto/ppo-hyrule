"""Per-field do-nothing table: which fields of the OOMDP state are even non-trivial to predict?

    .venv/bin/python scripts/eval_field_identity.py
    .venv/bin/python scripts/eval_field_identity.py --data datasets/transitions/v3-main/20260919-084052 -n 120

Stage 0 of the object-oriented causal dynamics model. No training, no network, minutes to run.

AGENTS.md section 1 says a number is not a result until its controls are run, and the first control is
do-nothing. This script runs that control for EVERY field before anything is built on top of them, because
a world model that predicts 77 fields will otherwise report one aggregate number dominated by fields whose
correct answer is always "unchanged".

Four columns per field, and the point is that none of them means anything alone (AGENTS.md section 2):

  chg%      how often the field changes at all. Near zero => trivially predictable, and any aggregate that
            averages this field in is measuring the aggregate's composition, not a model.
  identity  the do-nothing error: predict v_{t+1} = v_t. This is the FLOOR a model must beat, and with a
            zero-init residual head an untrained model scores exactly this.
  marginal  the best constant predictor (median / mode / per-bit majority). If identity is not much better
            than marginal, the field carries little temporal structure to exploit.
  id|chg    the do-nothing error restricted to the steps where the field DID change. This is the number a
            model is actually being asked to produce: how big is the jump, given that something happened.
            A field can have a tiny `identity` and a huge `id|chg` -- Deku Baba position is exactly that --
            and reporting only the first is how an aggregate metric hides the real error (log section 15).
  id/pair   identity error divided by the error between two RANDOM samples of the field. This is AGENTS.md
            section 2's move/rand generalised to every value type: it says how far the field moves per step
            relative to its own spread. Small means static.

Errors are in each field's own units -- world units for positions, degrees for angles, error RATE for
enums and bitfields -- because the whole point of the object-oriented model is that fields are never mixed
into one scalar loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.oomdp import (Field, TURNS, assert_partition,  # noqa: E402
                            assert_roundtrip, decompose, fit_cardinalities)
from zelda_rl.state_layout import STATE_DIM_V3  # noqa: E402

DEFAULT_DATA = "datasets/transitions/v3-main/20260919-084052"
RESERVOIR = 40_000          # values kept per field for the marginal and random-pair controls


def _bits(v: np.ndarray, n: int) -> np.ndarray:
    """(..., 1) packed integers -> (..., n) float bits."""
    x = v.astype(np.int64).reshape(*v.shape[:-1], 1)
    return ((x >> np.arange(n)) & 1).astype(np.float32)


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def _prep(f: Field, v: np.ndarray) -> np.ndarray:
    """Raw column values -> the representation the error metric works on."""
    k = f.vtype.kind
    if k == "bits":
        return _bits(v, f.vtype.n)
    if k == "angle":
        return v * (2 * np.pi / TURNS) if f.vtype.unit == "binang" else v.copy()
    return v.astype(np.float32)


def _err(f: Field, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample error between two prepared values, in the field's own units. Shape (n,)."""
    k = f.vtype.kind
    if k == "angle":
        return np.abs(_wrap(a - b)).mean(-1) * 180.0 / np.pi          # degrees
    if k in ("real", "count"):
        return np.abs(a - b).mean(-1)                                  # world units / items
    return (a != b).astype(np.float32).mean(-1)                        # error rate (enum, boolean, bits)


def _unit(f: Field) -> str:
    return {"angle": "deg", "real": f.vtype.unit or "abs", "count": "abs"}.get(f.vtype.kind, "rate")


class Acc:
    """Exact sums for the identity control, plus a reservoir for the marginal and random-pair controls."""

    def __init__(self, f: Field):
        self.f, self.n, self.err, self.chg, self.err_chg = f, 0, 0.0, 0, 0.0
        self.pool: list[np.ndarray] = []
        self.pool_n = 0

    def add(self, cur: np.ndarray, nxt: np.ndarray) -> None:
        if cur.size == 0:
            return
        e = _err(self.f, nxt, cur)
        self.n += len(e)
        self.err += float(e.sum())
        moved = e > 0
        self.chg += int(moved.sum())
        self.err_chg += float(e[moved].sum())
        if self.pool_n < RESERVOIR:
            take = min(len(cur), RESERVOIR - self.pool_n)
            self.pool.append(cur[:take])
            self.pool_n += take

    def row(self) -> dict | None:
        if self.n == 0:
            return None
        f, pool = self.f, np.concatenate(self.pool) if self.pool else np.zeros((0, 1), np.float32)
        ident = self.err / self.n
        marg, pair = float("nan"), float("nan")
        if len(pool) > 1:
            rng = np.random.default_rng(0)
            i, j = rng.integers(0, len(pool), 20_000), rng.integers(0, len(pool), 20_000)
            pair = float(_err(f, pool[i], pool[j]).mean())
            marg = float(_err(f, pool, self._constant(pool)[None]).mean())
        return {"field": f"{f.name}", "unit": _unit(f), "n": self.n, "chg": self.chg / self.n,
                "identity": ident, "marginal": marg, "pair": pair,
                "id_given_chg": (self.err_chg / self.chg) if self.chg else 0.0, "mask": f.is_mask,
                "ratio": ident / pair if pair else float("nan")}

    def _constant(self, pool: np.ndarray) -> np.ndarray:
        """The best constant predictor under this field's own error metric."""
        k = self.f.vtype.kind
        if k == "angle":
            return np.angle(np.exp(1j * pool).mean(0))                 # circular mean
        if k in ("real", "count"):
            return np.median(pool, 0)                                  # minimises MAE
        vals, counts = np.unique(pool, axis=0, return_counts=True)     # mode (per-bit majority for bits)
        return vals[counts.argmax()] if k != "bits" else (pool.mean(0) > 0.5).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[DEFAULT_DATA],
                    help="one or more collection dirs, pooled. Pool everything: a field that is\n"
                         "constant in one scenario set may be the only interesting one in another.")
    ap.add_argument("-n", "--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--json", default="", help="also write the table here")
    args = ap.parse_args()

    print(f"schema partition: {assert_partition()}")
    rows = []
    for dd in args.data:
        dd = Path(dd)
        rs = [json.loads(l) for l in (dd / "manifest.jsonl").read_text().splitlines() if l.strip()]
        rows += [(dd, r) for r in (rs[: args.episodes] if args.episodes else rs)]

    accs: dict[str, dict[str, Acc]] = {}
    checked = False
    groups_seen, counts, steps = [], {}, 0
    for d, r in rows:
        z = np.load(d / r["file"])
        st, act = z["state"], z["action"]
        assert st.shape[-1] == STATE_DIM_V3, f"{r['file']} is {st.shape[-1]}-d, not v3"
        a = np.r_[act, act[-1]][: len(st)]
        if not checked:                       # a partition proves the bookkeeping; this proves the decomposition
            assert_roundtrip(st, a)
            checked = True
        g = decompose(st, a)
        groups_seen.append(g)
        steps += len(st) - 1
        for cname, grp in g.items():
            counts[cname] = max(counts.get(cname, 0), grp.n_obj)
            ca = accs.setdefault(cname, {})
            both = grp.mask[:-1] & grp.mask[1:]                        # object exists at t AND t+1
            for f in grp.cls.fields:
                if f.role == "action" or f.name not in grp.attrs:
                    continue
                v = _prep(f, grp.attrs[f.name])
                # `present` is scored wherever the object exists at t -- its despawn (1 -> 0) IS the target.
                # Every other field has an undefined target once the object is gone, so it uses `both`.
                sel = grp.mask[:-1] if f.is_mask else both
                ca.setdefault(f.name, Acc(f)).add(v[:-1][sel], v[1:][sel])

    schema, observed = fit_cardinalities(groups_seen)
    print(f"\n{len(rows)} episodes, {steps} transitions, pooled from {len(args.data)} "
          f"collection(s): {', '.join(Path(x).parent.name for x in args.data)}")
    b = schema.edge_budget()
    flat = STATE_DIM_V3 * (STATE_DIM_V3 + 1)
    print(f"candidate causal edges: OOCG {b['total']} class-level ({b['local']} local + {b['global']} global, "
          f"fixed) vs flat CDL {flat} over the 363-d state -- {flat / b['total']:.0f}x fewer, and the OOCG "
          f"count does not change when an actor spawns or dies")

    out = {}
    for cname in ("Link", "Camera", "Room", "En_Dekubaba", "En_St", "En_Sw", "Actor"):
        if cname not in accs:
            continue
        table = [r for r in (a.row() for a in accs[cname].values()) if r]
        for r in table:
            vals = observed.get(f"{cname}.{r['field']}")
            r["values"] = len(vals) if vals else None
        table.sort(key=lambda r: -r["chg"])
        out[cname] = table
        print(f"\n=== {cname}  ({counts[cname]} instance{'s' if counts[cname] != 1 else ''}) "
              + "=" * max(0, 58 - len(cname)))
        print(f"  {'field':<20} {'unit':>6} {'chg%':>7} {'identity':>10} {'id|chg':>10} "
              f"{'marginal':>10} {'id/pair':>8}  vals")
        for r in table:
            # chg 0 with marginal 0 is a global constant; chg 0 with marginal > 0 varies BETWEEN episodes
            # but never within one -- an episode constant, which is the free-signal class AGENTS.md section 1
            # names (`time_left`, the `goal_*` block). Those are the dangerous ones: they look informative.
            trivial = ""
            if r["chg"] == 0:
                trivial = "  <- CONSTANT (global)" if r["marginal"] == 0 else "  <- CONSTANT (per-episode)"
            v = "" if r["values"] is None else str(r["values"])
            if r["mask"]:                                    # `present` is only ever sampled where it is 1
                r = {**r, "marginal": float("nan"), "ratio": float("nan")}
            print(f"  {r['field']:<20} {r['unit']:>6} {100 * r['chg']:7.3f} {r['identity']:10.4f} "
                  f"{r['id_given_chg']:10.4f} {r['marginal']:10.4f} {r['ratio']:8.4f}  {v:>4}{trivial}")

    n_const = sum(1 for t in out.values() for r in t if r["chg"] == 0)
    n_epis = sum(1 for t in out.values() for r in t if r["chg"] == 0 and r["marginal"] > 0)
    n_all = sum(len(t) for t in out.values())
    print(f"\n{n_const} of {n_all} scored fields never change in this data ({n_epis} of them vary between "
          f"episodes but never within one). Any metric that averages over fields, or over actor rows, is "
          f"{100 * n_const / n_all:.0f}% composition.")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
