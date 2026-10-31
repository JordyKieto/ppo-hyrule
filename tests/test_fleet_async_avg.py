"""The aggregator must average ranks that are on OTHER rounds.

Measured on 2026-10-30: 18 pods, every heartbeat fresh, and a sync round averaged 3 of 10 -- because the
round is `epoch // sync_every`, pods drift (104..120 epochs observed), and the aggregator only looked in its
own round's bucket. These tests fail against that version.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.fleet_link import FleetLink, _state_bytes, _cpu_state  # noqa: E402


class FakeU:
    """Just enough Upstash: a dict of blobs and string keys."""

    def __init__(self):
        self.blobs, self.kv = {}, {}

    def put_blob(self, key, data, ttl=None):
        self.blobs[key] = data
        return {"bytes": len(data)}

    def get_blob(self, key):
        return self.blobs.get(key)

    def cmd(self, *a):
        if a[0] == "SET":
            self.kv[a[1]] = a[2]; return "OK"
        if a[0] == "GET":
            return self.kv.get(a[1])
        return None

    def pipeline(self, cmds):
        return [self.cmd(*c) for c in cmds]


def _link(rank, world, live):
    lk = FleetLink.__new__(FleetLink)
    lk.exp, lk.rank, lk.world, lk.launch = "t", rank, world, "L"
    lk.u = FakeU()
    lk.live_peers = lambda max_age=150.0: set(live)
    lk._submit = lambda fn, *a, **k: None
    return lk


def _model(fill):
    m = nn.Linear(4, 2)
    with torch.no_grad():
        m.weight.fill_(fill); m.bias.fill_(fill)
    return m


def test_averages_peers_from_older_rounds():
    agg = _link(0, 3, live={1, 2})
    # peers 1 and 2 published at rounds 5 and 4; the aggregator is at round 6
    root = f"zrl:t:sync:L"
    for r, (rnd, fill) in {1: (5, 3.0), 2: (4, 5.0)}.items():
        agg.u.put_blob(f"{root}:newest:r{r}",
                       _state_bytes({"state": _cpu_state(_model(fill).state_dict()), "update": rnd}))
    m = _model(1.0)
    out = agg.average_weights(m, 6, aggregator=0, straggler_wait=0.1, timeout=0.1)
    assert out["sync_ranks"] == 3, f"averaged {out['sync_ranks']} of 3 (drifted peers were skipped)"
    assert out["rounds_mixed"] == "4..6", out["rounds_mixed"]
    # (1 + 3 + 5)/3 = 3
    assert abs(float(m.weight.detach().mean()) - 3.0) < 1e-6, float(m.weight.detach().mean())


def test_refuses_blobs_that_are_too_stale():
    agg = _link(0, 2, live={1})
    root = "zrl:t:sync:L"
    agg.u.put_blob(f"{root}:newest:r1",
                   _state_bytes({"state": _cpu_state(_model(9.0).state_dict()), "update": 0}))
    m = _model(1.0)
    out = agg.average_weights(m, 20, aggregator=0, straggler_wait=0.1, timeout=0.1,
                              max_stale_rounds=5)
    assert out["sync_ranks"] == 1, "a 20-round-old blob must not be averaged in"
    assert abs(float(m.weight.detach().mean()) - 1.0) < 1e-6


def test_current_round_blob_is_preferred():
    agg = _link(0, 2, live={1})
    root, base = "zrl:t:sync:L", "zrl:t:sync:L:u6"
    agg.u.put_blob(f"{base}:r1", _state_bytes(_cpu_state(_model(3.0).state_dict())))
    agg.u.put_blob(f"{root}:newest:r1",
                   _state_bytes({"state": _cpu_state(_model(99.0).state_dict()), "update": 2}))
    m = _model(1.0)
    out = agg.average_weights(m, 6, aggregator=0, straggler_wait=0.1, timeout=0.1)
    assert out["sync_ranks"] == 2
    assert abs(float(m.weight.detach().mean()) - 2.0) < 1e-6, "should use the round-6 blob (1+3)/2=2"


if __name__ == "__main__":
    fails = 0
    for fn in (test_averages_peers_from_older_rounds,
               test_refuses_blobs_that_are_too_stale,
               test_current_round_blob_is_preferred):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
