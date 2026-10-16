"""The GPU sync bugs that made a 4xH100 "distributed" WM run four independent quarter-data models.

Every assertion here FAILS against the code as it was: `weight_hash` called `.numpy()` on a device tensor,
the aggregator summed device tensors against CPU blobs with `.double()`, and a dead configured aggregator
left every follower waiting the full 600 s timeout per round. Run on mps when present (it raises the same
TypeError as cuda, with the same message), on cpu otherwise -- the point is that NO path may touch a device
tensor with numpy or mix devices in a reduction.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.fleet_link import _cpu_state, average_states, pick_aggregator, weight_hash  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def test_hash_is_device_independent():
    torch.manual_seed(0)
    m = nn.Linear(8, 4)
    h_cpu = weight_hash(m.state_dict())
    h_dev = weight_hash(m.to(DEV).state_dict())          # raised TypeError before the fix
    assert h_cpu == h_dev, f"hash depends on device: {h_cpu} vs {h_dev}"


def test_average_mixes_device_and_cpu_states():
    a = nn.Linear(4, 2); b = nn.Linear(4, 2)
    with torch.no_grad():
        a.weight.fill_(1.0); a.bias.fill_(0.0)
        b.weight.fill_(3.0); b.bias.fill_(2.0)
    avg = average_states([a.to(DEV).state_dict(), b.state_dict()])   # raised TypeError before the fix
    assert all(v.device.type == "cpu" for v in avg.values()), "average must come back on cpu"
    assert torch.allclose(avg["weight"], torch.full((2, 4), 2.0)), avg["weight"]
    assert torch.allclose(avg["bias"], torch.full((2,), 1.0)), avg["bias"]
    # and it must LOAD into a model that lives on the device
    tgt = nn.Linear(4, 2).to(DEV)
    tgt.load_state_dict(avg)
    assert abs(float(tgt.weight.detach().cpu().mean()) - 2.0) < 1e-6


def test_average_order_is_deterministic():
    xs = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        xs.append(nn.Linear(6, 3).state_dict())
    h1 = weight_hash(average_states(xs))
    h2 = weight_hash(average_states(list(xs)))
    assert h1 == h2


def test_dead_aggregator_is_replaced():
    # rank 0 dead, ranks 1-3 alive: the lowest live rank takes over instead of everyone waiting on rank 0
    assert pick_aggregator(0, {2, 3}, 1) == 1
    assert pick_aggregator(0, {1, 3}, 2) == 1
    assert pick_aggregator(0, {1, 2}, 3) == 1
    # configured aggregator alive: unchanged
    assert pick_aggregator(0, {1, 2, 3}, 0) == 0
    assert pick_aggregator(0, {0, 2, 3}, 1) == 0
    # alone: itself
    assert pick_aggregator(0, set(), 2) == 2


def test_cpu_state_does_not_alias_the_model():
    m = nn.Linear(3, 3).to(DEV)
    sd = _cpu_state(m.state_dict())
    with torch.no_grad():
        m.weight.fill_(9.0)
    assert float(sd["weight"].mean()) != 9.0, "uploaded state must be a snapshot, not a live view"


if __name__ == "__main__":
    fails = 0
    for fn in (test_hash_is_device_independent,
               test_average_mixes_device_and_cpu_states,
               test_average_order_is_deterministic,
               test_dead_aggregator_is_replaced,
               test_cpu_state_does_not_alias_the_model):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\ndevice={DEV}  {'ALL PASS' if not fails else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
