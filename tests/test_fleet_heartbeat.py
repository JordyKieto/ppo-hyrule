"""Liveness must not depend on the trainer's phase.

Fails against the version that heartbeat once per epoch: on 2026-10-30 an RvS epoch contained a ~24-episode
online eval at a 300 s cap, every rank went stale inside it, and one sync round averaged 2 of 18 ranks.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zelda_rl.fleet_link import FleetLink  # noqa: E402


class FakeLink(FleetLink):
    """A FleetLink with no Upstash: only the heartbeat machinery is under test."""

    def __init__(self):
        self.rank, self.world, self.exp, self.launch = 3, 18, "test", "0"
        self.sent = []

    def heartbeat(self, metrics: dict) -> None:
        self.sent.append(dict(metrics))


def test_heartbeat_continues_while_the_main_thread_is_busy():
    link = FakeLink()
    h = link.start_heartbeat(interval=1.0, metrics={"epoch": 0})
    time.sleep(0.2)
    n0 = len(link.sent)
    time.sleep(2.5)                      # the main thread is "inside a long eval"
    n1 = len(link.sent)
    h.stop()
    assert n0 >= 1, "no heartbeat was sent at startup"
    assert n1 - n0 >= 2, f"heartbeat stalled while the main thread blocked: {n0} -> {n1}"


def test_update_changes_what_is_reported():
    link = FakeLink()
    h = link.start_heartbeat(interval=1.0, metrics={"epoch": 0})
    time.sleep(0.2)
    h.update(epoch=7, dmg=1.5)
    time.sleep(1.4)
    h.stop()
    last = link.sent[-1]
    assert last.get("epoch") == 7 and last.get("dmg") == 1.5, last


def test_stop_ends_the_thread():
    link = FakeLink()
    h = link.start_heartbeat(interval=1.0)
    time.sleep(0.2)
    h.stop()
    n = len(link.sent)
    time.sleep(1.6)
    assert len(link.sent) == n, "heartbeat kept going after stop()"


if __name__ == "__main__":
    fails = 0
    for fn in (test_heartbeat_continues_while_the_main_thread_is_busy,
               test_update_changes_what_is_reported,
               test_stop_ends_the_thread):
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"FAIL {fn.__name__}: {e}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
