"""Pod <-> Mac link over Upstash Redis REST: heartbeats, published policy checkpoints, cross-pod weight averaging.

Credentials: UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN from the environment (pods: sourced from .fleet_secrets),
else fleet.secrets on the Mac. Keys are namespaced `zrl:{experiment}:...`; values never logged.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time

import requests

CHUNK = 400_000          # base64 chars per value (Upstash request limit headroom)
HEARTBEAT_TTL = 300
SYNC_TTL = 600           # sync blobs are only needed for one update; keep the Upstash footprint small at 20 ranks
BLOB_TTL = 24 * 3600


class Upstash:
    def __init__(self):
        url, token = os.environ.get("UPSTASH_REDIS_REST_URL"), os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if not (url and token):
            from fleet.secrets import load_secrets
            s = load_secrets()
            url, token = s.get("UPSTASH_REDIS_REST_URL"), s.get("UPSTASH_REDIS_REST_TOKEN")
        if not (url and token):
            raise KeyError("UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN not configured")
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}"})

    TIMEOUT = (5, 20)        # connect, read. A pod that can't reach Upstash must fail fast, not hang minutes (seen live).

    def cmd(self, *args, attempts: int = 3):
        for attempt in range(attempts):
            try:
                r = self.s.post(self.url, json=[str(a) for a in args], timeout=self.TIMEOUT)
                r.raise_for_status()
                return r.json().get("result")
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
                time.sleep(1.0 * (attempt + 1))

    def pipeline(self, commands: list[list]):
        r = self.s.post(f"{self.url}/pipeline", json=[[str(a) for a in c] for c in commands], timeout=self.TIMEOUT)
        r.raise_for_status()
        return [x.get("result") for x in r.json()]

    # ---- blobs (chunked, base64) ----
    def put_blob(self, key: str, data: bytes, ttl: int = BLOB_TTL) -> dict:
        b64 = base64.b64encode(data).decode()
        chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
        for i, c in enumerate(chunks):
            self.cmd("SET", f"{key}:chunk{i}", c, "EX", ttl)
        meta = {"chunks": len(chunks), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.cmd("SET", f"{key}:meta", json.dumps(meta), "EX", ttl)
        return meta

    def get_blob(self, key: str) -> bytes | None:
        meta = self.cmd("GET", f"{key}:meta")
        if not meta:
            return None
        meta = json.loads(meta)
        parts = self.pipeline([["GET", f"{key}:chunk{i}"] for i in range(meta["chunks"])])
        if any(p is None for p in parts):
            return None
        data = base64.b64decode("".join(parts))
        return data if hashlib.sha256(data).hexdigest() == meta["sha256"] else None


def _state_bytes(obj) -> bytes:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _state_load(data: bytes):
    import torch
    return torch.load(io.BytesIO(data), weights_only=False, map_location="cpu")


def _cpu_state(sd: dict) -> dict:
    """A CPU copy of a state dict. Everything that leaves a rank or gets reduced is CPU-side.

    The GPU WM fleet averaged NOTHING for its whole run because this was missing in three places: the
    weight hash called `.numpy()` on cuda tensors, the aggregator summed cuda tensors against CPU blobs,
    and `.double()` is unsupported on mps. The trainer caught the exception and printed "fleet sync FAILED
    ... training alone" once per round, so four pods trained four independent quarter-data models while the
    launch log said "averaging every 2 epoch(s)".
    """
    return {k: (v.detach().cpu() if hasattr(v, "detach") else v) for k, v in sd.items()}


def average_states(states: list[dict]) -> dict:
    """Mean of state dicts, on CPU, in float64, in the caller's given order -> bit-identical on every rank."""
    import torch
    with torch.no_grad():
        ref = _cpu_state(states[0])
        cpu = [_cpu_state(st) for st in states]
        return {k: (sum(st[k].double() for st in cpu) / len(cpu)).to(ref[k].dtype) for k in ref}


def weight_hash(sd: dict) -> str:
    """Device-independent fingerprint: the same weights hash the same from cpu, cuda or mps."""
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes()
                                   for v in sd.values())).hexdigest()[:12]


def pick_aggregator(configured: int, live: set[int], self_rank: int) -> int:
    """The configured aggregator if it is alive, else the lowest live rank promotes itself.

    Rank 0 of the WM fleet died at launch and never wrote a log line. Every other rank then waited the full
    600 s follower timeout on every sync round -- 616 s epochs against 15 s normally -- for an average that
    could never be published. A dead aggregator must cost one round, not the run."""
    roster = set(live) | {self_rank}
    return configured if configured in roster else min(roster)


class FleetLink:
    """Used by train_ppo.py on each pod (rank) of an experiment."""

    def __init__(self, experiment: str, rank: int, world_size: int, launch_id: str = "0"):
        self.exp, self.rank, self.world = experiment, rank, world_size
        self.launch = launch_id          # sync keys are per launch: a relaunch must never average with stale blobs
        self.u = Upstash()
        self.u.cmd("SADD", "zrl:experiments", experiment)

        import queue
        import threading
        self._q: "queue.Queue" = queue.Queue(maxsize=8)
        threading.Thread(target=self._worker, daemon=True, name="fleet-link").start()

    def _worker(self):
        while True:
            fn, args = self._q.get()
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 -- telemetry must never take training down
                print(f"[fleet-link] {fn.__name__} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)

    def _submit(self, fn, *args):
        try:
            self._q.put_nowait((fn, args))
        except Exception:  # noqa: BLE001 -- queue full (Upstash unreachable): drop, don't block the trainer
            pass

    def heartbeat(self, metrics: dict) -> None:
        """Non-blocking."""
        self._submit(self._heartbeat, {"rank": self.rank, "t": time.time(), **metrics})

    def _heartbeat(self, payload: dict) -> None:
        self.u.cmd("SET", f"zrl:{self.exp}:hb:{self.rank}", json.dumps(payload, default=float), "EX", HEARTBEAT_TTL)

    def publish_policy(self, model, config: dict, update: int, global_step: int, metrics: dict) -> None:
        """Non-blocking: snapshots the weights now, uploads in the background."""
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._submit(self._publish_policy, state, config, update, global_step, metrics)

    def live_peers(self, max_age: float = 150.0) -> set[int]:
        keys = self.u.cmd("KEYS", f"zrl:{self.exp}:hb:*") or []
        vals = self.u.pipeline([["GET", k] for k in keys]) if keys else []
        now = time.time()
        return {h["rank"] for h in (json.loads(v) for v in vals if v) if now - h["t"] < max_age and h["rank"] != self.rank}

    def _publish_policy(self, state, config: dict, update: int, global_step: int, metrics: dict) -> None:
        """Latest policy (weights + config) for Mac-side play/eval. One pointer per experiment: newest wins."""
        payload = {"model": state, "config": config, "update": update, "global_step": global_step,
                   "rank": self.rank, "best": metrics, "optimizer": None, "episode_count": metrics.get("episode_count", 0)}
        key = f"zrl:{self.exp}:policy:r{self.rank}:u{update}"
        cur = self.u.cmd("GET", f"zrl:{self.exp}:policy:latest")
        if cur:
            c = json.loads(cur)
            # The guard is per LAUNCH. Within one run a slow rank must not replace a newer policy; ACROSS
            # runs the update counter restarts at 0 on a --init-from resume, so comparing raw update numbers
            # silently drops every publish from the new run until it passes the old run's last update.
            # Observed: v15 resumed from v14 update 33, and its publishes at 5/10/15/20 were all discarded
            # against v14's stale pointer at 40 -- the eval fleet re-scored the same dead v14 checkpoint for
            # 25 minutes while believing it was tracking the live run. A different launch always wins.
            if c.get("launch") == self.launch and c.get("update", -1) > update:
                return
        meta = self.u.put_blob(key, _state_bytes(payload))
        self.u.cmd("SET", f"zrl:{self.exp}:policy:latest",
                   json.dumps({"key": key, "rank": self.rank, "update": update, "launch": self.launch,
                               "t": time.time(), **meta, "metrics": metrics}, default=float), "EX", BLOB_TTL)

    def average_weights(self, model, update: int, timeout: float = 300.0, aggregator: int = 0,
                        straggler_wait: float = 30.0) -> dict:
        """Parameter averaging after PPO update `update`, aggregator pattern, straggler tolerant.

        Every rank uploads its weights. The aggregator waits at most `straggler_wait` for live ranks, averages what
        arrived (rank order, float64 -> bit-identical), publishes `avg` + a `latest` pointer. Followers load their
        update's average, or catch up to any NEWER average; a slow rank never holds the fleet back and never drifts
        (seen live: fast ranks timed out waiting on a slow aggregator and trained alone)."""
        import torch
        root = f"zrl:{self.exp}:sync:{self.launch}"
        base = f"{root}:u{update}"
        t0 = time.time()
        local = _cpu_state(model.state_dict())
        self.u.put_blob(f"{base}:r{self.rank}", _state_bytes(local), ttl=SYNC_TTL)
        live0 = self.live_peers()
        aggregator = pick_aggregator(aggregator, live0, self.rank)
        if self.rank == aggregator:
            live = live0
            expected = live if (live or update > 2) else {r for r in range(self.world) if r != self.rank}
            peers = {self.rank: local}
            wait = straggler_wait if update > 2 else max(straggler_wait, 120.0)
            while time.time() - t0 < wait and not expected <= set(peers):
                for r in expected - set(peers):
                    blob = self.u.get_blob(f"{base}:r{r}")
                    if blob is not None:
                        peers[r] = _state_load(blob)
                if not expected <= set(peers):
                    time.sleep(1.0)
            states = [peers[r] for r in sorted(peers)]
            avg = average_states(states)
            self.u.put_blob(f"{base}:avg", _state_bytes({"state": avg, "ranks": sorted(peers), "update": update}), ttl=SYNC_TTL)
            self.u.cmd("SET", f"{root}:latest", str(update), "EX", SYNC_TTL)
            model.load_state_dict(avg)
            n, got, src = len(states), sorted(peers), update
            old = update - 5
            if old >= 1:
                self._submit(self._cleanup, [f"{root}:u{old}:avg"] + [f"{root}:u{old}:r{r}" for r in range(self.world)])
        else:
            payload, src = None, None
            # a follower never waits the full timeout on an aggregator that is not alive (see pick_aggregator)
            deadline = timeout if aggregator in live0 or update <= 2 else min(timeout, straggler_wait + 30.0)
            while time.time() - t0 < deadline and payload is None:
                blob = self.u.get_blob(f"{base}:avg")
                if blob is not None:
                    payload, src = _state_load(blob), update
                    break
                latest = self.u.cmd("GET", f"{root}:latest")
                if latest is not None and int(latest) > update:           # aggregator is ahead: catch up now
                    blob = self.u.get_blob(f"{root}:u{int(latest)}:avg")
                    if blob is not None:
                        payload, src = _state_load(blob), int(latest)
                        break
                time.sleep(1.0)
            if payload is None:
                n, got = 1, [self.rank]
            else:
                model.load_state_dict(payload["state"])
                n, got = len(payload["ranks"]), payload["ranks"]
        h = weight_hash(model.state_dict())
        return {"sync_ranks": n, "sync_wait_s": time.time() - t0, "weight_hash": h,
                "aggregator": aggregator,
                "sync_included_self": self.rank in got, "sync_from_update": src if src is not None else -1}

    def _cleanup(self, keys: list[str]) -> None:
        dels = []
        for k in keys:
            dels.append(f"{k}:meta")
            dels += [f"{k}:chunk{i}" for i in range(8)]
        for i in range(0, len(dels), 200):
            self.u.cmd("DEL", *dels[i:i + 200])

def fetch_latest_policy(experiment: str, dest_dir) -> "tuple[str, dict] | None":
    """Mac side: download the newest published policy for an experiment into dest_dir. Returns (path, pointer)."""
    from pathlib import Path
    u = Upstash()
    ptr = u.cmd("GET", f"zrl:{experiment}:policy:latest")
    if not ptr:
        return None
    ptr = json.loads(ptr)
    dest = Path(dest_dir) / f"{experiment}_r{ptr['rank']}_u{ptr['update']:05d}.pt"
    if not dest.exists():
        blob = u.get_blob(ptr["key"])
        if blob is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    return str(dest), ptr


def clear_sync_keys(experiment: str) -> int:
    """Delete an experiment's sync blobs (call before relaunching a fleet)."""
    u = Upstash()
    keys = u.cmd("KEYS", f"zrl:{experiment}:sync:*") or []
    for i in range(0, len(keys), 100):
        u.cmd("DEL", *keys[i:i + 100])
    return len(keys)


def heartbeats(experiment: str) -> list[dict]:
    u = Upstash()
    keys = u.cmd("KEYS", f"zrl:{experiment}:hb:*") or []
    vals = u.pipeline([["GET", k] for k in keys]) if keys else []
    return sorted((json.loads(v) for v in vals if v), key=lambda h: h["rank"])
