"""Emit one line per notable fleet event from Upstash heartbeats (for monitoring without SSH)."""
import sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.fleet_link import heartbeats
exp, world = sys.argv[1], int(sys.argv[2])
last_reported, seen_ranks, first = 0, set(), True
consensus, odd, lag_warned = set(), {}, {}
while True:
    try:
        hbs = heartbeats(exp)
    except Exception as e:  # noqa: BLE001
        print(f"upstash error: {e}", flush=True); time.sleep(60); continue
    now = time.time()
    ranks = {h["rank"]: h for h in hbs}
    for r in set(range(world)) - set(ranks):
        if r in seen_ranks:
            print(f"ALERT rank{r} heartbeat missing (crashed/hung?)", flush=True); seen_ranks.discard(r)
    for r, h in ranks.items():
        if r not in seen_ranks:
            seen_ranks.add(r)
            if not first: print(f"rank{r} alive: update {h.get("update")} run {h.get("run")}", flush=True)
        if now - h["t"] > 240:
            print(f"ALERT rank{r} heartbeat stale {now - h['t']:.0f}s", flush=True)
    # Straggler-tolerant sync: slow ranks adopt the newest average while their own update counter lags, so
    # compare each rank's hash against every consensus hash seen so far (majority hash per update).
    # A hash shared by >=2 ranks is an average (accumulated, since stragglers may hold an older/newer one alone).
    counts = {}
    for h in ranks.values():
        if h.get("weight_hash"):
            counts[h["weight_hash"]] = counts.get(h["weight_hash"], 0) + 1
    consensus.update(k for k, c in counts.items() if c >= 2)
    for r, h in ranks.items():
        wh, u = h.get("weight_hash"), h.get("update")
        if wh and wh not in consensus:
            seen_u = odd.setdefault(r, set()); seen_u.add(u)
            if len(seen_u) == 3:  # still unmatched after 3 of its own syncs: real divergence, not a lagging snapshot
                print(f"ALERT rank{r} weights {wh[:12]} match no fleet average for 3 syncs (update {u})", flush=True)
        else:
            odd.pop(r, None)
    if len(ranks) == world:
        upd = min(h.get("update", 0) for h in ranks.values())
        top_upd = max(h.get("update", 0) for h in ranks.values())
        for r, h in ranks.items():
            lag = top_upd - h.get("update", 0)
            if lag >= 15 and lag >= lag_warned.get(r, 0) + 10:
                lag_warned[r] = lag
                print(f"WARN rank{r} lags {lag} updates behind update {top_upd}", flush=True)
        hashes = {h.get("weight_hash", "")[:12] for h in ranks.values() if h.get("update") == top_upd}
        if first or top_upd >= last_reported + 10:
            first = False; last_reported = top_upd
            def f(k):
                vals = [h.get(k) for h in ranks.values() if isinstance(h.get(k), (int, float)) and math.isfinite(h.get(k))]
                return sum(vals) / len(vals) if vals else float("nan")
            sps = sum(h.get("env_steps_per_sec", 0) or 0 for h in ranks.values())
            if any("kill_frac" in h for h in ranks.values()):
                print(f"fleet update {top_upd} (slowest {upd}): succ {f('success_rate'):.0%} kills {f('kills'):.2f} kill% {f('kill_frac'):.0%} "
                      f"death {f('death_rate'):.0%} left {f('left_room_rate'):.0%} hpLost {f('hp_lost_hearts'):.2f} "
                      f"dmg {f('damage_dealt'):.2f} lockons {f('lockons'):.1f} return {f('return'):.2f} | "
                      f"fleet {sps:.0f} env steps/s | hash {hashes}", flush=True)
                continue
            print(f"fleet update {top_upd} (slowest {upd}): succ {f('success_rate'):.0%} final dist {f('final_pos_dist'):.0f} min dist "
                  f"{f('min_pos_dist'):.0f} rupees {f('final_rupees'):.2f}/6 return {f('return'):.2f} | fleet {sps:.0f} "
                  f"env steps/s | emu {f('emu_speed_x'):.1f}x | hash {hashes}", flush=True)
    time.sleep(45)
