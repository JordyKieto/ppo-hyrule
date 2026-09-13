"""Reproduce 'VI stuck' stalls: N envs, random actions + frequent resets; on stall, diagnose view vs fresh read vs process state.

    .venv/bin/python-dbg scripts/stress_stall.py <num_envs> <seconds>
"""
import os
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zelda_rl.env import N_ACTIONS, DolphinGoalEnv, EnvConfig, EnvError  # noqa: E402
from zelda_rl.machmem import MachTask, find_mem1_views  # noqa: E402
from zelda_rl.obs import VI_COUNTER, read_state_file  # noqa: E402

D = Path(os.path.expanduser("~/Library/Application Support/Dolphin/StateSaves"))
files = {k: D / f"NACE01.s{k:02d}" for k in (1, 2, 3, 5)}
states = {k: read_state_file(f) for k, f in files.items()}
N, DURATION = int(sys.argv[1]), float(sys.argv[2])
TORCH_LOAD = "--torch-load" in sys.argv            # mimic the trainer: per-step inference + periodic PPO-update bursts
if TORCH_LOAD:
    import torch
    from zelda_rl.ppo import ActorCritic
    net = ActorCritic(44, N_ACTIONS)
    print("torch threads:", torch.get_num_threads(), flush=True)
    last_burst = time.time()
cfg = EnvConfig(video_backend="Null", unlimited_speed=True, obs_version=1)
VI_OFF = 0xF64120 + (VI_COUNTER & 0x3FFFFF)


def make(i, tag):
    e = DolphinGoalEnv(i, Path(f"runs/instances/stress{i}{tag}"), files, states, states[5], cfg)
    t = MachTask(e.inst.proc.pid)
    return e, t, find_mem1_views(t)


envs, tasks, views = zip(*[make(i, "") for i in range(N)])
envs, tasks, views = list(envs), list(tasks), list(views)
print("MEM1 views at boot:", [[hex(v) for v in b] for b in views], flush=True)


def fresh_vis(t, vs):
    out = {}
    for v in vs:
        try:
            out[hex(v)] = struct.unpack(">I", t.read(v + VI_OFF, 4))[0]
        except OSError as ex:
            out[hex(v)] = f"ERR {ex}"
    return out


def diagnose(i, err):
    e, t = envs[i], tasks[i]
    pid = e.inst.proc.pid
    view_vi = struct.unpack_from(">I", e.ram, VI_COUNTER & 0x3FFFFF)[0]
    fresh_a = fresh_vis(t, views[i])
    now_views = find_mem1_views(t)
    time.sleep(1.0)
    view_vi2 = struct.unpack_from(">I", e.ram, VI_COUNTER & 0x3FFFFF)[0]
    fresh_b = fresh_vis(t, now_views)
    ps = subprocess.run(["ps", "-o", "%cpu=,state=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    smp = Path(f"runs/stall_sample_env{i}_{int(time.time())}.txt")
    subprocess.run(["sample", str(pid), "1", "-file", str(smp)], capture_output=True)
    print(f"\n### STALL env{i}: {err}\n"
          f"  our zero-copy view: vi {view_vi} -> {view_vi2} (1 s later)\n"
          f"  fresh mach reads @boot views: {fresh_a}\n"
          f"  views now: {[hex(v) for v in now_views]} -> fresh vi 1 s later: {fresh_b}\n"
          f"  dolphin ps (%cpu state): {ps} | stack sample: {smp}", flush=True)


t0 = time.time()
steps, stalls, resets = [0] * N, 0, 0
for e in envs:
    e.reset(random.choice([1, 2, 3]))
    resets += 1
reset_every = [random.randint(20, 80) for _ in range(N)]
while time.time() - t0 < DURATION:
    if TORCH_LOAD:
        with torch.no_grad():
            net.act(torch.randn(N, 44))
        if time.time() - last_burst > 20:          # ~PPO update: 4 epochs x 8 minibatches over 5400 samples
            tb = time.time()
            opt = torch.optim.Adam(net.parameters())
            for _ in range(32):
                _, lp, ent, v = net.act(torch.randn(675, 44), torch.randint(0, N_ACTIONS, (675,)))
                (lp.mean() + v.pow(2).mean()).backward(); opt.step(); opt.zero_grad()
            print(f"[torch burst {time.time() - tb:.2f}s]", flush=True)
            last_burst = time.time()
    for e in envs:
        e.send_action(random.randrange(N_ACTIONS))
    pending = set(range(N))
    while pending:
        for i in list(pending):
            try:
                if envs[i].ready():
                    envs[i].observe()
                    steps[i] += 1
                    pending.discard(i)
                    if steps[i] % reset_every[i] == 0:
                        envs[i].reset(random.choice([1, 2, 3]))
                        resets += 1
            except EnvError as err:
                stalls += 1
                diagnose(i, err)
                pending.discard(i)
                envs[i].close()
                envs[i], tasks[i], views[i] = make(i, f"r{stalls}")
                envs[i].reset(1)
        time.sleep(0.0005)
    if int(time.time() - t0) % 30 == 0 and sum(steps) % 50 < N:
        print(f"[{time.time() - t0:5.0f}s] steps {steps} resets {resets} stalls {stalls}", flush=True)
print(f"\nDONE {time.time() - t0:.0f}s steps {steps} resets {resets} stalls {stalls}", flush=True)
for e in envs:
    e.close()
