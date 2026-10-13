"""zelda-RL RunPod fleet CLI.

    python -m fleet create    --experiment E --count 1 [--vcpu 16] [--confirm]   # dry run without --confirm
    python -m fleet bootstrap --experiment E            # apt deps, code, NAND + savestates, Dolphin, venv, secrets
    python -m fleet assets    --experiment E [--upload] [--states-only]   # S3 -> pods (presigned URLs, datacenter speed)
    python -m fleet verify    --experiment E [--instances 8]   # acceptance test on every pod (load, input, speed)
    python -m fleet launch    --experiment E [--num-envs N] [--train-args "..."]
    python -m fleet status    --experiment E            # last update line + dolphin count + load per pod
    python -m fleet sync      --experiment E            # checkpoints/logs -> runs/<run>/, newest -> checkpoints/inbox/
    python -m fleet ssh       --experiment E --rank 0 -- <cmd>
    python -m fleet list                                # all zelda-RL pods on the account, with $/hr
    python -m fleet kill      --experiment E [--confirm]

Pod layout: /workspace/zelda-RL (code, assets/, runs/, logs/), /workspace/dolphin/install, /workspace/venv.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fleet.runpod_api import GPU_IMAGE

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / "runs" / "fleet"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
LOCAL_DOLPHIN_USER = Path.home() / "Library/Application Support/Dolphin"
DOLPHIN_TARBALL = Path.home() / ".cache/zelda-rl/dolphin-2603a-linux-x86_64.tgz"
R_REPO, R_DOLPHIN, R_VENV = "/workspace/zelda-RL", "/workspace/dolphin", "/workspace/venv"
R_DOLPHIN_BIN = f"{R_DOLPHIN}/install/bin/dolphin-emu"
POD_ENV = (f"export DOLPHIN_BIN={R_DOLPHIN_BIN} DOLPHIN_HOST_USER_DIR={R_REPO}/assets/dolphin_user "
           f"PYTHONUNBUFFERED=1; unset QT_QPA_PLATFORM")
# VERIFIED on a pod: DolphinQt under QT_QPA_PLATFORM=offscreen starts but never boots the game (no RAM arena).
# Under Xvfb it boots, and linuxmem finds /dev/shm/dolphin-emu.<pid> + MEM1. All instances share one Xvfb display.
XVFB = "xvfb-run -a -s '-screen 0 640x480x24'"
PY_DEPS = "numpy lz4 wandb matplotlib requests boto3"   # boto3: pods stream datasets to S3 as they collect
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
START_SLOTS = tuple(range(1, 11))  # all verified slots (scripts/verify_slots.py); 5 is the goal
FLEET_KEY = Path.home() / ".cache/zelda-rl/fleet_ed25519"      # pod->pod key: seeds rsync assets to new pods
R_FLEET_KEY = "/root/.ssh/zrl_fleet"


def ensure_fleet_key() -> str:
    if not FLEET_KEY.exists():
        FLEET_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "zrl-fleet", "-f", str(FLEET_KEY)],
                       check=True, capture_output=True)
    return FLEET_KEY.with_suffix(".pub").read_text().strip()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- local fleet state --------------------------------------------------------------------------
def state_path(exp: str) -> Path:
    return STATE_DIR / f"{exp}.json"


def load_state(exp: str) -> dict:
    p = state_path(exp)
    return json.loads(p.read_text()) if p.exists() else {"experiment": exp, "pods": []}


def save_state(st: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(st["experiment"]).write_text(json.dumps(st, indent=2))


# ---- ssh helpers ---------------------------------------------------------------------------------
def ssh_base(pod: dict) -> list[str]:
    return ["ssh", "-i", str(SSH_KEY), "-p", str(pod["port"]), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20", f"root@{pod['ip']}"]


def run(pod: dict, cmd: str, *, check: bool = True, capture: bool = False, timeout: float = 3600) -> str:
    for attempt in range(3):
        r = subprocess.run(ssh_base(pod) + [cmd], text=True, capture_output=capture, timeout=timeout)
        if r.returncode != 255:          # 255 = ssh transport failure -> retry
            break
        log(f"  rank{pod['rank']}: ssh transport error, retry {attempt + 1}/3")
        time.sleep(10)
    if check and r.returncode != 0:
        raise RuntimeError(f"rank{pod['rank']} command failed ({r.returncode}): {cmd[:120]}\n{(r.stderr or '')[-2000:]}")
    return r.stdout if capture else ""


def rsync(pod: dict, src: list[str], dest: str, *, pull: bool = False, extra: list[str] | None = None) -> None:
    rsh = f"ssh -i {SSH_KEY} -p {pod['port']} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
    remote = f"root@{pod['ip']}:"
    args = ["rsync", "-rlptz", "--no-owner", "--no-group", "-e", rsh, *(extra or [])]
    args += ([remote + s for s in src] + [dest]) if pull else (src + [remote + dest])
    subprocess.run(args, check=True)


def for_pods(pods: list[dict], fn, workers: int = 8) -> list:
    with ThreadPoolExecutor(max(1, min(workers, len(pods)))) as ex:
        return list(ex.map(fn, pods))


def select(st: dict, ranks: str | None) -> list[dict]:
    pods = st["pods"]
    if ranks:
        want = {int(r) for r in ranks.split(",")}
        pods = [p for p in pods if p["rank"] in want]
    if not pods:
        raise SystemExit(f"no pods for experiment {st['experiment']!r} (see runs/fleet/)")
    return pods


# ---- commands ------------------------------------------------------------------------------------
def cmd_create(a) -> None:
    from fleet.runpod_api import RunPod
    st = load_state(a.experiment)
    used = {p["rank"] for p in st["pods"]}
    free = [r for r in range(len(used) + a.count) if r not in used][:a.count]     # fill gaps so ranks stay 0..N-1
    plan = [f"zrl-{a.experiment}-r{r}" for r in free]
    if a.gpu:
        log(f"plan: {a.count} GPU pod(s) {plan} | {a.gpu_count}x {a.gpu} | {a.cloud} | "
            f"disk {a.container_disk}+{a.volume_gb}GB")
    else:
        log(f"plan: {a.count} CPU pod(s) {plan} | {a.vcpu} vCPU {a.flavors} | {a.cloud} | image {a.image}")
    if not a.confirm:
        log("dry run (add --confirm to create and start paying)")
        return
    rp = RunPod()
    pub = (SSH_KEY.with_suffix(".pub")).read_text().strip() + "\n" + ensure_fleet_key()

    def make(i_name):
        i, name = i_name
        for attempt in range(3):
            pod = None
            try:
                if a.gpu:
                    pod = rp.create_gpu_pod(name, pub, gpu_type_ids=a.gpu.split(","), gpu_count=a.gpu_count,
                                            image=a.gpu_image, container_disk_gb=a.container_disk,
                                            volume_gb=a.volume_gb, cloud=a.cloud)
                else:
                    pod = rp.create_cpu_pod(name, pub, vcpu=a.vcpu, flavors=a.flavors.split(","), image=a.image,
                                            container_disk_gb=a.container_disk, volume_gb=a.volume_gb,
                                            cloud=a.cloud)
                log(f"  created {name}: id={pod.get('id')} costPerHr={pod.get('costPerHr')} vcpu={pod.get('vcpuCount')} "
                    f"mem={pod.get('memoryInGb')}GB")
                ip, port, full = rp.wait_ssh(pod["id"], log=log, timeout=a.ssh_timeout)
                return {"rank": free[i], "name": name, "id": pod["id"], "ip": ip, "port": port,
                        "cost_per_hr": full.get("costPerHr"), "vcpu": full.get("vcpuCount"), "mem_gb": full.get("memoryInGb")}
            except Exception as e:  # noqa: BLE001
                log(f"  {name}: create attempt {attempt + 1}/3 failed: {e}")
                if pod and pod.get("id"):          # created but never got SSH: terminate it, don't leak a billed pod
                    try:
                        rp.delete(pod["id"])
                        log(f"  {name}: terminated stuck pod {pod['id']}")
                    except Exception as de:  # noqa: BLE001
                        log(f"  {name}: could not terminate stuck pod {pod['id']}: {de}")
                time.sleep(20)
        return None

    made = [p for p in for_pods(list(enumerate(plan)), make) if p]
    st["pods"] = sorted(st["pods"] + made, key=lambda p: p["rank"])
    save_state(st)
    total = sum(p["cost_per_hr"] or 0 for p in st["pods"])
    log(f"{len(made)}/{a.count} pods ready | experiment total ${total:.2f}/hr | state: {state_path(a.experiment)}")


def cmd_bootstrap(a) -> None:
    st = load_state(a.experiment)

    seeds = [p for p in st["pods"] if a.via_seeds and p["rank"] in {int(x) for x in a.via_seeds.split(",")}]

    def boot(pod):
        r = pod["rank"]
        log(f"rank{r}: apt base (rsync, tmux, python venv)")
        run(pod, "export DEBIAN_FRONTEND=noninteractive; (command -v rsync && command -v tmux && python3 -m venv --help) "
                 ">/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync tmux python3 python3-venv python3-pip)")
        run(pod, f"mkdir -p {R_REPO}/assets/states {R_REPO}/assets/dolphin_user {R_REPO}/logs {R_DOLPHIN}")
        usable_seeds = [sd for sd in seeds if sd["ip"] != pod["ip"] and sd["rank"] != r]   # same public IP = NAT hairpin
        if a.assets_s3:
            log(f"rank{r}: push code")
            rsync(pod, [f"{REPO}/"], f"{R_REPO}/", extra=CODE_EXCLUDES)
            push_policies(pod)
            s3_pull_assets(pod)
        elif usable_seeds:
            sd = usable_seeds[r % len(usable_seeds)]
            t0 = time.time()
            log(f"rank{r}: seed rank{sd['rank']} -> code + NAND + savestates + Dolphin (datacenter link)")
            seed_push(sd, pod)
            push_policies(pod)
            log(f"rank{r}: seeded in {time.time() - t0:.0f}s")
        else:
            log(f"rank{r}: push code")
            rsync(pod, [f"{REPO}/"], f"{R_REPO}/", extra=CODE_EXCLUDES)
            push_policies(pod)
            log(f"rank{r}: push NAND + savestates")
            rsync(pod, [str(LOCAL_DOLPHIN_USER / "Wii")], f"{R_REPO}/assets/dolphin_user/")
            rsync(pod, [str(LOCAL_DOLPHIN_USER / "StateSaves" / f"NACE01.s{k:02d}") for k in START_SLOTS
                        if (LOCAL_DOLPHIN_USER / "StateSaves" / f"NACE01.s{k:02d}").exists()], f"{R_REPO}/assets/states/")
        log(f"rank{r}: Dolphin runtime/build deps (apt)")
        run(pod, f"bash {R_REPO}/scripts/pod_build_dolphin.sh --deps-only > {R_REPO}/logs/apt_deps.log 2>&1 "
                 f"|| (tail -30 {R_REPO}/logs/apt_deps.log; exit 1)")
        if run(pod, f"SKIP_APT=1 DOLPHIN_ROOT={R_DOLPHIN} bash {R_REPO}/scripts/pod_build_dolphin.sh --check >/dev/null 2>&1 "
                    f"&& echo yes || echo no", capture=True).strip() == "yes":
            log(f"rank{r}: Dolphin already installed")
        elif DOLPHIN_TARBALL.exists():
            log(f"rank{r}: push cached Dolphin build ({DOLPHIN_TARBALL.stat().st_size / 1e6:.0f} MB)")
            rsync(pod, [str(DOLPHIN_TARBALL)], f"{R_DOLPHIN}/")
            run(pod, f"cd {R_DOLPHIN} && tar xzf {DOLPHIN_TARBALL.name} && rm {DOLPHIN_TARBALL.name}")
        else:
            log(f"rank{r}: building Dolphin 2603a from source (tmux 'dolphin_build', log {R_DOLPHIN}/build.log)")
            run(pod, f"tmux kill-session -t dolphin_build 2>/dev/null; tmux new-session -d -s dolphin_build "
                     f"\"SKIP_APT=1 DOLPHIN_ROOT={R_DOLPHIN} bash {R_REPO}/scripts/pod_build_dolphin.sh > {R_DOLPHIN}/build.log 2>&1; "
                     f"echo BUILD_EXIT=\\$? >> {R_DOLPHIN}/build.log\"")
            t0 = time.time()
            while run(pod, "tmux has-session -t dolphin_build 2>/dev/null && echo run || echo done", capture=True).strip() == "run":
                time.sleep(60)
                tail = run(pod, f"tail -n1 {R_DOLPHIN}/build.log | cut -c1-140", capture=True, check=False).strip()
                log(f"rank{r}: build {time.time() - t0:.0f}s | {tail}")
            if run(pod, f"grep -q '^BUILD_EXIT=0' {R_DOLPHIN}/build.log && echo ok || echo fail", capture=True).strip() != "ok":
                print(run(pod, f"tail -40 {R_DOLPHIN}/build.log", capture=True, check=False))
                raise RuntimeError(f"rank{r}: Dolphin build failed")
            log(f"rank{r}: build OK in {time.time() - t0:.0f}s -> caching tarball on the Mac")
            run(pod, f"cd {R_DOLPHIN} && tar czf /workspace/{DOLPHIN_TARBALL.name} install")
            DOLPHIN_TARBALL.parent.mkdir(parents=True, exist_ok=True)
            rsync(pod, [f"/workspace/{DOLPHIN_TARBALL.name}"], str(DOLPHIN_TARBALL.parent) + "/", pull=True)
        # PREBUILT VENV from S3 when available. Creating the venv and pip-installing torch is ~4-5 min per pod
        # and, once Dolphin comes from S3 pre-built, it is the ENTIRE remaining bootstrap cost -- the wall that
        # kept a 116-pod fleet from collecting. Baked once (scripts/bake_venv.py style: a pod tars its own venv
        # and uploads it), pulled as a download by everyone else. Falls back to pip if the tarball is absent.
        if run(pod, f"{R_VENV}/bin/python -c 'import torch,boto3' 2>/dev/null && echo yes || echo no",
               capture=True, check=False).strip() == "yes":
            log(f"rank{r}: venv already present")
        elif venv_in_s3():
            log(f"rank{r}: pull prebuilt venv from S3 ({VENV_MB:.0f} MB)")
            import boto3 as _b3
            _s3 = _b3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
            url = _s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": VENV_KEY},
                                             ExpiresIn=6 * 3600)
            run(pod, f"cd /workspace && curl -sfL --retry 5 -o venv.tgz {shlex.quote(url)} && "
                     f"rm -rf {R_VENV} && tar xzf venv.tgz && rm venv.tgz && "
                     f"{R_VENV}/bin/python -c 'import torch,boto3'")
        else:
            log(f"rank{r}: python venv (pip; no prebuilt tarball in S3)")
            run(pod, f"[ -x {R_VENV}/bin/python ] || python3 -m venv {R_VENV}; "
                     f"{R_VENV}/bin/python -c 'import torch' 2>/dev/null || {R_VENV}/bin/pip install -q --index-url {TORCH_INDEX} torch; "
                     f"{R_VENV}/bin/pip install -q {PY_DEPS}")
        push_secrets(pod)
        ping = run(pod, f"set -a; . {R_REPO}/.fleet_secrets; set +a; {R_VENV}/bin/python -c \"import os,time,requests; "
                        f"t=time.time(); r=requests.post(os.environ['UPSTASH_REDIS_REST_URL'], json=['PING'], "
                        f"headers={{'Authorization': 'Bearer ' + os.environ['UPSTASH_REDIS_REST_TOKEN']}}, timeout=(5, 10)); "
                        f"print(r.json().get('result'), round(time.time() - t, 2))\"", capture=True, check=False).strip()
        if not ping.startswith("PONG"):
            raise RuntimeError(f"rank{r}: pod cannot reach Upstash ({ping[-200:] or 'timeout'}) -- kill and recreate this pod")
        log(f"rank{r}: Upstash reachable ({ping})")
        info = run(pod, f"nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; df -h /dev/shm | tail -1; free -g | sed -n 2p; "
                        f"{R_VENV}/bin/python -c 'import torch; print(\"torch\", torch.__version__)'; "
                        f"SKIP_APT=1 DOLPHIN_ROOT={R_DOLPHIN} bash {R_REPO}/scripts/pod_build_dolphin.sh --check | tail -2",
                   capture=True, check=False)
        log(f"rank{r}: bootstrap done\n    " + info.strip().replace("\n", "\n    "))
        return True

    pods = select(st, a.ranks)
    if seeds:
        log(f"seeds: {[sd['rank'] for sd in seeds]} -> authorizing fleet key on {len(st['pods'])} pods")
        setup_seed_access(st["pods"], seeds)
        pods = [p for p in pods if p["rank"] not in {sd["rank"] for sd in seeds}]
        for_pods(pods, boot, workers=a.workers)
        return
    if not DOLPHIN_TARBALL.exists() and len(pods) > 1:
        log("no cached Dolphin build yet: bootstrapping rank %d first (builds + caches), then the rest" % pods[0]["rank"])
        boot(pods[0])
        pods = pods[1:]
    for_pods(pods, boot)


S3_BUCKET, S3_REGION, S3_PROFILE = "zelda-rl", "us-west-1", "race-wm"
# Start states in S3, per TASK. Defaulted to gohma, which silently gave a Deku Baba collection run 10 slots
# with ZERO live Babas ("no live Deku Baba -> wrong savestates"), so it is overridable: --states-prefix.
S3_STATES_PREFIX = "zelda/boss/gohma/states"          # current training start states (slots 1-10)
S3_STATES_DEKU = "zelda/combat/deku-baba/states"      # what a Deku Baba collection needs
S3_FLEET_PREFIX = "fleet"                             # Dolphin build + NAND


def set_states_prefix(prefix: str) -> None:
    global S3_STATES_PREFIX
    S3_STATES_PREFIX = prefix


def s3_asset_keys() -> dict[str, str]:
    """S3 key -> pod destination path for every bootstrap asset."""
    keys = {f"{S3_FLEET_PREFIX}/{DOLPHIN_TARBALL.name}": f"{R_DOLPHIN}/{DOLPHIN_TARBALL.name}",
            f"{S3_FLEET_PREFIX}/nand-Wii.tgz": f"{R_REPO}/assets/dolphin_user/nand-Wii.tgz"}
    keys.update({f"{S3_STATES_PREFIX}/NACE01.s{k:02d}": f"{R_REPO}/assets/states/NACE01.s{k:02d}" for k in START_SLOTS})
    return keys


def s3_upload_assets() -> None:
    """Mac -> S3, once (multipart, parallel). Skips objects whose size already matches."""
    import boto3
    import tarfile
    s3 = boto3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
    nand = REPO / "runs" / "fleet" / "s3stage" / "nand-Wii.tgz"
    nand.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nand, "w:gz") as tf:
        tf.add(LOCAL_DOLPHIN_USER / "Wii", arcname="Wii")
    local = {f"{S3_FLEET_PREFIX}/{DOLPHIN_TARBALL.name}": DOLPHIN_TARBALL, f"{S3_FLEET_PREFIX}/nand-Wii.tgz": nand}
    local.update({f"{S3_STATES_PREFIX}/NACE01.s{k:02d}": LOCAL_DOLPHIN_USER / "StateSaves" / f"NACE01.s{k:02d}" for k in START_SLOTS})
    for key, path in local.items():
        try:
            same = s3.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"] == path.stat().st_size and not key.startswith(S3_STATES_PREFIX)
        except Exception:  # noqa: BLE001
            same = False
        if same:
            log(f"s3: {key} up to date")
            continue
        t0 = time.time()
        s3.upload_file(str(path), S3_BUCKET, key)
        log(f"s3: uploaded {key} ({path.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")


def s3_pull_assets(pod: dict, only_states: bool = False) -> None:
    """Pod <- S3 over the datacenter link, with presigned GET URLs (no AWS credentials on the pod)."""
    import boto3
    s3 = boto3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
    items = [(k, v) for k, v in s3_asset_keys().items() if not only_states or k.startswith(S3_STATES_PREFIX)]
    lines = ["set -e", f"mkdir -p {R_REPO}/assets/states {R_REPO}/assets/dolphin_user {R_DOLPHIN}"]
    for key, dest in items:
        url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=6 * 3600)
        lines.append(f"curl -sfL --retry 5 -o {shlex.quote(dest)}.part {shlex.quote(url)} && mv {shlex.quote(dest)}.part {shlex.quote(dest)} &")
    lines += ["wait",
              f"tar -C {R_REPO}/assets/dolphin_user -xzf {R_REPO}/assets/dolphin_user/nand-Wii.tgz" if not only_states else "true",
              f"cd {R_DOLPHIN} && tar xzf {DOLPHIN_TARBALL.name} && rm {DOLPHIN_TARBALL.name}" if not only_states else "true",
              f"ls {R_REPO}/assets/states | wc -l"]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
        f.write("\n".join(lines) + "\n")
        tmp = f.name
    rsync(pod, [tmp], "/workspace/s3_pull.sh")
    Path(tmp).unlink(missing_ok=True)
    t0 = time.time()
    n = run(pod, "bash /workspace/s3_pull.sh && rm /workspace/s3_pull.sh", capture=True).strip().splitlines()[-1]
    log(f"rank{pod['rank']}: pulled {'savestates' if only_states else 'Dolphin + NAND + savestates'} from S3 in {time.time() - t0:.0f}s ({n} states)")


def cmd_assets(a) -> None:
    st = load_state(a.experiment)
    if getattr(a, "states_prefix", None):
        set_states_prefix(a.states_prefix)
        log(f"states prefix: s3://{S3_BUCKET}/{S3_STATES_PREFIX}")
    if a.upload:
        s3_upload_assets()
    for_pods(select(st, a.ranks), lambda pod: s3_pull_assets(pod, only_states=a.states_only), workers=24)


def setup_seed_access(pods: list[dict], seeds: list[dict]) -> None:
    """Authorize the fleet key on every pod (idempotent) and install its private half on the seeds."""
    pub = ensure_fleet_key()

    def auth(pod):
        run(pod, f"mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys && "
                 f"(grep -qF '{pub.split()[1]}' /root/.ssh/authorized_keys || echo '{pub}' >> /root/.ssh/authorized_keys)")
    for_pods(pods, auth)
    for seed in seeds:
        rsync(seed, [str(FLEET_KEY)], R_FLEET_KEY)
        run(seed, f"chmod 600 {R_FLEET_KEY}")


def seed_push(seed: dict, target: dict) -> None:
    """Seed pod rsyncs code + assets (NAND, savestates) + Dolphin install to the target over the datacenter link."""
    rsh = (f"ssh -i {R_FLEET_KEY} -p {target['port']} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
           f"-o ServerAliveInterval=30")
    base = f"rsync -rlptz --no-owner --no-group -e '{rsh}'"
    run(seed, f"{base} --exclude runs/ --exclude logs/ --exclude .fleet_secrets --exclude __pycache__/ --exclude recordings/ "
              f"{R_REPO}/ root@{target['ip']}:{R_REPO}/ && "
              f"{base} {R_DOLPHIN}/install root@{target['ip']}:{R_DOLPHIN}/", timeout=3600)


def push_secrets(pod: dict) -> None:
    from fleet.secrets import pod_secrets_file_content
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(pod_secrets_file_content())
        tmp = f.name
    try:
        subprocess.run(["chmod", "600", tmp], check=True)
        rsync(pod, [tmp], f"{R_REPO}/.fleet_secrets")
        run(pod, f"chmod 600 {R_REPO}/.fleet_secrets")
    finally:
        Path(tmp).unlink(missing_ok=True)


def cmd_verify(a) -> None:
    st = load_state(a.experiment)

    def verify(pod):
        extra = f" --instances {a.instances} --unlimited" if a.instances else ""
        out = run(pod, f"{POD_ENV}; cd {R_REPO} && {XVFB} {R_VENV}/bin/python -u scripts/verify_pod_dolphin.py "
                       f"--states-dir assets/states --backend Null{extra} 2>&1 | tee logs/verify.log; exit ${{PIPESTATUS[0]}}",
                  capture=True, check=False, timeout=1800)
        return pod["rank"], out

    for rank, out in for_pods(select(st, a.ranks), verify):
        print(f"\n===== rank{rank} verify =====\n{out[-6000:]}")


VENV_KEY = "fleet/venv-linux-x86_64.tgz"
VENV_MB = 0.0


def venv_in_s3() -> bool:
    """Is a prebuilt venv tarball in S3? Cached, so 116 pods cause one HEAD request."""
    global VENV_MB
    if not hasattr(venv_in_s3, "_v"):
        try:
            import boto3
            s3 = boto3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
            VENV_MB = s3.head_object(Bucket=S3_BUCKET, Key=VENV_KEY)["ContentLength"] / 1e6
            venv_in_s3._v = True
        except Exception:
            venv_in_s3._v = False
    return venv_in_s3._v


def mix_policy_paths():
    """Repo-relative checkpoints that a collection mix names.

    CODE_EXCLUDES deliberately drops checkpoints/ and *.pt -- they are large and a trainer does not need them.
    A COLLECTION run does: `resolve_checkpoint` fails on the pod without the exact policies its mix references.
    Read from the mix definitions rather than hardcoded, so adding a checkpoint to a mix cannot silently break
    a fleet run.
    """
    try:
        from scripts.collect_transitions import BAD_PLAY_MIX, DEFAULT_MIX
    except Exception:
        return []
    return sorted({p for p, _, _ in list(BAD_PLAY_MIX) + list(DEFAULT_MIX) if (REPO / p).exists()})


def push_policies(pod) -> None:
    """rsync just the mix checkpoints, preserving their repo-relative paths (rsync -R)."""
    rel = mix_policy_paths()
    if not rel:
        return
    mb = sum((REPO / r).stat().st_size for r in rel) / 1e6
    # S3 FIRST. Pushing 47 MB from the Mac to each of 78 pods is 3.7 GB over a ~0.7 MB/s uplink -- about 88
    # minutes, and it silently stalled a 116-pod bootstrap. This is rule 15 in AGENTS.md, violated by the very
    # function written to fix rule 16. Presigned GETs make it a parallel datacenter download instead.
    try:
        import boto3
        s3 = boto3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
        keys = {r: f"fleet/policies/{r}" for r in rel}
        missing = []
        for r, k in keys.items():
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=k)
            except Exception:
                missing.append(r)
        if not missing:
            lines = ["set -e"] + [f"mkdir -p {R_REPO}/{d}" for d in sorted({str(Path(r).parent) for r in rel})]
            for r, k in keys.items():
                url = s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": k},
                                                ExpiresIn=6 * 3600)
                lines.append(f"curl -sfL --retry 5 -o {R_REPO}/{r} {shlex.quote(url)} &")
            lines.append("wait")
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
                f.write("\n".join(lines) + "\n")
                tf = f.name
            rsync(pod, [tf], "/workspace/pol_pull.sh")
            Path(tf).unlink(missing_ok=True)
            run(pod, "bash /workspace/pol_pull.sh && rm /workspace/pol_pull.sh")
            log(f"rank{pod['rank']}: pulled {len(rel)} mix policies from S3")
            return
    except Exception as e:  # noqa: BLE001
        log(f"rank{pod['rank']}: S3 policy pull unavailable ({str(e)[:60]}), pushing from the Mac")
    log(f"rank{pod['rank']}: push {len(rel)} mix policies ({mb:.0f} MB)")
    # Explicit destination per file. `rsync -R` with a /./ marker was supposed to preserve the repo-relative
    # path but rsync kept the ABSOLUTE one, so the files landed at
    # /workspace/zelda-RL/Users/jordy/Documents/dev/zelda-RL/checkpoints/... and resolve_checkpoint failed.
    dirs = sorted({str(Path(r).parent) for r in rel})
    run(pod, "mkdir -p " + " ".join(f"{R_REPO}/{d}" for d in dirs))
    for d in dirs:
        rsync(pod, [str(REPO / r) for r in rel if str(Path(r).parent) == d], f"{R_REPO}/{d}/")


CODE_EXCLUDES = ["--exclude", ".venv/", "--exclude", "runs/", "--exclude", "assets/", "--exclude", "__pycache__/",
                 # results/ holds the Mac's pulled evaluation rows, including WITHDRAWN ones. Pushing them to
                 # every pod meant a later pull brought them home nested one level deeper, outside the
                 # .withdrawn marker, where a pooling script counted them as real measurements. Pods WRITE
                 # results; they never read the Mac's.
                 "--exclude", "results/",
                 "--exclude", "recordings/", "--exclude", "*.mp4", "--exclude", "*.mov",
                 "--exclude", "checkpoints/", "--exclude", ".git/", "--exclude", "*.pt", "--exclude", ".DS_Store",
                 # datasets/ was NOT excluded, so 52 GB of collected episodes was pushed to every pod. Pods
                 # GENERATE data; they never read the local dataset tree. `instances/` are the per-env Dolphin
                 # working dirs those collections leave behind, and they contain FIFO pipes that rsync warns on.
                 "--exclude", "datasets/", "--exclude", "instances/", "--exclude", "wandb/"]


def cmd_launch(a) -> None:
    st = load_state(a.experiment)
    launch_id = time.strftime("%Y%m%d%H%M%S")
    try:
        from zelda_rl.fleet_link import clear_sync_keys
        log(f"launch id {launch_id}; cleared {clear_sync_keys(a.experiment)} stale sync keys")
    except Exception as e:  # noqa: BLE001
        log(f"(could not clear sync keys: {e})")
    def prepare(pod):
        rsync(pod, [f"{REPO}/"], f"{R_REPO}/", extra=CODE_EXCLUDES)      # always run the Mac's current code
        push_secrets(pod)
        init = re.search(r"--init-from\s+(\S+)", a.train_args or "")
        if init and "{rank}" not in init.group(1) and (REPO / init.group(1)).is_file():   # local warm-start checkpoint
            rel = init.group(1)
            run(pod, f"mkdir -p {R_REPO}/{Path(rel).parent}")
            rsync(pod, [str(REPO / rel)], f"{R_REPO}/{rel}")
        tag = f"-{a.run_tag}" if a.run_tag else ""
        run_name = f"{a.experiment}{tag}-r{pod['rank']}"
        envs = a.num_envs or max(1, (pod.get("vcpu") or 2) // 2)
        if a.collect:
            # A COLLECTOR is not a trainer: scripts/collect_transitions.py has no --run-name, --num-envs or
            # --fleet-* flags, so passing them aborts on argparse before a single episode. Emit the bare
            # command and let --train-args carry everything.
            script = (f"#!/bin/bash\nset -a; . {R_REPO}/.fleet_secrets; set +a\n{POD_ENV}\ncd {R_REPO}\n"
                      f"exec {XVFB} {R_VENV}/bin/python -u scripts/{a.script} {a.train_args} "
                      f"> logs/{run_name}.log 2>&1\n")
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
                f.write(script)
                tmpf = f.name
            run(pod, f"mkdir -p {R_REPO}/logs")
            rsync(pod, [tmpf], f"{R_REPO}/logs/launch_{run_name}.sh")
            Path(tmpf).unlink(missing_ok=True)
            run(pod, f"chmod +x {R_REPO}/logs/launch_{run_name}.sh; tmux kill-session -t zrl 2>/dev/null; "
                     f"pkill -x dolphin-emu; sleep 1; "
                     f"tmux new-session -d -s zrl '{R_REPO}/logs/launch_{run_name}.sh'")
            pod["run_name"] = run_name
            log(f"rank{pod['rank']}: collector started (tmux zrl), log logs/{run_name}.log")
            return
        script = (f"#!/bin/bash\nset -a; . {R_REPO}/.fleet_secrets; set +a\n{POD_ENV}\ncd {R_REPO}\n"
                  f"exec {XVFB} {R_VENV}/bin/python -u scripts/{a.script} --run-name {run_name} --num-envs {envs} "
                  f"--slots-dir assets/states --video-backend Null --seed {pod['rank']} "
                  f"--wandb-project {a.wandb_project} --wandb-group {a.experiment}{tag} --fleet-experiment {a.experiment} "
                  f"--fleet-rank {pod['rank']} --fleet-world-size {len(st['pods'])} --fleet-launch-id {launch_id} "
                  f"{a.train_args} > logs/train_{run_name}.log 2>&1\n")
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
            f.write(script)
            tmp = f.name
        run(pod, f"mkdir -p {R_REPO}/logs")
        rsync(pod, [tmp], f"{R_REPO}/logs/launch_{run_name}.sh")
        Path(tmp).unlink(missing_ok=True)
        run(pod, f"tmux kill-session -t train 2>/dev/null; pkill -x dolphin-emu; sleep 1; rm -rf {R_REPO}/runs/instances {R_REPO}/runs/*/instances")
        pod["run_name"], pod["_envs"] = run_name, envs
        return pod

    want = select(st, a.ranks)
    if not a.skip_preflight:
        ok, bad = preflight(want)
        if bad:
            log(f"PREFLIGHT: {len(bad)} of {len(want)} pods are not ready and will be SKIPPED")
            for r in sorted(bad)[:12]:
                log(f"  rank{r}: {', '.join(bad[r])}")
            log("  fix them with `fleet bootstrap --assets-s3 --ranks <those>` then relaunch "
                "(or pass --skip-preflight)")
        want = [p for p in want if p["rank"] in set(ok)]
        if not want:
            raise SystemExit("no ready pods; nothing launched")
        log(f"launching on {len(want)} ready pods")
    pods = for_pods(want, prepare, workers=24)           # code/secrets/scripts first, in parallel
    if a.collect:
        # the collect branch already started its own tmux session and returned None; the trainer start below
        # would index that None
        save_state(st)
        return
    for_pods(pods, lambda pod: run(pod, f"tmux new-session -d -s train 'bash {R_REPO}/logs/launch_{pod['run_name']}.sh'"),
             workers=24)                                                # then start every rank within seconds
    for pod in pods:
        log(f"rank{pod['rank']}: launched {pod['run_name']} with {pod.pop('_envs')} envs")
    save_state(st)


def cmd_status(a) -> None:
    st = load_state(a.experiment)
    try:
        from zelda_rl.fleet_link import heartbeats
        hbs = heartbeats(a.experiment)
        print(f"Upstash heartbeats ({len(hbs)} live):")
        for h in hbs:
            print(f"  rank{h['rank']} {h.get('run', '')} upd {h.get('update')} steps {h.get('global_step', 0):,} "
                  f"succ {h.get('success_rate', 0):.0%} final dist {h.get('final_pos_dist', float('nan')):.0f} "
                  f"min dist {h.get('min_pos_dist', float('nan')):.0f} rupees {h.get('final_rupees', float('nan')):.1f} "
                  f"sps {h.get('env_steps_per_sec', 0):.0f} emu {h.get('emu_speed_x', 0):.1f}x hash {h.get('weight_hash')} "
                  f"age {time.time() - h['t']:.0f}s")
    except Exception as e:  # noqa: BLE001
        print(f"(no Upstash heartbeats: {e})")

    def status(pod):
        rn = pod.get("run_name", "?")
        cmd = (f"cd {R_REPO}; echo \"dolphins=$(pgrep -xc dolphin-emu) load=$(cut -d' ' -f1-3 /proc/loadavg) "
               f"tmux=$(tmux has-session -t train 2>/dev/null && echo up || echo DOWN)\"; "
               f"grep -E 'INFO +[0-9]+ ' runs/{rn}/train.log 2>/dev/null | tail -n {a.lines}; "
               f"grep -E 'WARN|Error|Traceback' logs/train_{rn}.log 2>/dev/null | tail -n 3")
        return pod, run(pod, cmd, capture=True, check=False, timeout=60)

    for pod, out in for_pods(select(st, a.ranks), status):
        print(f"── rank{pod['rank']} {pod.get('run_name', '')} ({pod['ip']}:{pod['port']}, ${pod.get('cost_per_hr')}/hr)\n{out.rstrip()}")


def cmd_sync(a) -> None:
    st = load_state(a.experiment)
    if a.datasets:
        # Dataset collection, not training: every pod writes its own episodes, so pull each rank into its own
        # subdirectory. `load_many` takes a list of directories and only the FIRST needs dataset.json, so the
        # per-rank layout concatenates without any merge step.
        name = Path(a.datasets.rstrip("/")).name
        total_eps = 0
        for pod in select(st, a.ranks):
            dest = REPO / "datasets" / "transitions" / name / f"rank{pod['rank']:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            rsync(pod, [f"{R_REPO}/{a.datasets.rstrip('/')}/"], str(dest) + "/", pull=True)
            n = len(list((dest / "episodes").glob("*.npz"))) if (dest / "episodes").exists() else 0
            total_eps += n
            log(f"rank{pod['rank']}: {n} episodes -> {dest.relative_to(REPO)}")
        log(f"synced {total_eps} episodes total into datasets/transitions/{name}/")
        return
    inbox = REPO / "checkpoints" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for pod in select(st, a.ranks):
        rn = pod.get("run_name")
        if not rn:
            continue
        dest = REPO / "runs" / rn
        dest.mkdir(parents=True, exist_ok=True)
        rsync(pod, [f"{R_REPO}/runs/{rn}/"], str(dest) + "/", pull=True, extra=["--exclude", "instances/", "--exclude", "wandb/"])
        latest = dest / "checkpoints" / "latest.pt"
        if latest.exists():
            import torch
            ck = torch.load(latest, weights_only=False, map_location="cpu")
            out = inbox / f"{rn}_u{ck['update']:05d}.pt"
            if not out.exists():
                out.write_bytes(latest.read_bytes())
            log(f"rank{pod['rank']}: synced {rn} (update {ck['update']}, best min dist "
                f"{ck.get('best', {}).get('min_pos_dist', float('nan')):.0f}) -> {out.relative_to(REPO)}")
        else:
            log(f"rank{pod['rank']}: synced {rn} (no checkpoint yet)")


def cmd_collected(a) -> None:
    """Total transitions collected across the fleet. Reads each pod's manifest, no dataset transfer."""
    st = load_state(a.experiment)
    out = a.out.rstrip("/")

    # `steps` is the per-episode transition count and `end` the outcome; both are in the manifest, so this
    # needs no dataset transfer. Counting npz files would give episodes but not transitions or the outcome mix,
    # and the mix is the point of the badplay collection (deaths and damage, not clears).
    PROBE = ("import json,collections as c;"
             "r=[json.loads(l) for l in open('manifest.jsonl') if l.strip()];"
             "k=c.Counter(x.get('end') for x in r);"
             "print(len(r),sum(x.get('steps',0) for x in r),"
             "k['success'],k['death'],k['left_room'],k['timeout'])")

    def one(pod):
        try:
            cmd = f"cd {R_REPO}/{out} 2>/dev/null && python3 -c {shlex.quote(PROBE)} || echo 0 0 0 0 0 0"
            r = run(pod, cmd, capture=True, check=False).split()
            v = [int(x) for x in r[:6]] if len(r) >= 6 and all(x.lstrip('-').isdigit() for x in r[:6]) else [0] * 6
            alive = run(pod, "pgrep -c dolphin-emu || echo 0", capture=True, check=False).strip()
            return (pod["rank"], v[0], v[1], alive, v[2], v[3], v[4], v[5])
        except Exception:
            return (pod["rank"], 0, 0, "?", 0, 0, 0, 0)

    res = sorted(for_pods(select(st, a.ranks), one, workers=24))
    tot_e = sum(x[1] for x in res if x[1] > 0)
    tot_t = sum(x[2] for x in res if x[2] > 0)
    su, de, lf, to = (sum(x[i] for x in res) for i in (4, 5, 6, 7))
    dead = [x[0] for x in res if x[3] in ("0", "?")]
    if a.verbose:
        print(f"{'rank':>5} {'episodes':>9} {'transitions':>12} {'dolphins':>9}")
        for x in res:
            print(f"{x[0]:>5} {x[1]:>9} {x[2]:>12,} {x[3]:>9}")
    n = max(su + de + lf + to, 1)
    print(f"\nTOTAL: {tot_t:,} transitions over {tot_e:,} episodes on {len(res)} pods")
    print(f"outcomes: success {su:,} ({su/n:.0%})  death {de:,} ({de/n:.0%})  "
          f"left_room {lf:,} ({lf/n:.0%})  timeout {to:,} ({to/n:.0%})")
    if dead:
        print(f"WARNING: {len(dead)} pods with no dolphin running: {dead}")


PREFLIGHT = {
    # name -> (shell test printing 1/0, what a 0 means)
    "dolphin":   (f"test -x {R_DOLPHIN_BIN} && echo 1 || echo 0", "Dolphin binary missing (bootstrap incomplete)"),
    "torch":     (f"{R_VENV}/bin/python -c 'import torch' 2>/dev/null && echo 1 || echo 0", "venv has no torch"),
    "boto3":     (f"{R_VENV}/bin/python -c 'import boto3' 2>/dev/null && echo 1 || echo 0",
                  "no boto3: the collector CANNOT upload to S3 and pod-local storage becomes the only copy"),
    "xvfb":      ("command -v xvfb-run >/dev/null && echo 1 || echo 0", "xvfb-run missing: Dolphin will not boot"),
    "secrets":   (f"test -s {R_REPO}/.fleet_secrets && echo 1 || echo 0", "no .fleet_secrets (no AWS/W&B creds)"),
    "policies":  (f"test $(ls {R_REPO}/checkpoints/*/*.pt 2>/dev/null | wc -l) -ge 9 && echo 1 || echo 0",
                  "fewer than 9 mix policies: resolve_checkpoint will fail on episode 1"),
    "states":    (f"test $(ls {R_REPO}/assets/states 2>/dev/null | wc -l) -ge 10 && echo 1 || echo 0",
                  "fewer than 10 savestates"),
}


def preflight(pods: list[dict], workers: int = 60) -> tuple[list[int], dict]:
    """Per-pod readiness. Returns (ok_ranks, {rank: [failed checks]}).

    WHY THIS IS CODE AND NOT A PARAGRAPH: every failure on this fleet tonight was already written down as a
    rule, and every one was repeated anyway, because a rule in a document has to be REMEMBERED at the moment it
    matters. A check that runs automatically and fails loudly does not. `launch` calls this and refuses to start
    on pods that would fail, so "started: 48" can no longer mean "48 commands dispatched into the void".
    """
    keys = list(PREFLIGHT)
    script = "; ".join(PREFLIGHT[k][0] for k in keys)

    def one(pod):
        try:
            out = run(pod, script, capture=True, check=False).split()
            vals = [x == "1" for x in out[:len(keys)]] if len(out) >= len(keys) else [False] * len(keys)
        except Exception:
            vals = [False] * len(keys)
        return pod["rank"], [k for k, v in zip(keys, vals) if not v]

    res = dict(for_pods(pods, one, workers=workers))
    return sorted(r for r, bad in res.items() if not bad), {r: b for r, b in res.items() if b}


def s3_mirror_gaps(st: dict, out: str, s3_prefix: str) -> tuple[int, int, list]:
    """(on_disk, in_s3, [(rank, disk, s3)]) per pod. The keys MUST be per-rank namespaced."""
    import boto3
    s3 = boto3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
    bkt, pre = s3_prefix.replace("s3://", "").split("/", 1)
    counts = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bkt, Prefix=pre.rstrip("/") + "/"):
        for o in page.get("Contents", []):
            if o["Key"].endswith(".npz"):
                parts = o["Key"][len(pre.rstrip("/")) + 1:].split("/")
                counts[parts[0]] = counts.get(parts[0], 0) + 1

    def disk(pod):
        try:
            return (pod["rank"], int(run(pod, f"ls {R_REPO}/{out}/episodes 2>/dev/null | wc -l",
                                         capture=True, check=False).strip() or 0))
        except Exception:
            return (pod["rank"], -1)

    res = dict(for_pods(st["pods"], disk, workers=60))
    gaps, d_tot, s_tot = [], 0, 0
    for r, nd in sorted(res.items()):
        ns = counts.get(f"rank{r:03d}", 0)
        d_tot += max(nd, 0)
        s_tot += ns
        if nd > ns:
            gaps.append((r, nd, ns))
    return d_tot, s_tot, gaps


def cmd_verify_s3(a) -> None:
    """Per-rank disk-vs-S3 comparison. Run BEFORE kill: pod-local storage is never the only copy."""
    st = load_state(a.experiment)
    d, s, gaps = s3_mirror_gaps(st, a.out, a.s3_prefix)
    print(f"on disk {d:,} episodes | in S3 {s:,} | ~{s * 830 / 1e6:.1f}M transitions durable")
    print(f"pods with a gap: {len(gaps)}")
    for r, nd, ns in gaps[:12]:
        print(f"  rank{r}: disk {nd} s3 {ns}  (missing {nd - ns})")
    if gaps:
        raise SystemExit(1)


def cmd_preflight(a) -> None:
    st = load_state(a.experiment)
    ok, bad = preflight(select(st, a.ranks))
    print(f"READY: {len(ok)}/{len(ok) + len(bad)} pods")
    if bad:
        from collections import Counter
        why = Counter(c for b in bad.values() for c in b)
        for c, n in why.most_common():
            print(f"  {n:>4} pods fail {c:<10} -- {PREFLIGHT[c][1]}")
        print(f"  not ready: {','.join(map(str, sorted(bad)))[:160]}")
        if a.write_bad:
            Path(a.write_bad).write_text(",".join(map(str, sorted(bad))))
            print(f"  -> {a.write_bad}")


def cmd_audit(a) -> None:
    """What is EVERY pod doing right now: collecting / ready-but-idle / bootstrap-incomplete.

    This exists because 78 of 116 pods sat idle and billing for 25 minutes and nothing surfaced it. There was
    no way to see per-pod state without hand-writing an ssh fan-out, so the assumption that "bootstrap ran" was
    never checked against reality. Run this after every phase, not when something already looks wrong.
    """
    st = load_state(a.experiment)

    def one(pod):
        try:
            o = run(pod, f"D=$(test -x {R_DOLPHIN_BIN} && echo 1 || echo 0); "
                         f"T=$({R_VENV}/bin/python -c 'import torch,boto3' 2>/dev/null && echo 1 || echo 0); "
                         f"C=$(pgrep -c dolphin-emu 2>/dev/null || echo 0); "
                         f"P=$(tmux has-session -t zrl 2>/dev/null && echo 1 || echo 0); "   # tmux is true the INSTANT a collector starts; pgrep on dolphin is not (12 instances take >40s to boot, which made a relaunch daemon kill working pods in a loop)
                         f"E=$(ls {R_REPO}/{a.out}/episodes 2>/dev/null | wc -l); echo $D $T $C $P $E",
                    capture=True, check=False).split()
            v = [int(x) for x in o[:5]] if len(o) >= 5 else [0] * 5
            return (pod["rank"], *v)
        except Exception:
            return (pod["rank"], -1, -1, -1, -1, -1)

    res = sorted(for_pods(select(st, a.ranks), one, workers=50))
    busy = [r for r, d, t, c, pr, e in res if c > 0 or pr > 0]
    idle = [r for r, d, t, c, pr, e in res if c == 0 and pr == 0 and d == 1 and t == 1]
    broken = [r for r, d, t, c, pr, e in res if c == 0 and pr == 0 and (d != 1 or t != 1)]
    eps = sum(e for *_, e in res if e > 0)
    cost = sum(p.get("cost_per_hr") or 0 for p in st["pods"])
    print(f"{len(res)} pods, ${cost:.2f}/hr total")
    print(f"  WORKING (dolphin or a script running): {len(busy)}")
    print(f"  READY BUT IDLE -- paying for nothing:  {len(idle)}  {','.join(map(str, idle))[:100]}")
    print(f"  BOOTSTRAP INCOMPLETE:                  {len(broken)}  {','.join(map(str, broken))[:100]}")
    print(f"  episodes on pod disks: {eps:,}")
    if idle or broken:
        waste = (len(idle) + len(broken)) / max(len(res), 1) * cost
        print(f"  ${waste:.2f}/hr is being spent on pods that are doing nothing")
    if a.write_idle:
        Path(a.write_idle).write_text(",".join(map(str, idle + broken)))
        print(f"  -> wrote {len(idle) + len(broken)} ranks to {a.write_idle}")


def cmd_metrics(a) -> None:
    """Training curve per rank, straight from each pod's metrics.jsonl. No wandb, no dataset transfer."""
    st = load_state(a.experiment)
    PROBE = ("import json,sys;"
             "rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()];"
             "rs=rs[-int(sys.argv[2]):];"
             "print('|'.join(f\"{r['epoch']},{r['train']:.4f},{r.get('val_loss',0):.4f},"
             "{r['skill']:+.4f},{r['dcell_acc']:.4f},{r.get('roll/h10_skill',0):+.4f}\" for r in rs))")

    def one(pod):
        try:
            out = run(pod, f"python3 -c {shlex.quote(PROBE)} {R_REPO}/{a.dir}/metrics.jsonl {a.last} "
                           f"2>/dev/null || echo ''", capture=True).strip()
            return (pod["rank"], out)
        except Exception:
            return (pod["rank"], "")

    res = sorted(for_pods(select(st, a.ranks), one, workers=24))
    rows = {}
    for rank, out in res:
        for rec in (out.split("|") if out else []):
            f = rec.split(",")
            if len(f) == 6:
                rows.setdefault(int(f[0]), []).append([float(x) for x in f[1:]])
    if not rows:
        print("no metrics yet")
        return
    print(f"{'epoch':>6} {'train':>9} {'val':>9} {'skill':>9} {'dcell':>8} {'rollH10':>9} {'ranks':>6}")
    for ep in sorted(rows):
        v = np.mean(rows[ep], axis=0) if (np := __import__("numpy")) else None
        print(f"{ep:>6} {v[0]:>9.4f} {v[1]:>9.4f} {v[2]:>+9.4f} {v[3]:>8.4f} {v[4]:>+9.4f} {len(rows[ep]):>6}")
    print("(mean across ranks; each pod trains its own shard and weights are averaged every epoch)")


def cmd_ckpt(a) -> None:
    """Put one or more repo-relative checkpoints on every pod, via S3 presigned GETs.

    `push_policies` already does this, but only for the checkpoints a COLLECTION mix names. A training
    relaunch needs an arbitrary `--init-from` file on every pod, and CODE_EXCLUDES deliberately drops
    checkpoints/ and *.pt from the code rsync, so the file is simply absent and every rank dies on
    `FileNotFoundError: checkpoint not found` (AGENTS 16).

    Why S3 and not rsync: 8 MB x 20 pods is 160 MB over the Mac's ~0.7 MB/s uplink, about four minutes of
    serialised upload during which the launch looks like it is progressing. Over ~10 MB total this belongs in
    S3 with presigned GETs -- AGENTS 20, which exists because `push_policies` violated AGENTS 15 the night 15
    was written. Presigned URLs need no credentials on the pod.

    Verifies the file landed at the path the trainer will read, with the right size, rather than trusting the
    curl exit status (AGENTS 16: a success log line is not evidence).
    """
    import boto3
    st = load_state(a.experiment)
    rel = [r.strip() for r in a.path.split(",") if r.strip()]
    for r in rel:
        if not (REPO / r).exists():
            log(f"not found on the Mac: {r}")
            return
    sizes = {r: (REPO / r).stat().st_size for r in rel}
    total = sum(sizes.values()) / 1e6
    s3 = boto3.Session(profile_name=S3_PROFILE, region_name=S3_REGION).client("s3")
    keys = {r: f"fleet/ckpt/{r}" for r in rel}
    for r, k in keys.items():
        up = True
        try:
            up = s3.head_object(Bucket=S3_BUCKET, Key=k)["ContentLength"] != sizes[r]
        except Exception:                                                       # noqa: BLE001
            pass
        if up:
            log(f"upload {r} ({sizes[r]/1e6:.1f} MB) -> s3://{S3_BUCKET}/{k}")
            s3.upload_file(str(REPO / r), S3_BUCKET, k)
        else:
            log(f"already in S3 at the right size: {k}")

    urls = {r: s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": k}, ExpiresIn=6 * 3600)
            for r, k in keys.items()}
    dirs = sorted({str(Path(r).parent) for r in rel})

    def one(pod):
        lines = ["set -e"] + [f"mkdir -p {R_REPO}/{d}" for d in dirs]
        for r in rel:
            lines.append(f"curl -sfL --retry 5 -o {R_REPO}/{r} {shlex.quote(urls[r])} &")
        lines.append("wait")
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as f:
            f.write("\n".join(lines) + "\n")
            tf = f.name
        try:
            rsync(pod, [tf], "/workspace/ckpt_pull.sh")
        finally:
            Path(tf).unlink(missing_ok=True)
        run(pod, "bash /workspace/ckpt_pull.sh && rm /workspace/ckpt_pull.sh")
        # VERIFY, do not assume: stat the destination the trainer will actually open.
        got = run(pod, "; ".join(f"stat -c %s {R_REPO}/{r} 2>/dev/null || echo 0" for r in rel),
                  capture=True).split()
        bad = [r for r, g in zip(rel, got) if int(g or 0) != sizes[r]]
        return (pod["rank"], bad)

    res = sorted(for_pods(select(st, a.ranks), one, workers=24))
    bad = {rank: b for rank, b in res if b}
    log(f"{len(res)} pods x {len(rel)} file(s), {total:.1f} MB each")
    if bad:
        log(f"WRONG SIZE OR MISSING on ranks {sorted(bad)}: {bad}")
    else:
        log(f"verified by size at {', '.join(R_REPO + '/' + r for r in rel)} on all {len(res)} pods")


def cmd_ppo(a) -> None:
    """Fleet-wide PPO curve, straight from each rank's train log. The answer to "is it improving".

    `fleet metrics` reads the world-model trainer's metrics.jsonl and says "no metrics yet" for a PPO run,
    so every "status?" during a Gohma fleet became a hand-written ssh fan-out -- which is the thing AGENTS
    17 and 21 exist to stop. One rank's succ column swings 0%-18% between consecutive updates on ~20
    episodes, so the per-rank rows are printed for liveness and the FLEET MEAN at a shared update is the
    only number to read.
    """
    st = load_state(a.experiment)
    PROBE = (
        "import glob,sys;"
        "fs=sorted(glob.glob('/workspace/zelda-RL/logs/*.log'),key=lambda f:__import__('os').path.getmtime(f));"
        "out=[]\n"
        "for l in (open(fs[-1]) if fs else []):\n"
        "    p=l.split()\n"
        "    if len(p)>8 and p[1]=='INFO' and p[2].isdigit() and p[5].endswith('%'):\n"
        "        k=l.split('klA')[1].split()[0] if 'klA' in l else '-'\n"
        "        out.append(','.join([p[2],p[3].replace(',',''),p[5].rstrip('%'),p[7].rstrip('%'),k]))\n"
        "print('|'.join(out[-int(sys.argv[1]):]))"
    )

    def one(pod):
        try:
            return (pod["rank"], run(pod, f"python3 -c {shlex.quote(PROBE)} {a.last} 2>/dev/null || echo ''",
                                     capture=True).strip())
        except Exception:                                                               # noqa: BLE001
            return (pod["rank"], "")

    res = sorted(for_pods(select(st, a.ranks), one, workers=24))
    per_upd: dict[int, list[tuple[float, float, float]]] = {}
    print(f"{'rank':>4} {'upd':>5} {'steps':>10} {'succ':>6} {'death':>6} {'klA':>7}")
    quiet = []
    for rank, out in res:
        recs = [r.split(",") for r in out.split("|") if r.count(",") == 4]
        if not recs:
            quiet.append(rank)
            continue
        for u, s, sc, de, kl in recs:
            try:
                per_upd.setdefault(int(u), []).append((float(sc), float(de),
                                                       float(kl) if kl not in ("-", "") else float("nan")))
            except ValueError:
                pass
        u, s, sc, de, kl = recs[-1]
        print(f"{rank:>4} {u:>5} {int(s):>10,} {sc + '%':>6} {de + '%':>6} {kl:>7}")
    if quiet:
        print(f"  no update lines yet on ranks {quiet} -- still in the first rollout, or not training")
    if per_upd:
        print("\nFLEET MEAN (succ and death averaged over the ranks that reached that update)")
        for u in sorted(per_upd)[-a.last:]:
            v = per_upd[u]
            sc = sum(x[0] for x in v) / len(v)
            de = sum(x[1] for x in v) / len(v)
            kl = [x[2] for x in v if x[2] == x[2]]
            klt = f"  klA {sum(kl)/len(kl):.4f}" if kl else ""
            print(f"  upd {u:>4}  succ {sc:5.1f}%  death {de:5.1f}%{klt}   [{len(v)} ranks]")


def cmd_ssh(a) -> None:
    st = load_state(a.experiment)
    pod = select(st, str(a.rank))[0]
    parts = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    cmd = parts[0] if len(parts) == 1 else " ".join(shlex.quote(x) for x in parts)
    sys.exit(subprocess.call(ssh_base(pod) + ([cmd] if cmd else [])))


def cmd_list(a) -> None:
    from fleet.runpod_api import RunPod
    total = 0.0
    for p in RunPod().list():
        if not str(p.get("name", "")).startswith("zrl-"):
            continue
        total += p.get("costPerHr") or 0
        print(f"{p.get('name'):28s} {p.get('id'):16s} {p.get('desiredStatus'):10s} vcpu={p.get('vcpuCount')} "
              f"mem={p.get('memoryInGb')}GB ${p.get('costPerHr')}/hr {p.get('publicIp')}:{(p.get('portMappings') or {}).get('22')}")
    print(f"total ${total:.2f}/hr")


def cmd_adopt(a) -> None:
    """Rebuild fleet state from the RunPod API: keep one SSH-ready pod per rank, terminate stuck/duplicate pods."""
    from fleet.runpod_api import RunPod
    rp = RunPod()
    st = load_state(a.experiment)
    pods = [p for p in rp.list() if str(p.get("name", "")).startswith(f"zrl-{a.experiment}-r")]
    by_rank: dict[int, list] = {}
    for p in pods:
        by_rank.setdefault(int(p["name"].rsplit("-r", 1)[1]), []).append(p)
    keep, kill = [], []
    for rank, ps in sorted(by_rank.items()):
        ready = [p for p in ps if p.get("publicIp") and (p.get("portMappings") or {}).get("22")]
        if ready:
            keep.append((rank, ready[0]))
            kill += [p for p in ps if p is not ready[0]]
        else:
            kill += ps
    st["pods"] = [{"rank": r, "name": p["name"], "id": p["id"], "ip": p["publicIp"], "port": int(p["portMappings"]["22"]),
                   "cost_per_hr": p.get("costPerHr"), "vcpu": p.get("vcpuCount"), "mem_gb": p.get("memoryInGb")}
                  for r, p in keep]
    log(f"adopt: keeping {len(keep)} SSH-ready pods (ranks {[r for r, _ in keep]}); {len(kill)} stuck/duplicate: "
        f"{[(p['name'], p['id']) for p in kill]}")
    if a.confirm:
        for p in kill:
            rp.delete(p["id"])
            log(f"  terminated {p['name']} ({p['id']})")
        save_state(st)
        log(f"fleet state saved: {len(st['pods'])} pods, ${sum(p['cost_per_hr'] or 0 for p in st['pods']):.2f}/hr")


def cmd_reconcile(a) -> None:
    """Terminate account pods named zrl-<experiment>-* that are not in the local fleet state (orphans from retries)."""
    from fleet.runpod_api import RunPod
    st = load_state(a.experiment)
    known = {p["id"] for p in st["pods"]}
    rp = RunPod()
    orphans = [p for p in rp.list() if str(p.get("name", "")).startswith(f"zrl-{a.experiment}-") and p["id"] not in known]
    log(f"{len(orphans)} orphan pod(s): {[(p['name'], p['id'], p.get('desiredStatus')) for p in orphans]}")
    if orphans and a.confirm:
        for p in orphans:
            rp.delete(p["id"])
            log(f"  terminated orphan {p['name']} ({p['id']})")


def cmd_kill(a) -> None:
    if getattr(a, "check_s3", None):
        st0 = load_state(a.experiment)
        d, srv, gaps = s3_mirror_gaps(st0, a.check_s3_out, a.check_s3)
        if gaps:
            raise SystemExit(f"REFUSING TO KILL: {len(gaps)} pods have episodes not in S3 "
                             f"(disk {d:,} vs S3 {srv:,}). Back them up first, or drop --check-s3.\n"
                             f"  worst: " + ", ".join(f"rank{r} missing {nd-ns}" for r, nd, ns in gaps[:5]))
        log(f"S3 mirror verified: {srv:,} episodes, no gaps -- safe to terminate")

    from fleet.runpod_api import RunPod
    st = load_state(a.experiment)
    pods = select(st, a.ranks)
    log(f"terminate {[p['name'] for p in pods]}")
    if not a.confirm:
        log("dry run (add --confirm). Tip: `python -m fleet sync` first to keep checkpoints.")
        return
    rp = RunPod()
    for p in pods:
        rp.delete(p["id"])
        log(f"  terminated {p['name']} ({p['id']})")
    killed = {p["id"] for p in pods}
    st["pods"] = [p for p in st["pods"] if p["id"] not in killed]
    save_state(st)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m fleet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, exp=True, ranks=True):
        p = sub.add_parser(name)
        if exp:
            p.add_argument("--experiment", required=True)
        if ranks:
            p.add_argument("--ranks", default=None, help="comma-separated ranks (default: all)")
        p.set_defaults(fn=fn)
        return p

    p = add("create", cmd_create, ranks=False)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--vcpu", type=int, default=16)
    p.add_argument("--flavors", default="cpu5c,cpu3c")
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--image", default="runpod/base:1.2.0-ubuntu2404")
    # GPU pods. The world model is 2M params and needs no VRAM to speak of; what a GPU buys is throughput
    # (measured 193k frames/s at dim 128 against 9.4k on the Mac) and, once the dataset is packed to ~590
    # B/frame, the ability to hold all 91.5M transitions resident so training does no host-device transfer.
    # gpu_type_ids is a PRIORITY LIST: pass several so a sold-out A100 falls through to an H100 instead of
    # failing. RunPod returns "no instances available" and "balance too low" in the SAME 500, so if creation
    # fails at every size, probe a tiny one -- the message drops the capacity clause when it is really money.
    p.add_argument("--gpu", default=None,
                   help="comma-separated gpuTypeIds in priority order, e.g. "
                        "'NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3'")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--gpu-image", default=GPU_IMAGE, help="CUDA+torch image; building torch-cuda on the pod is a 10-min tax")
    p.add_argument("--container-disk", type=int, default=40, help="GB of container disk")
    p.add_argument("--volume-gb", type=int, default=20,
                   help="GB on /workspace. A collector needs ~1; a TRAINER holding the whole dataset needs "
                        "the dataset size plus room to unpack -- v5 is ~25 GB, so 20 silently fills up")
    p.add_argument("--ssh-timeout", type=float, default=420, help="give up on (and terminate) a pod without SSH after this")
    p.add_argument("--confirm", action="store_true")
    p = add("bootstrap", cmd_bootstrap)
    p.add_argument("--via-seeds", default=None, help="comma-separated seed ranks that rsync assets to the others")
    p.add_argument("--assets-s3", action="store_true", help="pods download Dolphin + NAND + savestates from S3 "
                   "(presigned URLs; run `fleet assets --upload` once first) instead of pushes from the Mac")
    p.add_argument("--workers", type=int, default=8)
    p = add("assets", cmd_assets)
    p.add_argument("--upload", action="store_true", help="first upload the Mac's savestates + NAND + Dolphin build to S3")
    p.add_argument("--states-only", action="store_true", help="pods re-download only the savestates")
    p.add_argument("--states-prefix", default=None,
                   help=f"S3 prefix for the start states; use {S3_STATES_DEKU!r} for a Deku Baba run "
                        f"(the default is the gohma task's states, which have no live Babas)")
    p = add("verify-s3", cmd_verify_s3)
    p.add_argument("--out", default="datasets/transitions/v5", help="collector --out path on the pod")
    p.add_argument("--s3-prefix", default="s3://zelda-rl/zelda/transitions/v5")
    p = add("preflight", cmd_preflight)
    p.add_argument("--write-bad", default=None, help="write not-ready ranks here, for a re-bootstrap")
    p = add("audit", cmd_audit)
    p.add_argument("--out", default="datasets/transitions/v5", help="collector --out path, to count episodes")
    p.add_argument("--write-idle", default=None, help="write idle+broken ranks to this file, for a re-bootstrap")
    p = add("metrics", cmd_metrics)
    p.add_argument("--dir", default="runs/wm-fleet", help="--ckpt-dir used at launch")
    p.add_argument("--last", type=int, default=12, help="show the last N epochs")
    p = add("ppo", cmd_ppo)
    p.add_argument("--last", type=int, default=8, help="show the last N updates")
    p = add("ckpt", cmd_ckpt)
    p.add_argument("--path", required=True,
                   help="comma-separated repo-relative checkpoint path(s) to place on every pod, e.g. "
                        "checkpoints/kept/gohma-kl-v15-r4_latest.pt")
    p = add("collected", cmd_collected)
    p.add_argument("--out", default="datasets/transitions/v4-collect", help="the collector's --out path")
    p.add_argument("--verbose", action="store_true", help="per-pod breakdown")
    p = add("verify", cmd_verify)
    p.add_argument("--instances", type=int, default=0)
    p = add("launch", cmd_launch)
    p.add_argument("--num-envs", type=int, default=0)
    p.add_argument("--train-args", default="")
    p.add_argument("--run-tag", default="", help="suffix for run names + wandb group (e.g. rupees7)")
    p.add_argument("--script", default="train_ppo.py", help="trainer under scripts/ (train_ppo.py | train_combat.py | collect_transitions.py)")
    p.add_argument("--skip-preflight", action="store_true", help="launch even on pods that fail readiness")
    p.add_argument("--collect", action="store_true",
                   help="the script is a COLLECTOR, not a trainer: omit --run-name/--num-envs/--fleet-* "
                        "(collect_transitions.py rejects them) and pass only --train-args")
    p.add_argument("--wandb-project", default="zelda-rl-goal-ppo")
    p = add("status", cmd_status)
    p.add_argument("--lines", type=int, default=2)
    p = add("sync", cmd_sync)
    p.add_argument("--datasets", default=None,
                   help="pull a dataset dir (e.g. datasets/transitions/v4-scale/<ts>) from every "
                        "pod into datasets/transitions/<name>/rank<NN>/ instead of runs/+checkpoints")
    p = add("ssh", cmd_ssh, ranks=False)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    add("list", cmd_list, exp=False, ranks=False)
    p = add("adopt", cmd_adopt, ranks=False)
    p.add_argument("--confirm", action="store_true")
    p = add("reconcile", cmd_reconcile, ranks=False)
    p.add_argument("--confirm", action="store_true")
    p = add("kill", cmd_kill)
    p.add_argument("--confirm", action="store_true")
    # A collection fleet should not be terminatable while episodes exist only on pod disks: that is how a
    # 20M-transition dataset was lost. Pass --check-s3 and kill refuses until the mirror is complete.
    p.add_argument("--check-s3", default=None,
                   help="S3 prefix to verify against before terminating, e.g. "
                        "s3://zelda-rl/zelda/transitions/v5 -- refuses to kill while any pod has unmirrored data")
    p.add_argument("--check-s3-out", default="datasets/transitions/v5", help="collector --out path on the pod")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
