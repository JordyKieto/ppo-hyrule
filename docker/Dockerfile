# Dolphin 2603a (DolphinQt + nogui) built from source + the zelda-RL Python stack, for linux/amd64.
# An alternative to building Dolphin on every pod (scripts/runpod.sh --build-dolphin).
#
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t zelda-rl:2603a .
#
# The image holds NO game assets, savestates or secrets. Those are rsynced to /workspace at run
# time (scripts/runpod.sh --assets), so the image is safe to push to a private registry. Pushing
# it and pointing a RunPod template at it is left to the user.
#
# Local smoke run (assets mounted, NOTE --shm-size: Docker's default 64 MB /dev/shm is smaller
# than ONE Dolphin Wii arena (~97 MB) and Dolphin dies with SIGBUS; verified with a fake arena):
#   docker run --rm -it --platform linux/amd64 --shm-size=4g \
#     -v $PWD:/workspace/zelda-RL zelda-rl:2603a \
#     bash -lc 'cd /workspace/zelda-RL && python scripts/verify_pod_dolphin.py'
# (x86_64 emulation on an Apple Silicon Mac is far too slow for real numbers; build with
#  --platform linux/arm64 to smoke-test locally, since Dolphin has an ARM64 JIT on Linux too.)

ARG UBUNTU=ubuntu:24.04

# ---------------------------------------------------------------------------------------------
FROM ${UBUNTU} AS build
ENV DEBIAN_FRONTEND=noninteractive
COPY scripts/pod_build_dolphin.sh /tmp/pod_build_dolphin.sh
# DOLPHIN_ROOT decides the install prefix, which Dolphin bakes in to find its Sys/ data dir,
# so the runtime stage must copy it to the identical path.
RUN DOLPHIN_ROOT=/opt/dolphin bash /tmp/pod_build_dolphin.sh \
 && rm -rf /opt/dolphin/src /opt/dolphin/build /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
FROM ${UBUNTU} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# Runtime libraries of the build above (Qt6 widgets/svg + offscreen QPA plugin, GL/EGL, X11,
# udev/evdev, ALSA/Pulse), xvfb as fallback display, tooling used by scripts/runpod.sh, sshd for RunPod.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libqt6widgets6t64 libqt6gui6t64 libqt6svg6 qt6-qpa-plugins \
      libgl1 libegl1 libopengl0 libx11-6 libxrandr2 libxi6 \
      libudev1 libevdev2 libsystemd0 libasound2t64 libpulse0 \
      xvfb xauth \
      python3 python3-venv python3-pip ca-certificates \
      tmux rsync openssh-server procps psmisc less \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/dolphin/install /opt/dolphin/install
COPY scripts/pod_build_dolphin.sh /opt/dolphin/pod_build_dolphin.sh
# Fails the image build if any shared library of the built binaries is missing at runtime.
RUN DOLPHIN_ROOT=/opt/dolphin SKIP_APT=1 bash /opt/dolphin/pod_build_dolphin.sh --check

# Python stack in a venv (Ubuntu 24.04's system pip refuses global installs, PEP 668).
# CPU torch wheels: observations are RAM features and the policy is a small MLP. For a GPU pod,
# build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 (or similar).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir --index-url ${TORCH_INDEX} torch \
 && /opt/venv/bin/pip install --no-cache-dir numpy lz4 wandb pillow matplotlib

ENV PATH=/opt/venv/bin:$PATH \
    DOLPHIN_BIN=/opt/dolphin/install/bin/dolphin-emu \
    QT_QPA_PLATFORM=offscreen \
    PYTHONUNBUFFERED=1

# RunPod convention: inject the SSH public key via the PUBLIC_KEY env var, start sshd, stay alive.
RUN mkdir -p /run/sshd \
 && printf '%s\n' '#!/bin/bash' \
      'if [[ -n "${PUBLIC_KEY:-}" ]]; then mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys; fi' \
      'ssh-keygen -A >/dev/null 2>&1; /usr/sbin/sshd' \
      'exec sleep infinity' > /start.sh && chmod +x /start.sh
WORKDIR /workspace
CMD ["/start.sh"]
