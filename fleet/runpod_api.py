"""RunPod REST v1 (https://rest.runpod.io/v1) — CPU pods. Field names checked against RunPod's OpenAPI spec."""
from __future__ import annotations

import time

import requests

from fleet.secrets import require

API = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/base:1.2.0-ubuntu2404"
# CUDA + torch preinstalled; building torch-cuda into the venv on the pod is a 10-minute tax.
GPU_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


class RunPod:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {require('RUNPOD')}", "Content-Type": "application/json"})

    def _req(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{API}{path}", timeout=60, **kw)
        if not r.ok:
            raise RuntimeError(f"RunPod {method} {path} -> {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else {}

    def create_cpu_pod(self, name: str, public_key: str, *, vcpu: int = 16, flavors=("cpu5c", "cpu3c"),
                       image: str = DEFAULT_IMAGE, container_disk_gb: int = 40, volume_gb: int = 20,
                       cloud: str = "SECURE", data_centers: list[str] | None = None) -> dict:
        body = {
            "name": name, "computeType": "CPU", "cpuFlavorIds": list(flavors), "vcpuCount": vcpu,
            "imageName": image, "containerDiskInGb": container_disk_gb, "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": public_key},
            "cloudType": cloud,
        }
        if data_centers:
            body["dataCenterIds"] = data_centers
        return self._req("POST", "/pods", json=body)

    def create_gpu_pod(self, name: str, public_key: str, *, gpu_type_ids: list[str], gpu_count: int = 1,
                       image: str = GPU_IMAGE, container_disk_gb: int = 60, volume_gb: int = 200,
                       cloud: str = "SECURE", data_centers: list[str] | None = None) -> dict:
        """A GPU pod. gpu_type_ids is a PRIORITY LIST -- RunPod takes the first with capacity, so pass
        several equivalent A100 spellings rather than failing on the one that happens to be sold out."""
        body = {
            "name": name, "computeType": "GPU", "gpuTypeIds": gpu_type_ids, "gpuCount": gpu_count,
            "imageName": image, "containerDiskInGb": container_disk_gb, "volumeInGb": volume_gb,
            "volumeMountPath": "/workspace", "ports": ["22/tcp"], "env": {"PUBLIC_KEY": public_key},
            "cloudType": cloud,
        }
        if data_centers:
            body["dataCenterIds"] = data_centers
        return self._req("POST", "/pods", json=body)

    def get(self, pod_id: str) -> dict:
        return self._req("GET", f"/pods/{pod_id}")

    def list(self) -> list[dict]:
        out = self._req("GET", "/pods")
        return out if isinstance(out, list) else out.get("pods", out.get("data", []))

    def delete(self, pod_id: str) -> None:
        self._req("DELETE", f"/pods/{pod_id}")

    def wait_ssh(self, pod_id: str, timeout: float = 900, log=print) -> tuple[str, int, dict]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            pod = self.get(pod_id)
            ip, port = pod.get("publicIp"), (pod.get("portMappings") or {}).get("22")
            if pod.get("desiredStatus") == "RUNNING" and ip and port:
                return ip, int(port), pod
            log(f"  {pod_id}: status={pod.get('desiredStatus')} ip={ip or '-'} ssh_port={port or '-'} ({time.time() - t0:.0f}s)")
            time.sleep(15)
        raise TimeoutError(f"{pod_id} not SSH-ready after {timeout:.0f}s")
