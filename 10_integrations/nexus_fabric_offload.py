# ---
# deploy: false
# cmd: ["python", "-m", "10_integrations.nexus_fabric_offload"]
# ---

# # Offloading heavy compute from a local scheduler to Modal
#
# This example shows how a local performance scheduler can offload
# non-realtime GPU workloads to Modal while keeping latency-sensitive
# work on the user's own hardware. The pattern is simple: the local
# process classifies each workload, keeps the realtime ones local,
# and ships the heavy batched ones to a Modal Function.
#
# The usage shape is deliberately small — a `worker` Function that
# takes a `(workload_name, payload)` pair and returns a result dict.
# This is the exact contract a classifier-driven mesh orchestrator
# can target without knowing anything else about Modal.
#
# Why this pattern:
#
# - Shader precompilation, texture decompression, per-game upscaler
#   fine-tunes — all tolerate 1–5s round-trip
# - The user's local GPU can stay dedicated to realtime rendering
# - The local orchestrator picks the right endpoint; Modal is one of
#   several possible backends (HuggingFace Spaces is another)

import time
from typing import Any

import modal

app = modal.App("nexus-fabric-worker")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.1.0")
)


# ## The worker
#
# One Function, GPU-backed, with a short 10-minute timeout. The signature
# matches what the local Nexus Fabric orchestrator expects when it marks a
# job as `CLOUD_MESH`: `(workload_name, payload) -> result dict`.
#
# For this example we implement a BLAS-shape attention block in NumPy
# that runs on the Modal T4; swap in any heavier compute (shader compile,
# texture decode, ML upscaler training) once you've shaped the payload.


@app.function(gpu="t4", timeout=600, image=image)
def run(workload_name: str, payload: dict) -> dict:
    import numpy as np

    started = time.time()
    if workload_name == "shader_compile":
        # Stand-in: a batched attention block sized by `payload['dim']`.
        # Deterministic seed so the client can verify output cheaply.
        rng = np.random.default_rng(payload.get("seed", 42))
        dim = int(payload.get("dim", 512))
        batches = int(payload.get("batches", 32))
        a = rng.standard_normal((batches, dim, dim), dtype=np.float32)
        b = rng.standard_normal((batches, dim, dim), dtype=np.float32)
        attn = np.matmul(a, np.transpose(b, (0, 2, 1)))
        attn -= attn.max(axis=-1, keepdims=True)
        np.exp(attn, out=attn)
        attn /= attn.sum(axis=-1, keepdims=True)
        out = np.matmul(attn, b)
        checksum = float(out.sum())
        return {
            "status": "ok",
            "workload": workload_name,
            "dim": dim,
            "batches": batches,
            "checksum": checksum,
            "elapsed_s": round(time.time() - started, 3),
        }
    return {"status": "unknown_workload", "workload": workload_name}


# ## Local orchestrator side
#
# This is what the local Nexus Fabric classifier does: hand off the
# workload to the Modal Function and get a result back. The local caller
# never touches `modal.Function` directly — it goes through a dispatcher
# that knows about multiple backends. Here we call the Function in-place
# so the example is self-contained.


@app.local_entrypoint()
def main() -> None:
    payload = {"dim": 512, "batches": 32, "seed": 42}
    print(f"[local] submitting shader_compile payload={payload}")
    start = time.time()
    result: Any = run.remote("shader_compile", payload)
    print(f"[local] response in {time.time() - start:.2f}s: {result}")


# ## Running it
#
# ```shell
# pip install modal
# modal setup                       # one-time auth
# modal run 10_integrations/nexus_fabric_offload.py
# ```
#
# To wire this into a local orchestrator (the full example lives at
# [github.com/61465/nexus-fabric](https://github.com/61465/nexus-fabric)),
# set `NEXUS_MODAL_APP=nexus-fabric-worker` in the orchestrator's `.env`
# and route workloads whose classifier returns `CLOUD_MESH` through
# `modal.Function.from_name("nexus-fabric-worker", "run")`.
