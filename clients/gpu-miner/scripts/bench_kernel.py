"""Benchmark one kernel launch with synthetic inputs. Reports time/launch and
the implied per-WI throughput.

Run from clients/gpu-miner:
    python scripts/bench_kernel.py --batch 64 --runs 3
"""

from __future__ import annotations

import argparse
import secrets
import time

import numpy as np
import pyopencl as cl

from equiumminer import constants as C
from equiumminer.gpu import (
    GpuWorker,
    Job,
    build_input113,
    compute_blake2b_h_base,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print(f"device={args.device} batch={args.batch} local={args.local}")
    worker = GpuWorker(args.device, args.batch, args.local)
    print(f"on {worker.device.name}, {worker.device.global_mem_size/(1<<30):.1f} GB")

    # Synthetic job — challenge = random, height = 0, miner = random pubkey.
    challenge = secrets.token_bytes(32)
    miner = secrets.token_bytes(32)
    height = 0
    nonce_template = secrets.token_bytes(24) + b"\x00" * 8
    input113 = build_input113(challenge, miner, height, nonce_template)
    h_base = compute_blake2b_h_base()
    # Random target (relaxed) - we don't care if anything is below it
    target = bytes([0xFF] * 32)

    job = Job(
        block_height=height,
        challenge=challenge,
        target=target,
        miner_pubkey=miner,
        h_base=h_base,
        input113=input113,
    )

    # Warm-up
    print("warming up...")
    _ = worker.launch(job, secrets.randbits(64))

    print(f"running {args.runs} launches of {args.batch} WI each")
    for r in range(args.runs):
        t0 = time.time()
        hits = worker.launch(job, secrets.randbits(64))
        dt = time.time() - t0
        per_nonce_ms = dt * 1000.0 / args.batch
        throughput = args.batch / dt
        print(
            f"  run {r+1}: {dt*1000:.0f}ms total | {per_nonce_ms:.2f}ms/nonce | "
            f"{throughput:.1f} nonces/sec | {len(hits)} hits"
        )


if __name__ == "__main__":
    main()
