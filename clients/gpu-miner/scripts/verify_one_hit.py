"""Sanity: run one kernel launch with relaxed target, take a hit, feed it to
the Python CPU verifier. If verifier disagrees, kernel is buggy."""

import secrets
import time

from equiumminer.gpu import GpuWorker, Job, build_input113, compute_blake2b_h_base
from equiumminer.verify import _python_verify


def main():
    challenge = secrets.token_bytes(32)
    miner = secrets.token_bytes(32)
    height = 12345
    nonce_template = secrets.token_bytes(24) + b"\x00" * 8
    input113 = build_input113(challenge, miner, height, nonce_template)
    h_base = compute_blake2b_h_base()
    target = bytes([0xFF] * 32)  # relaxed: every Equihash solution passes

    job = Job(
        block_height=height,
        challenge=challenge,
        target=target,
        miner_pubkey=miner,
        h_base=h_base,
        input113=input113,
    )

    worker = GpuWorker(0, batch_size=4, local_size=256)
    print(f"on {worker.device.name}")

    t0 = time.time()
    hits = worker.launch(job, secrets.randbits(64))
    print(f"got {len(hits)} hits in {(time.time()-t0)*1000:.0f}ms")
    if not hits:
        print("no hits to verify — bug in kernel (should always produce hits at FF target)")
        return

    ok_count = 0
    fail_count = 0
    for hit in hits[:5]:
        vr = _python_verify(challenge, miner, height, hit.nonce, hit.soln_indices, target)
        if vr.ok:
            ok_count += 1
            print(f"  OK nonce {hit.nonce[:6].hex()}... hash {vr.solution_hash_hex[:16]}")
        else:
            fail_count += 1
            print(f"  FAIL nonce {hit.nonce[:6].hex()}... REJECTED: {vr.reason}")
    print(f"\n{ok_count}/{ok_count+fail_count} hits validated by Python verifier")


if __name__ == "__main__":
    main()
