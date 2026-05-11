"""PyOpenCL device discovery + per-device worker + farm orchestrator.

Each device runs an independent worker thread. Workers receive a Job
(challenge, target, height, miner pubkey) and grind nonce batches until either
a hit is found (passed back through the results queue) or they're told to stop.

Workers share NO state with each other; nonce-base offsets are randomized per
launch, so two devices grinding the same job can't collide.
"""

from __future__ import annotations

import logging
import os
import queue
import secrets
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyopencl as cl

from . import constants as C

logger = logging.getLogger(__name__)

_KERNEL_PATH = Path(__file__).parent / "kernels" / "equihash_96_5.cl"

# Mirrors the kernel's WORKSPACE_PER_WI macro. Keep in sync.
WORKSPACE_PER_WI = (
    2 * C.N_INIT * 144     # buf_a + buf_b, ROW_BYTES = 144
    + 2 * 65536 * 4        # counts + starts
)
COMPRESSED_BYTES = 68
MAX_HITS_PER_BATCH = 16


@dataclass
class Job:
    """A mining job pinned to a specific round."""

    block_height: int
    challenge: bytes              # 32
    target: bytes                 # 32 (big-endian)
    miner_pubkey: bytes           # 32
    # h_base = BLAKE2b initial state (IV XOR param) — computed once on host.
    # Same across all jobs since param is constant. Provided here for kernel.
    h_base: np.ndarray            # uint64 [8]
    # The 113-byte input passed to the kernel: persn(9) + I-block(72) + nonce-template(32).
    # The kernel overwrites bytes [105..113] with the per-WI nonce counter.
    input113: bytes


@dataclass
class Hit:
    """A candidate from the GPU. Caller MUST CPU-verify before submitting."""

    device_idx: int
    nonce: bytes               # 32
    soln_indices: bytes        # 68 (compressed)
    block_height: int


def compute_blake2b_h_base() -> np.ndarray:
    """h_base = BLAKE2b IV XOR param.

    Equihash uses BLAKE2b with a 64-byte parameter block; the relevant fields
    differ from a stock keyed BLAKE2b. Returns 8 × uint64.
    """
    # 64-byte param block (little-endian as words):
    #   byte 0  = digest_length = 60
    #   byte 1  = key_length = 0
    #   byte 2  = fanout = 1
    #   byte 3  = depth = 1
    #   bytes 4..8 = leaf_length = 0
    #   bytes 8..16 = node_offset = 0
    #   byte 16 = node_depth = 0
    #   byte 17 = inner_length = 0
    #   bytes 18..32 = reserved = 0
    #   bytes 32..48 = salt = 0
    #   bytes 48..64 = personal = "ZcashPoW" + n_le32 + k_le32
    param = bytearray(64)
    param[0] = C.BLAKE2B_OUTPUT_BYTES   # 60
    param[1] = 0                        # key_length
    param[2] = 1                        # fanout
    param[3] = 1                        # depth
    # leaf_length, node_offset, node_depth, inner_length, reserved all zero
    # personal at [48..64]
    param[48:64] = C.BLAKE2B_PERSONAL

    iv = np.array(
        [
            0x6A09E667F3BCC908,
            0xBB67AE8584CAA73B,
            0x3C6EF372FE94F82B,
            0xA54FF53A5F1D36F1,
            0x510E527FADE682D1,
            0x9B05688C2B3E6C1F,
            0x1F83D9ABFB41BD6B,
            0x5BE0CD19137E2179,
        ],
        dtype=np.uint64,
    )
    param_words = np.frombuffer(bytes(param), dtype="<u8").astype(np.uint64)
    return iv ^ param_words


def build_input113(challenge: bytes, miner_pubkey: bytes, block_height: int, nonce_template: bytes) -> bytes:
    """113-byte buffer passed to the kernel.

    Layout matches what the kernel expects:
      [0..9)   persn = "Equium-v1"
      [9..41)  current_challenge
      [41..73) miner_pubkey
      [73..81) block_height LE
      [81..113) nonce template — kernel overwrites bytes [105..113] (low 8) per WI.
    """
    if len(challenge) != 32 or len(miner_pubkey) != 32 or len(nonce_template) != 32:
        raise ValueError("expected 32-byte challenge/miner_pubkey/nonce_template")
    buf = bytearray(113)
    buf[0:9] = C.PERSONALIZATION
    buf[9:41] = challenge
    buf[41:73] = miner_pubkey
    buf[73:81] = block_height.to_bytes(8, "little")
    buf[81:113] = nonce_template
    return bytes(buf)


def list_devices() -> List[cl.Device]:
    """Flat list of all OpenCL devices across all platforms."""
    devs: List[cl.Device] = []
    for plat in cl.get_platforms():
        for d in plat.get_devices():
            devs.append(d)
    return devs


def select_devices(spec) -> List[int]:
    """Return flat indices to use given a spec like 'all' or [0, 2] or '0,1'."""
    all_devs = list_devices()
    n = len(all_devs)
    if isinstance(spec, str) and spec.lower() == "all":
        return list(range(n))
    if isinstance(spec, str):
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    raise ValueError(f"unrecognized device spec: {spec!r}")


class GpuWorker:
    """One OpenCL queue + program + buffers, sized for one device."""

    def __init__(self, device_idx: int, batch_size: int, local_size: int):
        all_devs = list_devices()
        if device_idx >= len(all_devs):
            raise IndexError(f"device {device_idx} (have {len(all_devs)})")
        self.device_idx = device_idx
        self.device = all_devs[device_idx]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.batch_size = batch_size
        self.local_size = local_size
        self._compile()
        self._alloc_buffers()

    def _compile(self):
        src = _KERNEL_PATH.read_text()
        self.program = cl.Program(self.ctx, src).build()
        self.kernel = self.program.equihash_96_5

    def _alloc_buffers(self):
        mf = cl.mem_flags
        # Workspace: batch_size × WORKSPACE_PER_WI. Largest single allocation.
        ws_size = int(self.batch_size) * int(WORKSPACE_PER_WI)
        logger.info(
            "device %d (%s): allocating %.1f MB workspace (%d WI × %.1f MB)",
            self.device_idx,
            self.device.name,
            ws_size / (1 << 20),
            self.batch_size,
            WORKSPACE_PER_WI / (1 << 20),
        )
        self.buf_workspace = cl.Buffer(self.ctx, mf.READ_WRITE, size=ws_size)
        self.buf_h_base = cl.Buffer(self.ctx, mf.READ_ONLY, size=8 * 8)
        self.buf_input113 = cl.Buffer(self.ctx, mf.READ_ONLY, size=113)
        self.buf_hit_nonces = cl.Buffer(self.ctx, mf.READ_WRITE, size=MAX_HITS_PER_BATCH * 32)
        self.buf_hit_solutions = cl.Buffer(
            self.ctx, mf.READ_WRITE, size=MAX_HITS_PER_BATCH * COMPRESSED_BYTES
        )
        self.buf_hit_count = cl.Buffer(self.ctx, mf.READ_WRITE, size=4)

    def launch(self, job: Job, nonce_low_offset: int) -> List[Hit]:
        """Run one kernel launch over `batch_size` work-items. Returns hits found."""
        # Reset hit counter.
        zero32 = np.zeros(1, dtype=np.uint32)
        cl.enqueue_copy(self.queue, self.buf_hit_count, zero32)
        cl.enqueue_copy(self.queue, self.buf_h_base, job.h_base.astype(np.uint64))
        cl.enqueue_copy(self.queue, self.buf_input113, np.frombuffer(job.input113, dtype=np.uint8))

        self.kernel.set_arg(0, self.buf_h_base)
        self.kernel.set_arg(1, self.buf_input113)
        self.kernel.set_arg(2, np.uint64(nonce_low_offset))
        self.kernel.set_arg(3, self.buf_workspace)
        self.kernel.set_arg(4, self.buf_hit_nonces)
        self.kernel.set_arg(5, self.buf_hit_solutions)
        self.kernel.set_arg(6, self.buf_hit_count)
        self.kernel.set_arg(7, np.uint32(MAX_HITS_PER_BATCH))

        global_size = (self.batch_size,)
        local_size = (min(self.local_size, self.batch_size),)
        cl.enqueue_nd_range_kernel(self.queue, self.kernel, global_size, local_size)

        # Read back hit count, then nonces + solutions if any.
        hit_count = np.zeros(1, dtype=np.uint32)
        cl.enqueue_copy(self.queue, hit_count, self.buf_hit_count)
        self.queue.finish()
        n_hits = int(min(hit_count[0], MAX_HITS_PER_BATCH))
        if n_hits == 0:
            return []
        nonces = np.zeros(n_hits * 32, dtype=np.uint8)
        solutions = np.zeros(n_hits * COMPRESSED_BYTES, dtype=np.uint8)
        cl.enqueue_copy(self.queue, nonces, self.buf_hit_nonces)
        cl.enqueue_copy(self.queue, solutions, self.buf_hit_solutions)
        self.queue.finish()
        hits = []
        for i in range(n_hits):
            n_bytes = bytes(nonces[i * 32 : (i + 1) * 32])
            s_bytes = bytes(solutions[i * COMPRESSED_BYTES : (i + 1) * COMPRESSED_BYTES])
            hits.append(Hit(self.device_idx, n_bytes, s_bytes, job.block_height))
        return hits


class GpuFarm:
    """Manages one worker thread per device. Workers pull from a shared `current
    job` slot (updated atomically), push hits onto a shared results queue."""

    def __init__(self, device_indices: List[int], batch_size: int, local_size: int):
        self._workers: List[GpuWorker] = []
        for idx in device_indices:
            try:
                self._workers.append(GpuWorker(idx, batch_size, local_size))
            except Exception as e:
                logger.exception("device %d init failed: %s", idx, e)
        if not self._workers:
            raise RuntimeError("no usable OpenCL devices")

        self._lock = threading.Lock()
        self._job: Optional[Job] = None
        self._job_gen = 0
        self.results: "queue.Queue[Hit]" = queue.Queue()
        self.stats_attempts = [0] * len(self._workers)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    def set_job(self, job: Optional[Job]):
        with self._lock:
            self._job = job
            self._job_gen += 1

    def start(self):
        for i, w in enumerate(self._workers):
            t = threading.Thread(target=self._worker_loop, args=(i, w), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    def _worker_loop(self, slot: int, worker: GpuWorker):
        while not self._stop.is_set():
            with self._lock:
                job = self._job
                gen = self._job_gen
            if job is None:
                time.sleep(0.05)
                continue
            # Fresh random nonce low offset per launch so two workers never overlap.
            nonce_low = secrets.randbits(64)
            try:
                hits = worker.launch(job, nonce_low)
            except Exception as e:
                logger.exception("device %d launch failed: %s", worker.device_idx, e)
                time.sleep(0.5)
                continue
            self.stats_attempts[slot] += worker.batch_size
            # Drop hits if the job changed under us — stale solutions can never
            # land on-chain anyway.
            with self._lock:
                stale = (gen != self._job_gen)
            if stale:
                continue
            for h in hits:
                self.results.put(h)

    @property
    def device_names(self) -> List[str]:
        return [w.device.name for w in self._workers]
