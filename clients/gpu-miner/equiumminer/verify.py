"""CPU re-verification of GPU candidates.

Two backends, in preference order:
  1. The Rust `equium-verify` binary — byte-for-byte identical to the on-chain
     verifier. Build with `cargo build -p equium-verify --release`. Preferred.
  2. A pure-Python fallback that re-implements Equihash verification via
     `hashlib.blake2b` + the canonical Wagner tree-walk. Slower but doesn't
     need a Rust toolchain. Good enough to catch obvious kernel bugs.

The Rust binary is selected automatically when present.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import constants as C

logger = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    ok: bool
    solution_hash_hex: str = ""
    reason: str = ""


def find_verifier_binary() -> Optional[str]:
    """Look up the equium-verify binary.

    Preference:
      1. EQM_VERIFY_BIN env var
      2. ../../target/release/equium-verify (workspace release build)
      3. PATH lookup
    """
    env = os.environ.get("EQM_VERIFY_BIN")
    if env and Path(env).exists():
        return env

    # clients/gpu-miner/equiumminer/verify.py → repo root is parents[3]
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "target" / "release" / "equium-verify",
        repo_root / "target" / "release" / "equium-verify.exe",
        repo_root / "target" / "debug" / "equium-verify",
        repo_root / "target" / "debug" / "equium-verify.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    p = shutil.which("equium-verify")
    return p


def cpu_verify(
    challenge: bytes,
    miner_pubkey: bytes,
    block_height: int,
    nonce: bytes,
    soln_indices: bytes,
    target: bytes,
    n: int = C.EQUIHASH_N,
    k: int = C.EQUIHASH_K,
    binary_path: Optional[str] = None,
) -> VerifyResult:
    """Run the Rust verifier on a candidate. Returns ok=True iff valid + under target."""
    if len(challenge) != 32 or len(miner_pubkey) != 32 or len(nonce) != 32 or len(target) != 32:
        raise ValueError("bad fixed-size input lengths")

    binary = binary_path or find_verifier_binary()
    if not binary:
        return VerifyResult(
            ok=False,
            reason="equium-verify binary not found; build with `cargo build -p equium-verify --release`",
        )

    # Reproduce the I-block exactly like challenge::build_input.
    input_block = bytearray(C.I_LEN)
    input_block[0:9] = C.PERSONALIZATION
    input_block[9:41] = challenge
    input_block[41:73] = miner_pubkey
    input_block[73:81] = block_height.to_bytes(8, "little")

    payload = bytearray()
    payload.extend(struct.pack("<I", n))
    payload.extend(struct.pack("<I", k))
    payload.extend(bytes(input_block))
    payload.extend(nonce)
    payload.extend(target)
    payload.extend(struct.pack("<I", len(soln_indices)))
    payload.extend(soln_indices)

    try:
        proc = subprocess.run(
            [binary],
            input=bytes(payload),
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(ok=False, reason="verifier timed out")
    except FileNotFoundError:
        return VerifyResult(ok=False, reason=f"verifier binary missing: {binary}")

    stdout = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    if proc.returncode == 0 and stdout.startswith("ok "):
        return VerifyResult(ok=True, solution_hash_hex=stdout.split(" ", 1)[1])
    if stdout.startswith("bad_equihash"):
        return VerifyResult(ok=False, reason="invalid Equihash (kernel bug)")
    if stdout.startswith("above_target"):
        parts = stdout.split(" ", 1)
        return VerifyResult(
            ok=False,
            solution_hash_hex=parts[1] if len(parts) > 1 else "",
            reason="above target",
        )
    return VerifyResult(ok=False, reason=f"verifier output: {stdout!r} (rc={proc.returncode})")


# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------


def _blake2b_personal_state() -> hashlib.blake2b:
    """Personalised BLAKE2b matching equihash-core (Zcash scheme)."""
    return hashlib.blake2b(
        digest_size=C.BLAKE2B_OUTPUT_BYTES,
        person=C.BLAKE2B_PERSONAL,
    )


def _gen_leaf(input113: bytes, batch_idx: int, sub: int) -> bytes:
    """Generate one 12-byte Equihash leaf."""
    s = _blake2b_personal_state()
    s.update(input113)
    s.update(batch_idx.to_bytes(4, "little"))
    full = s.digest()
    off = sub * C.LEAF_BYTES
    return full[off : off + C.LEAF_BYTES]


def _decompress_indices(soln: bytes) -> List[int]:
    """Unpack 32 × 17-bit indices, MSB-first within byte stream."""
    bits_per = C.CBITS + 1
    total_bits = bits_per * C.SOLUTION_INDICES_COUNT
    if len(soln) * 8 < total_bits:
        raise ValueError("solution bytes too short")
    out: List[int] = []
    for i in range(C.SOLUTION_INDICES_COUNT):
        val = 0
        for b in range(bits_per):
            pos = i * bits_per + b
            byte_off = pos >> 3
            shift = 7 - (pos & 7)
            bit = (soln[byte_off] >> shift) & 1
            val = (val << 1) | bit
        out.append(val)
    return out


def _python_verify(
    challenge: bytes,
    miner_pubkey: bytes,
    block_height: int,
    nonce: bytes,
    soln_indices: bytes,
    target: bytes,
) -> VerifyResult:
    """Reference-quality Python verifier. Slow (~tens of ms per call) but correct."""
    # I-block (81) + nonce (32) = 113-byte BLAKE2b input.
    input113 = bytearray(113)
    input113[0:9] = C.PERSONALIZATION
    input113[9:41] = challenge
    input113[41:73] = miner_pubkey
    input113[73:81] = block_height.to_bytes(8, "little")
    input113[81:113] = nonce
    input113 = bytes(input113)

    try:
        indices = _decompress_indices(soln_indices)
    except Exception as e:
        return VerifyResult(ok=False, reason=f"decompress: {e}")

    # Indices must all be in range, all distinct.
    if any(i >= C.N_INIT for i in indices):
        return VerifyResult(ok=False, reason="index out of range")
    if len(set(indices)) != len(indices):
        return VerifyResult(ok=False, reason="duplicate indices")

    # Walk the canonical tree: each level pairs neighbours.
    leaves = [_gen_leaf(input113, i // C.INDICES_PER_HASH, i % C.INDICES_PER_HASH) for i in indices]
    sub_indices = [[i] for i in indices]

    for r in range(C.EQUIHASH_K):
        if len(leaves) % 2 != 0:
            return VerifyResult(ok=False, reason="odd row count")
        new_leaves: List[bytes] = []
        new_subs: List[List[int]] = []
        for p in range(0, len(leaves), 2):
            a, b = leaves[p], leaves[p + 1]
            sa, sb = sub_indices[p], sub_indices[p + 1]
            # Canonical order: subtree with smaller min-index first.
            if sa[0] >= sb[0]:
                # Swap is INVALID for canonical encoding — reject.
                return VerifyResult(ok=False, reason="non-canonical order at round %d" % r)
            # First 16 bits must match (cbits = 16).
            if a[0] != b[0] or a[1] != b[1]:
                return VerifyResult(ok=False, reason="prefix mismatch at round %d" % r)
            xored = bytes(x ^ y for x, y in zip(a, b))
            new_leaves.append(xored)
            new_subs.append(sa + sb)
        leaves = new_leaves
        sub_indices = new_subs

    if len(leaves) != 1:
        return VerifyResult(ok=False, reason="tree did not collapse to one")
    if any(b != 0 for b in leaves[0]):
        return VerifyResult(ok=False, reason="final hash not zero")

    # Difficulty check.
    h = hashlib.sha256(soln_indices + input113[:81]).digest()
    if not _hash_under_target(h, target):
        return VerifyResult(ok=False, solution_hash_hex=h.hex(), reason="above target")
    return VerifyResult(ok=True, solution_hash_hex=h.hex())


def _hash_under_target(h: bytes, target: bytes) -> bool:
    for a, b in zip(h, target):
        if a < b:
            return True
        if a > b:
            return False
    return False


# Wire the fallback into cpu_verify when the Rust binary is unavailable.
_original_cpu_verify = cpu_verify  # type: ignore[name-defined]


def cpu_verify_with_fallback(
    challenge: bytes,
    miner_pubkey: bytes,
    block_height: int,
    nonce: bytes,
    soln_indices: bytes,
    target: bytes,
    n: int = C.EQUIHASH_N,
    k: int = C.EQUIHASH_K,
    binary_path: Optional[str] = None,
) -> VerifyResult:
    bin_path = binary_path or find_verifier_binary()
    if bin_path:
        return _original_cpu_verify(
            challenge=challenge,
            miner_pubkey=miner_pubkey,
            block_height=block_height,
            nonce=nonce,
            soln_indices=soln_indices,
            target=target,
            n=n,
            k=k,
            binary_path=bin_path,
        )
    return _python_verify(challenge, miner_pubkey, block_height, nonce, soln_indices, target)


# Expose the smart variant as the canonical entry point.
cpu_verify = cpu_verify_with_fallback  # type: ignore[assignment]
