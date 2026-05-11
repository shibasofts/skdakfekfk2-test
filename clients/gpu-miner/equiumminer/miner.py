"""Main mining loop. Drives the GPU farm + chain reads + tx submission.

State machine (per iteration):
  1. Read on-chain config PDA (round + target + challenge).
  2. If round changed → push a fresh Job to the GpuFarm; reset round stats.
  3. Drain the results queue (non-blocking). For each hit:
       a. CPU-verify with the Rust verifier.
       b. If valid: build + sign + submit `mine` tx, on success bump counters.
       c. If invalid: log loudly (kernel bug or stale).
  4. Handle empty-round watchdog (advance_empty_round if chain stalled).
  5. Sleep briefly and loop.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from . import chain
from . import constants as C
from . import submit
from .config import MinerConfig
from .gpu import GpuFarm, Job, build_input113, compute_blake2b_h_base
from .verify import cpu_verify, find_verifier_binary

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    blocks_mined: int = 0
    total_earned_base: int = 0
    started_at: float = 0.0


def hash_under_target(hash_bytes: bytes, target: bytes) -> bool:
    """256-bit big-endian lex compare. Mirrors equihash-core::hash_under_target."""
    for a, b in zip(hash_bytes, target):
        if a < b:
            return True
        if a > b:
            return False
    return False


def _format_base(base: int) -> str:
    """Format an EQM base-unit amount (6 decimals) for display."""
    whole = base // 1_000_000
    frac = base % 1_000_000
    if frac == 0:
        return str(whole)
    return f"{whole}.{frac:06d}".rstrip("0").rstrip(".")


def _fmt_hashrate(h: float) -> str:
    if h >= 1_000_000:
        return f"{h/1_000_000:.2f} MH/s"
    if h >= 1_000:
        return f"{h/1_000:.2f} kH/s"
    return f"{h:.1f} H/s"


def _fmt_duration(sec: float) -> str:
    if sec >= 3600:
        return f"{sec/3600:.1f}h"
    if sec >= 60:
        return f"{sec/60:.1f}m"
    return f"{sec:.0f}s"


def _estimate_seconds_to_block(hashrate: float, target: bytes) -> Optional[float]:
    """Rough expected seconds-to-block given the 256-bit target. p(hit) ≈ target/2^256.
    Counts only the *target* probability — Equihash validity is ~50% per nonce,
    so the effective per-nonce hit chance is ~half this; we apply that factor."""
    if hashrate <= 0 or len(target) != 32:
        return None
    target_int = int.from_bytes(target, "big")
    if target_int == 0:
        return None
    # Probability a uniformly-random hash is under target.
    p_target = target_int / (1 << 256)
    # ~50% of nonces yield a valid Equihash solution; combine.
    p_hit = p_target * 0.5
    expected_attempts = 1.0 / max(p_hit, 1e-30)
    return expected_attempts / hashrate


def run_miner(cfg: MinerConfig, keypair: Keypair) -> None:
    program_id = Pubkey.from_string(cfg.network.program_id)
    config_pda = chain.find_config_pda(program_id)
    vault_pda = chain.find_vault_pda(program_id)
    miner = keypair.pubkey()
    miner_bytes = bytes(miner)

    rpc = chain.RpcRotator(cfg.network.rpc_url, cfg.network.rpc_fallbacks)
    h_base = compute_blake2b_h_base()
    verifier_path = find_verifier_binary()
    if cfg.behaviour.verify_on_cpu and not verifier_path:
        logger.warning(
            "verify_on_cpu=true but equium-verify binary not found. Build with: "
            "cargo build -p equium-verify --release"
        )

    # Resolve token program once (mint owner). If mint is changed off-chain we'd
    # need to restart; the program never rewrites it once authority is revoked.
    # Never give up — the network may be transiently degraded across every
    # provider, but the miner should resume the moment any of them comes back.
    initial_cfg = None
    attempt = 0
    while initial_cfg is None:
        attempt += 1
        try:
            initial_cfg = chain.fetch_config(rpc.client, config_pda)
            if initial_cfg is not None:
                break
            logger.warning(
                "rpc fetch_config returned None on %s (wrong network?) - rotating",
                rpc.url,
            )
        except Exception as e:
            logger.warning(
                "rpc fetch_config failed on %s: %s: %s, rotating",
                rpc.url,
                type(e).__name__,
                str(e) or repr(e),
            )
        rpc.rotate()
        # Tight retry — the rotator already moved on; sleeping >1s here only
        # delays the start when one endpoint is briefly down.
        time.sleep(0.3 if attempt < 10 else 1.0)
    if initial_cfg.equihash_n != C.EQUIHASH_N or initial_cfg.equihash_k != C.EQUIHASH_K:
        raise RuntimeError(
            f"this kernel is hardcoded for Equihash (96, 5); chain reports "
            f"({initial_cfg.equihash_n}, {initial_cfg.equihash_k})"
        )
    token_program = chain.fetch_mint_owner(rpc.client, initial_cfg.mint)
    miner_ata = chain.derive_ata(miner, initial_cfg.mint, token_program)

    logger.info("=" * 66)
    logger.info("EQUIUM GPU MINER")
    logger.info("=" * 66)
    logger.info("miner       %s", miner)
    logger.info("program     %s", program_id)
    logger.info("mint        %s", initial_cfg.mint)
    logger.info("miner ATA   %s", miner_ata)
    logger.info("rpc         %s", cfg.network.rpc_url)
    logger.info("-" * 66)

    # Spin up GPU farm.
    from .gpu import select_devices
    device_indices = select_devices(cfg.gpu.devices)
    if not device_indices:
        raise RuntimeError("no GPU devices selected")
    farm = GpuFarm(device_indices, cfg.gpu.batch_size, cfg.gpu.local_size)
    logger.info("GPU farm    %d device(s): %s", len(device_indices), farm.device_names)
    farm.start()

    stats = SessionStats(started_at=time.time())
    current_height: Optional[int] = None
    last_height_change_at = time.time()
    last_advance_attempt_at = 0.0
    last_config_fetch = 0.0
    last_hashrate_log = time.time()

    try:
        while True:
            now = time.time()

            # Poll on-chain state ~1x/sec. On RPC failure: log, rotate, and
            # try the NEXT endpoint immediately (next loop iteration). Don't
            # slow the cadence — the miner depends on fresh challenge state.
            if now - last_config_fetch >= 1.0:
                last_config_fetch = now
                try:
                    onchain = chain.fetch_config(rpc.client, config_pda)
                except Exception as e:
                    logger.warning(
                        "config fetch failed on %s: %s: %s - rotating",
                        rpc.url,
                        type(e).__name__,
                        str(e) or repr(e),
                    )
                    rpc.rotate()
                    onchain = None
                    last_config_fetch = 0.0  # retry the new endpoint immediately

                if onchain is not None:
                    if not onchain.mining_open:
                        logger.warning("mining not open yet — vault unfunded?")
                        farm.set_job(None)
                        time.sleep(3)
                        continue

                    if onchain.block_height != current_height:
                        current_height = onchain.block_height
                        last_height_change_at = now
                        last_advance_attempt_at = 0.0
                        logger.info(
                            "ROUND #%d  reward %s EQM  target 0x%s",
                            onchain.block_height,
                            _format_base(onchain.current_epoch_reward),
                            onchain.current_target[:4].hex(),
                        )
                        # Build a fresh nonce template — high 24 bytes random,
                        # low 8 swept by the kernel.
                        nonce_template = secrets.token_bytes(24) + b"\x00" * 8
                        input113 = build_input113(
                            onchain.current_challenge,
                            miner_bytes,
                            onchain.block_height,
                            nonce_template,
                        )
                        farm.set_job(
                            Job(
                                block_height=onchain.block_height,
                                challenge=onchain.current_challenge,
                                target=onchain.current_target,
                                miner_pubkey=miner_bytes,
                                h_base=h_base,
                                input113=input113,
                            )
                        )

            # Empty-round watchdog
            stall = now - last_height_change_at
            cooled = now - last_advance_attempt_at >= cfg.empty_round.cooldown_secs
            if (
                stall >= cfg.empty_round.stall_secs
                and cooled
                and current_height is not None
            ):
                last_advance_attempt_at = now
                logger.info("round stalled %.0fs -- calling advance_empty_round", stall)
                try:
                    sig = submit.send_advance_empty_round_tx(
                        rpc.client, keypair, program_id, config_pda
                    )
                    logger.info("   advanced empty round  sig %s...", sig[:10])
                except Exception as e:
                    reason = submit.classify_submit_err(str(e))
                    logger.info("   advance failed: %s", reason)

            # Drain GPU results
            drained_any = False
            while True:
                try:
                    hit = farm.results.get_nowait()
                except Exception:
                    break
                drained_any = True
                if onchain is None or hit.block_height != current_height:
                    logger.debug("dropping stale hit from height %d", hit.block_height)
                    continue

                # FAST PRE-FILTER: kernel emits every valid Equihash solution,
                # but only ~1 in 2^20 is under target. SHA-256 + lex compare is
                # microseconds; full Equihash re-verify is ~50ms. Drop hits that
                # would fail target before paying full verify cost.
                cfg_now = onchain
                input81 = chain.build_input_block(
                    cfg_now.current_challenge, miner_bytes, cfg_now.block_height
                )
                pre_hash = hashlib.sha256(hit.soln_indices + input81).digest()
                if not hash_under_target(pre_hash, cfg_now.current_target):
                    # Silent drop — expected for nearly all hits.
                    continue
                logger.info(
                    ">> GPU hit UNDER TARGET on dev%d  hash 0x%s...",
                    hit.device_idx,
                    pre_hash[:6].hex(),
                )

                if cfg.behaviour.verify_on_cpu:
                    vr = cpu_verify(
                        challenge=cfg_now.current_challenge,
                        miner_pubkey=miner_bytes,
                        block_height=cfg_now.block_height,
                        nonce=hit.nonce,
                        soln_indices=hit.soln_indices,
                        target=cfg_now.current_target,
                        n=cfg_now.equihash_n,
                        k=cfg_now.equihash_k,
                        binary_path=verifier_path,
                    )
                    if not vr.ok:
                        logger.warning(
                            "GPU hit dev%d rejected by CPU verify: %s (hash=%s)",
                            hit.device_idx,
                            vr.reason,
                            vr.solution_hash_hex or "?",
                        )
                        continue
                    logger.info(
                        "   verified by CPU  hash %s...",
                        vr.solution_hash_hex[:16],
                    )
                else:
                    logger.info("   (CPU verify skipped)")

                if cfg.behaviour.dry_run:
                    logger.info("[dry-run] would submit mine tx -- skipping")
                    continue

                # Submit.
                try:
                    sig = submit.send_mine_tx(
                        rpc.client,
                        keypair,
                        program_id,
                        config_pda,
                        cfg_now.mint,
                        vault_pda,
                        miner_ata,
                        token_program,
                        hit.nonce,
                        hit.soln_indices,
                        cu_limit=cfg.compute.cu_limit,
                        priority_micro_lamports=cfg.compute.priority_micro_lamports,
                    )
                    stats.blocks_mined += 1
                    stats.total_earned_base += cfg_now.current_epoch_reward
                    elapsed = max(time.time() - stats.started_at, 0.001)
                    total_attempts = sum(farm.stats_attempts)
                    hashrate = total_attempts / elapsed
                    logger.info("=" * 66)
                    logger.info(
                        "*** MINED BLOCK #%d  +%s EQM  sig %s...  ***",
                        cfg_now.block_height,
                        _format_base(cfg_now.current_epoch_reward),
                        sig[:10],
                    )
                    logger.info(
                        "    total: %d blocks  %s EQM earned  %s avg",
                        stats.blocks_mined,
                        _format_base(stats.total_earned_base),
                        _fmt_hashrate(hashrate),
                    )
                    logger.info("=" * 66)
                except Exception as e:
                    reason = submit.classify_submit_err(str(e))
                    logger.warning("submit failed: %s", reason)

            # Periodic hashrate log (every 5s). Includes a per-device
            # breakdown (helpful for spotting one underperforming GPU on a
            # multi-card box) and an expected-time-to-block estimate based on
            # the current target.
            if now - last_hashrate_log >= 5.0:
                last_hashrate_log = now
                elapsed = max(now - stats.started_at, 0.001)
                total_attempts = sum(farm.stats_attempts)
                hashrate = total_attempts / elapsed
                target_hex = onchain.current_target[:4].hex() if onchain else "?"
                etb = _estimate_seconds_to_block(hashrate, onchain.current_target) if onchain else None
                etb_str = f" | ~{_fmt_duration(etb)}/block" if etb else ""
                logger.info(
                    "HASHRATE  %s | %d blocks | target 0x%s%s",
                    _fmt_hashrate(hashrate),
                    stats.blocks_mined,
                    target_hex,
                    etb_str,
                )
                if len(farm.device_names) > 1:
                    parts = []
                    for i, name in enumerate(farm.device_names):
                        dev_hr = farm.stats_attempts[i] / elapsed
                        parts.append(f"dev{i} {_fmt_hashrate(dev_hr)}")
                    logger.info("  per-device  %s", "  ".join(parts))

            if not drained_any:
                time.sleep(0.1)
    finally:
        farm.stop()
