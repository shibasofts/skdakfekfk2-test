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
    initial_cfg = None
    for _ in range(5):
        try:
            initial_cfg = chain.fetch_config(rpc.client, config_pda)
            break
        except Exception as e:
            logger.warning("rpc fetch_config failed: %s, rotating", e)
            rpc.rotate()
            time.sleep(1)
    if initial_cfg is None:
        raise RuntimeError("could not fetch config PDA from any RPC endpoint")
    if initial_cfg.equihash_n != C.EQUIHASH_N or initial_cfg.equihash_k != C.EQUIHASH_K:
        raise RuntimeError(
            f"this kernel is hardcoded for Equihash (96, 5); chain reports "
            f"({initial_cfg.equihash_n}, {initial_cfg.equihash_k})"
        )
    token_program = chain.fetch_mint_owner(rpc.client, initial_cfg.mint)
    miner_ata = chain.derive_ata(miner, initial_cfg.mint, token_program)

    logger.info("miner       %s", miner)
    logger.info("program     %s", program_id)
    logger.info("config PDA  %s", config_pda)
    logger.info("vault PDA   %s", vault_pda)
    logger.info("mint        %s (token program %s)", initial_cfg.mint, token_program)
    logger.info("miner ATA   %s", miner_ata)

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
    config_poll_interval = 1.0  # seconds

    try:
        while True:
            now = time.time()

            # Poll on-chain state at most ~1x/sec.
            if now - last_config_fetch >= config_poll_interval:
                last_config_fetch = now
                try:
                    onchain = chain.fetch_config(rpc.client, config_pda)
                except Exception as e:
                    logger.warning("config fetch failed: %s — rotating RPC", e)
                    rpc.rotate()
                    onchain = None

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
                            "round #%d opened · reward %d base · target 0x%s…",
                            onchain.block_height,
                            onchain.current_epoch_reward,
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
                logger.info("round stalled %.0fs — calling advance_empty_round", stall)
                try:
                    sig = submit.send_advance_empty_round_tx(
                        rpc.client, keypair, program_id, config_pda
                    )
                    logger.info("↳ advanced empty round · sig %s…", sig[:8])
                except Exception as e:
                    reason = submit.classify_submit_err(str(e))
                    logger.info("↳ %s", reason)

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

                # CPU verify — re-derive challenge from current on-chain state
                # so we never submit a stale candidate.
                cfg_now = onchain
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
                        "✓ GPU dev%d candidate VERIFIED · hash %s…",
                        hit.device_idx,
                        vr.solution_hash_hex[:12],
                    )
                else:
                    logger.info("✓ GPU dev%d candidate (skipping CPU verify)", hit.device_idx)

                if cfg.behaviour.dry_run:
                    logger.info("[dry-run] would submit mine tx — skipping")
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
                    logger.info(
                        "★ MINED #%d (+%d base) · sig %s… · total %d blocks · ~%.1f H/s",
                        cfg_now.block_height,
                        cfg_now.current_epoch_reward,
                        sig[:8],
                        stats.blocks_mined,
                        hashrate,
                    )
                except Exception as e:
                    reason = submit.classify_submit_err(str(e))
                    logger.warning("submit failed: %s", reason)

            if not drained_any:
                time.sleep(0.1)
    finally:
        farm.stop()
