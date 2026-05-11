"""Build / sign / send the `mine` and `advance_empty_round` instructions."""

from __future__ import annotations

import logging
import struct
from typing import List

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction

from . import constants as C

logger = logging.getLogger(__name__)


def _account_meta(pk: Pubkey, *, signer: bool = False, writable: bool = False) -> AccountMeta:
    return AccountMeta(pubkey=pk, is_signer=signer, is_writable=writable)


def build_mine_ix(
    program_id: Pubkey,
    miner: Pubkey,
    config_pda: Pubkey,
    mint: Pubkey,
    vault_pda: Pubkey,
    miner_ata: Pubkey,
    token_program: Pubkey,
    nonce: bytes,
    soln_indices: bytes,
) -> Instruction:
    """Anchor `mine(nonce: [u8;32], soln_indices: Vec<u8>)`.

    Data layout = 8-byte discriminator || 32-byte nonce || 4-byte LE length || bytes.
    Account order MUST match programs/equium/src/lib.rs Mine context.
    """
    if len(nonce) != 32:
        raise ValueError(f"nonce must be 32 bytes, got {len(nonce)}")

    data = bytearray()
    data.extend(C.DISCRIMINATOR_MINE)
    data.extend(nonce)
    data.extend(struct.pack("<I", len(soln_indices)))
    data.extend(soln_indices)

    accounts = [
        _account_meta(miner, signer=True, writable=True),
        _account_meta(config_pda, writable=True),
        _account_meta(mint),
        _account_meta(vault_pda, writable=True),
        _account_meta(miner_ata, writable=True),
        _account_meta(token_program),
        _account_meta(Pubkey.from_string(C.ASSOCIATED_TOKEN_PROGRAM_ID)),
        _account_meta(Pubkey.from_string(C.SYSTEM_PROGRAM_ID)),
        _account_meta(Pubkey.from_string(C.SLOT_HASHES_SYSVAR_ID)),
    ]
    return Instruction(program_id=program_id, data=bytes(data), accounts=accounts)


def build_advance_empty_round_ix(
    program_id: Pubkey,
    caller: Pubkey,
    config_pda: Pubkey,
) -> Instruction:
    """Anchor `advance_empty_round()` — no args."""
    data = bytes(C.DISCRIMINATOR_ADVANCE_EMPTY_ROUND)
    accounts = [
        _account_meta(caller, signer=True),
        _account_meta(config_pda, writable=True),
        _account_meta(Pubkey.from_string(C.SLOT_HASHES_SYSVAR_ID)),
    ]
    return Instruction(program_id=program_id, data=data, accounts=accounts)


def send_mine_tx(
    client: Client,
    miner_kp: Keypair,
    program_id: Pubkey,
    config_pda: Pubkey,
    mint: Pubkey,
    vault_pda: Pubkey,
    miner_ata: Pubkey,
    token_program: Pubkey,
    nonce: bytes,
    soln_indices: bytes,
    cu_limit: int = 1_400_000,
    priority_micro_lamports: int = 0,
    skip_preflight: bool = True,
) -> str:
    """Build, sign, send the mine tx. Returns the signature string."""
    ixs: List[Instruction] = []
    if cu_limit > 0:
        ixs.append(set_compute_unit_limit(cu_limit))
    if priority_micro_lamports > 0:
        ixs.append(set_compute_unit_price(priority_micro_lamports))
    ixs.append(
        build_mine_ix(
            program_id=program_id,
            miner=miner_kp.pubkey(),
            config_pda=config_pda,
            mint=mint,
            vault_pda=vault_pda,
            miner_ata=miner_ata,
            token_program=token_program,
            nonce=nonce,
            soln_indices=soln_indices,
        )
    )
    bh = client.get_latest_blockhash(commitment=Confirmed).value.blockhash
    tx = Transaction.new_signed_with_payer(ixs, miner_kp.pubkey(), [miner_kp], bh)
    resp = client.send_transaction(
        tx, opts=TxOpts(skip_preflight=skip_preflight, preflight_commitment=Confirmed)
    )
    return str(resp.value)


def send_advance_empty_round_tx(
    client: Client,
    caller_kp: Keypair,
    program_id: Pubkey,
    config_pda: Pubkey,
    skip_preflight: bool = False,
) -> str:
    bh = client.get_latest_blockhash(commitment=Confirmed).value.blockhash
    ix = build_advance_empty_round_ix(program_id, caller_kp.pubkey(), config_pda)
    tx = Transaction.new_signed_with_payer([ix], caller_kp.pubkey(), [caller_kp], bh)
    resp = client.send_transaction(
        tx, opts=TxOpts(skip_preflight=skip_preflight, preflight_commitment=Confirmed)
    )
    return str(resp.value)


def classify_submit_err(s: str) -> str:
    if "0x1773" in s or "AboveTarget" in s:
        return "above target"
    if "0x1772" in s or "InvalidEquihash" in s:
        return "invalid equihash"
    if "0x1774" in s or "StaleChallenge" in s:
        return "stale challenge"
    if "0x177d" in s or "MiningNotOpen" in s:
        return "mining not open"
    if "BlockhashNotFound" in s or "blockhash not found" in s:
        return "blockhash expired"
    if "insufficient lamports" in s:
        return "not enough SOL"
    if "RoundStillActive" in s:
        return "round still active"
    return s[:120]
