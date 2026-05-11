"""Solana RPC reads: config PDA, mint owner, balances."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import List, Optional

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey

from . import constants as C

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EquiumConfig:
    """Decoded EquiumConfig PDA. Field order matches programs/equium/src/state.rs."""

    mint: Pubkey
    mineable_vault: Pubkey
    mineable_vault_bump: int
    config_bump: int
    genesis_slot: int
    genesis_unix_ts: int
    equihash_n: int
    equihash_k: int
    current_target: bytes  # [u8; 32] big-endian
    block_height: int
    current_challenge: bytes  # [u8; 32]
    current_round_open_slot: int
    current_round_open_unix_ts: int
    last_winner: Pubkey
    current_epoch_reward: int
    next_halving_block: int
    next_retarget_block: int
    last_retarget_unix_ts: int
    cumulative_mined: int
    empty_rounds: int
    mining_open: bool
    admin: Pubkey
    admin_renounced: bool


def _read_pubkey(buf: bytes, off: int) -> Pubkey:
    return Pubkey.from_bytes(buf[off : off + 32])


def _read_u64_le(buf: bytes, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def _read_i64_le(buf: bytes, off: int) -> int:
    return struct.unpack_from("<q", buf, off)[0]


def _read_u32_le(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def decode_config(data: bytes) -> EquiumConfig:
    """Decode raw account data into EquiumConfig.

    Account layout = 8 bytes anchor discriminator + fields in declaration order
    using Borsh (little-endian, dense). Sticks to the schema in
    programs/equium/src/state.rs (don't reorder without updating this).
    """
    if len(data) < 8:
        raise ValueError(f"config account too small: {len(data)} bytes")
    o = 8  # skip discriminator
    mint = _read_pubkey(data, o); o += 32
    mineable_vault = _read_pubkey(data, o); o += 32
    mineable_vault_bump = data[o]; o += 1
    config_bump = data[o]; o += 1
    genesis_slot = _read_u64_le(data, o); o += 8
    genesis_unix_ts = _read_i64_le(data, o); o += 8
    equihash_n = _read_u32_le(data, o); o += 4
    equihash_k = _read_u32_le(data, o); o += 4
    current_target = bytes(data[o : o + 32]); o += 32
    block_height = _read_u64_le(data, o); o += 8
    current_challenge = bytes(data[o : o + 32]); o += 32
    current_round_open_slot = _read_u64_le(data, o); o += 8
    current_round_open_unix_ts = _read_i64_le(data, o); o += 8
    last_winner = _read_pubkey(data, o); o += 32
    current_epoch_reward = _read_u64_le(data, o); o += 8
    next_halving_block = _read_u64_le(data, o); o += 8
    next_retarget_block = _read_u64_le(data, o); o += 8
    last_retarget_unix_ts = _read_i64_le(data, o); o += 8
    cumulative_mined = _read_u64_le(data, o); o += 8
    empty_rounds = _read_u64_le(data, o); o += 8
    mining_open = data[o] != 0; o += 1
    admin = _read_pubkey(data, o); o += 32
    admin_renounced = data[o] != 0
    return EquiumConfig(
        mint=mint,
        mineable_vault=mineable_vault,
        mineable_vault_bump=mineable_vault_bump,
        config_bump=config_bump,
        genesis_slot=genesis_slot,
        genesis_unix_ts=genesis_unix_ts,
        equihash_n=equihash_n,
        equihash_k=equihash_k,
        current_target=current_target,
        block_height=block_height,
        current_challenge=current_challenge,
        current_round_open_slot=current_round_open_slot,
        current_round_open_unix_ts=current_round_open_unix_ts,
        last_winner=last_winner,
        current_epoch_reward=current_epoch_reward,
        next_halving_block=next_halving_block,
        next_retarget_block=next_retarget_block,
        last_retarget_unix_ts=last_retarget_unix_ts,
        cumulative_mined=cumulative_mined,
        empty_rounds=empty_rounds,
        mining_open=mining_open,
        admin=admin,
        admin_renounced=admin_renounced,
    )


def find_config_pda(program_id: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([C.CONFIG_SEED], program_id)[0]


def find_vault_pda(program_id: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([C.VAULT_SEED], program_id)[0]


class RpcRotator:
    """Round-robin RPC client wrapper with fallback list. Switches on connection
    error (not on RPC-side errors like rate limits, which raise inside the call)."""

    def __init__(self, primary: str, fallbacks: Optional[List[str]] = None, commitment: str = Confirmed):
        urls = [primary] + list(fallbacks or [])
        self._clients = [Client(u, commitment=commitment) for u in urls]
        self._urls = urls
        self._idx = 0

    @property
    def client(self) -> Client:
        return self._clients[self._idx]

    @property
    def url(self) -> str:
        return self._urls[self._idx]

    def rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self._clients)
        logger.info("rotated RPC → %s", self.url)


def fetch_config(client: Client, config_pda: Pubkey) -> Optional[EquiumConfig]:
    resp = client.get_account_info(config_pda, commitment=Confirmed)
    if resp.value is None:
        return None
    return decode_config(bytes(resp.value.data))


def fetch_mint_owner(client: Client, mint: Pubkey) -> Pubkey:
    """Token program that owns the mint (classic SPL or Token-2022)."""
    resp = client.get_account_info(mint, commitment=Confirmed)
    if resp.value is None:
        raise RuntimeError(f"mint {mint} not found")
    return resp.value.owner


def derive_ata(owner: Pubkey, mint: Pubkey, token_program: Pubkey) -> Pubkey:
    """ATA derivation. Seeds = [owner, token_program, mint], program = ATA program."""
    ata_program = Pubkey.from_string(C.ASSOCIATED_TOKEN_PROGRAM_ID)
    pda, _ = Pubkey.find_program_address(
        [bytes(owner), bytes(token_program), bytes(mint)], ata_program
    )
    return pda


def build_input_block(challenge: bytes, miner_pubkey: bytes, block_height: int) -> bytes:
    """Reproduce equihash_core::challenge::build_input exactly."""
    assert len(challenge) == 32
    assert len(miner_pubkey) == 32
    buf = bytearray(C.I_LEN)
    buf[0:9] = C.PERSONALIZATION
    buf[9:41] = challenge
    buf[41:73] = miner_pubkey
    buf[73:81] = block_height.to_bytes(8, "little")
    return bytes(buf)
