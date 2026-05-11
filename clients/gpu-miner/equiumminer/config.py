"""Config loading: miner.toml + env vars + a few CLI overrides.

Env var prefix: EQM_*. Env wins over toml. Secrets (keypair contents) never
touch toml; only the path does."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from . import constants as C


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return int(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class NetworkConfig:
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    rpc_fallbacks: List[str] = field(default_factory=list)
    program_id: str = C.PROGRAM_ID


@dataclass
class WalletConfig:
    keypair_path: str = "wallet.json"


@dataclass
class GpuConfig:
    devices: Union[str, List[int]] = "all"
    # batch_size = number of work-groups (= nonces) per kernel launch
    batch_size: int = 128
    # local_size = WIs per WG; one WG processes one nonce
    local_size: int = 256


@dataclass
class BehaviourConfig:
    dry_run: bool = False
    verify_on_cpu: bool = True
    log_level: str = "INFO"
    max_nonces_per_round: int = 65536


@dataclass
class ComputeConfig:
    cu_limit: int = 1_400_000
    priority_micro_lamports: int = 0


@dataclass
class EmptyRoundConfig:
    stall_secs: int = 75
    cooldown_secs: int = 30


@dataclass
class MinerConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    behaviour: BehaviourConfig = field(default_factory=BehaviourConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    empty_round: EmptyRoundConfig = field(default_factory=EmptyRoundConfig)


def load_config(toml_path: Optional[Path] = None) -> MinerConfig:
    cfg = MinerConfig()

    if toml_path and toml_path.exists():
        raw = tomllib.loads(toml_path.read_text())
        if "network" in raw:
            n = raw["network"]
            cfg.network.rpc_url = n.get("rpc_url", cfg.network.rpc_url)
            cfg.network.rpc_fallbacks = list(n.get("rpc_fallbacks", []))
            cfg.network.program_id = n.get("program_id", cfg.network.program_id)
        if "wallet" in raw:
            cfg.wallet.keypair_path = raw["wallet"].get("keypair_path", cfg.wallet.keypair_path)
        if "gpu" in raw:
            g = raw["gpu"]
            cfg.gpu.devices = g.get("devices", cfg.gpu.devices)
            cfg.gpu.batch_size = int(g.get("batch_size", cfg.gpu.batch_size))
            cfg.gpu.local_size = int(g.get("local_size", cfg.gpu.local_size))
        if "behaviour" in raw:
            b = raw["behaviour"]
            cfg.behaviour.dry_run = bool(b.get("dry_run", cfg.behaviour.dry_run))
            cfg.behaviour.verify_on_cpu = bool(b.get("verify_on_cpu", cfg.behaviour.verify_on_cpu))
            cfg.behaviour.log_level = b.get("log_level", cfg.behaviour.log_level)
            cfg.behaviour.max_nonces_per_round = int(
                b.get("max_nonces_per_round", cfg.behaviour.max_nonces_per_round)
            )
        if "compute" in raw:
            c = raw["compute"]
            cfg.compute.cu_limit = int(c.get("cu_limit", cfg.compute.cu_limit))
            cfg.compute.priority_micro_lamports = int(
                c.get("priority_micro_lamports", cfg.compute.priority_micro_lamports)
            )
        if "empty_round" in raw:
            e = raw["empty_round"]
            cfg.empty_round.stall_secs = int(e.get("stall_secs", cfg.empty_round.stall_secs))
            cfg.empty_round.cooldown_secs = int(e.get("cooldown_secs", cfg.empty_round.cooldown_secs))

    # Env overrides
    cfg.network.rpc_url = _env_str("EQM_RPC_URL", cfg.network.rpc_url) or cfg.network.rpc_url
    fbs = _env_str("EQM_RPC_FALLBACKS")
    if fbs:
        cfg.network.rpc_fallbacks = [x.strip() for x in fbs.split(",") if x.strip()]
    cfg.network.program_id = _env_str("EQM_PROGRAM_ID", cfg.network.program_id) or cfg.network.program_id
    cfg.wallet.keypair_path = _env_str("EQM_KEYPAIR_PATH", cfg.wallet.keypair_path) or cfg.wallet.keypair_path

    dev = _env_str("EQM_GPU_DEVICES")
    if dev:
        cfg.gpu.devices = dev
    cfg.gpu.batch_size = _env_int("EQM_GPU_BATCH_SIZE", cfg.gpu.batch_size) or cfg.gpu.batch_size
    cfg.gpu.local_size = _env_int("EQM_GPU_LOCAL_SIZE", cfg.gpu.local_size) or cfg.gpu.local_size

    cfg.behaviour.dry_run = _env_bool("EQM_DRY_RUN", cfg.behaviour.dry_run)
    cfg.behaviour.verify_on_cpu = _env_bool("EQM_VERIFY_ON_CPU", cfg.behaviour.verify_on_cpu)
    cfg.behaviour.log_level = _env_str("EQM_LOG_LEVEL", cfg.behaviour.log_level) or cfg.behaviour.log_level

    cfg.compute.cu_limit = _env_int("EQM_CU_LIMIT", cfg.compute.cu_limit) or cfg.compute.cu_limit
    cfg.compute.priority_micro_lamports = _env_int(
        "EQM_PRIORITY_MICRO_LAMPORTS", cfg.compute.priority_micro_lamports
    ) or cfg.compute.priority_micro_lamports

    return cfg


def load_keypair(path_or_str: str):
    """Load a Solana Keypair from one of: env var (preferred) or a file path.

    Env vars (checked in order):
      PRIVATE_KEY               base58 / hex / JSON-array secret (full 64 bytes or 32-byte seed)
      EQM_KEYPAIR_BASE58        base58 secret (legacy alias)
      EQM_KEYPAIR_HEX           hex (with or without 0x prefix)

    File fallback: the configured `wallet.keypair_path` is read as JSON array,
    base58, or hex, auto-detected.

    The raw secret is never printed.
    """
    from solders.keypair import Keypair  # local import to keep cli importable without solders

    for env_name in ("PRIVATE_KEY", "EQM_KEYPAIR_BASE58", "EQM_KEYPAIR_HEX"):
        raw = os.environ.get(env_name)
        if raw and raw.strip():
            return _parse_secret(raw.strip())

    p = Path(path_or_str)
    if not p.exists():
        raise FileNotFoundError(
            f"keypair not found. Either set the PRIVATE_KEY env var or place "
            f"a keypair file at '{p}' (JSON array, base58, or hex)."
        )
    return _parse_secret(p.read_text().strip())


def _parse_secret(s: str):
    """Auto-detect format. Order: JSON-array, hex (with/without 0x), base58."""
    from solders.keypair import Keypair

    s = s.strip()
    if s.startswith("["):
        import json
        arr = json.loads(s)
        if not isinstance(arr, list) or len(arr) not in (32, 64):
            raise ValueError("JSON keypair must be a 32- or 64-byte array")
        return _keypair_from_bytes(bytes(arr))

    # Hex (with 0x or pure)
    hex_str = s[2:] if s.lower().startswith("0x") else s
    if all(c in "0123456789abcdefABCDEF" for c in hex_str) and len(hex_str) in (64, 128):
        return _keypair_from_bytes(bytes.fromhex(hex_str))

    # Base58 fallback
    import base58
    try:
        decoded = base58.b58decode(s)
    except Exception as e:
        raise ValueError(
            "couldn't parse secret. Expected one of: JSON array, hex (64 or 128 chars), "
            "base58. " + str(e)
        )
    return _keypair_from_bytes(decoded)


def _keypair_from_bytes(b: bytes):
    """Accept either a 64-byte secret key or a 32-byte seed."""
    from solders.keypair import Keypair
    if len(b) == 64:
        return Keypair.from_bytes(b)
    if len(b) == 32:
        return Keypair.from_seed(b)
    raise ValueError(f"expected 32- or 64-byte secret, got {len(b)} bytes")
