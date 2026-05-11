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
    batch_size: int = 256
    local_size: int = 64
    workspace_bytes_per_nonce: int = 3_145_728  # informational; see gpu.WORKSPACE_PER_WI


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
            cfg.gpu.workspace_bytes_per_nonce = int(
                g.get("workspace_bytes_per_nonce", cfg.gpu.workspace_bytes_per_nonce)
            )
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
    """Load a Solana Keypair from either a path to a JSON array or a base58 string env."""
    from solders.keypair import Keypair  # local import to keep cli importable without solders

    # Allow env override: EQM_KEYPAIR_BASE58 (raw secret, never logged).
    raw = os.environ.get("EQM_KEYPAIR_BASE58")
    if raw:
        import base58
        return Keypair.from_bytes(base58.b58decode(raw.strip()))

    import json
    p = Path(path_or_str)
    if not p.exists():
        raise FileNotFoundError(
            f"keypair not found at {p}. Either set EQM_KEYPAIR_BASE58 or "
            f"place the JSON keypair at the configured path."
        )
    arr = json.loads(p.read_text())
    if not isinstance(arr, list) or len(arr) != 64:
        raise ValueError(f"keypair JSON at {p} must be a 64-byte array")
    return Keypair.from_bytes(bytes(arr))
