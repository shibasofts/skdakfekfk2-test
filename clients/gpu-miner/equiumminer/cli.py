"""CLI entry point. Mirrors the structure of the reference hash256-miner."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import constants as C
from .config import MinerConfig, load_config, load_keypair


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s · %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group(help="Equium OpenCL GPU miner.")
def main() -> None:
    pass


@main.command(help="List OpenCL platforms/devices visible on this box.")
def devices() -> None:
    import pyopencl as cl

    idx = 0
    for plat in cl.get_platforms():
        click.echo(f"Platform: {plat.name} ({plat.vendor})")
        for d in plat.get_devices():
            mem_mb = int(d.global_mem_size / (1 << 20))
            cu = d.max_compute_units
            click.echo(
                f"  [{idx}] {d.name}  · {mem_mb} MB · {cu} CUs · "
                f"max wg {d.max_work_group_size}"
            )
            idx += 1


@main.command(help="Run the miner.")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=Path("miner.toml"))
@click.option("--dry-run/--no-dry-run", default=None, help="Build & sign but don't broadcast.")
@click.option("--rpc", "rpc_url", help="Override network.rpc_url.")
@click.option("--devices", "device_spec", help='Override gpu.devices (e.g. "all" or "0,1").')
@click.option("--batch-size", type=int, help="Override gpu.batch_size.")
@click.option("--local-size", type=int, help="Override gpu.local_size.")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
)
def run(config_path, dry_run, rpc_url, device_spec, batch_size, local_size, log_level):
    cfg: MinerConfig = load_config(config_path if config_path.exists() else None)

    if dry_run is not None:
        cfg.behaviour.dry_run = dry_run
    if rpc_url:
        cfg.network.rpc_url = rpc_url
    if device_spec:
        cfg.gpu.devices = device_spec
    if batch_size:
        cfg.gpu.batch_size = batch_size
    if local_size:
        cfg.gpu.local_size = local_size
    if log_level:
        cfg.behaviour.log_level = log_level

    _setup_logging(cfg.behaviour.log_level)
    log = logging.getLogger("cli")

    log.info("rpc       %s", cfg.network.rpc_url)
    log.info("program   %s", cfg.network.program_id)
    log.info("dry_run   %s", cfg.behaviour.dry_run)
    log.info(
        "gpu       devices=%s batch=%d local=%d",
        cfg.gpu.devices,
        cfg.gpu.batch_size,
        cfg.gpu.local_size,
    )

    kp = load_keypair(cfg.wallet.keypair_path)
    log.info("miner     %s", kp.pubkey())

    from .miner import run_miner
    try:
        run_miner(cfg, kp)
    except KeyboardInterrupt:
        log.info("interrupted")
    except Exception:
        log.exception("miner crashed")
        sys.exit(1)


@main.command(help="Quick CPU self-test of the verifier binary against a synthetic input.")
@click.option("--log-level", default="INFO")
def selftest(log_level: str) -> None:
    _setup_logging(log_level)
    log = logging.getLogger("selftest")
    from .verify import find_verifier_binary

    bin_path = find_verifier_binary()
    if not bin_path:
        log.error(
            "equium-verify binary not found. Build it: "
            "`cargo build -p equium-verify --release` from the repo root."
        )
        sys.exit(2)
    log.info("found verifier: %s", bin_path)
    log.info("(self-test is a stub — real selftest requires a known-valid solution input)")


if __name__ == "__main__":
    main()
