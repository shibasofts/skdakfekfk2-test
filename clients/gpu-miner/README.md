# equium-gpu-miner

OpenCL GPU miner for Equium ($EQM) on Solana.

**Full server deploy guide:** [DEPLOY.md](DEPLOY.md).

## Install (Linux + NVIDIA + Python 3.11+)

```bash
git clone https://github.com/shibasofts/equium-gpu.git
cd equium-gpu
pip install -e clients/gpu-miner

# optional: faster Rust verifier
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
. "$HOME/.cargo/env"
cargo build -p equium-verify --release
```

## Configure

```bash
export PRIVATE_KEY="<base58 / hex / JSON-array>"
export EQM_RPC_URL="https://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
```

## Run

```bash
cd clients/gpu-miner
equium-gpu-miner devices
equium-gpu-miner run
```

`--dry-run` builds + signs but never broadcasts. `--batch-size N` per-GPU work-group count (default 64; bump to 256+ on 32 GB cards). `--devices "0,1"` mines on specific cards only.

## Files

| | |
|---|---|
| `equiumminer/kernels/equihash_96_5.cl` | OpenCL kernel: 1 nonce per WG, parallel BLAKE2b + bucket sort + Wagner |
| `equiumminer/gpu.py` | device discovery + per-device worker + farm |
| `equiumminer/chain.py` | EquiumConfig PDA reader, ATA derivation |
| `equiumminer/submit.py` | `mine` + `advance_empty_round` ix builders |
| `equiumminer/verify.py` | CPU re-verify (Rust binary if built, else Python fallback) |
| `equiumminer/miner.py` | main loop |
| `../../crates/equium-verify` | Rust verifier binary |
