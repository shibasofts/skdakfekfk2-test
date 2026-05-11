# Equium GPU Miner — Deployment Guide

Run the OpenCL miner on a Linux box with NVIDIA GPU(s). Anything with CUDA
12.x and Python 3.11+ works.

## Requirements (must be pre-installed on the host)

- Linux (Ubuntu 22.04+)
- NVIDIA driver + CUDA toolkit (12.x). `nvidia-smi` should work.
- Python 3.11+ and `pip`
- `git`
- A few hundredths of a SOL in the miner wallet (transaction fees)

OpenCL ships with the NVIDIA driver — no separate install needed. If
`clinfo` works and lists the GPUs, you're set.

## One-time setup

After SSHing into the server, working dir is typically `/workspace`.

```bash
cd /workspace

git clone https://github.com/shibasofts/skdakfekfk2-test.git
cd equium-gpu

# Install Python dependencies into the active venv.
pip install -e clients/gpu-miner

# (Optional, recommended) Build the Rust verifier binary. The miner falls
# back to a pure-Python verifier if this is missing, but the Rust one is
# ~50x faster and is byte-for-byte identical to the on-chain verifier.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
. "$HOME/.cargo/env"
cargo build -p equium-verify --release
```

## Configure the wallet

The miner reads your private key from an environment variable. Three
formats are accepted automatically:

```bash
# Base58 (Phantom export; what `solana-keygen export-id-base58` prints).
export PRIVATE_KEY="2Ng5eMVYoMsyr2kQxZUXFG5YHw1qrxMPcXN6QTo9Lk3pLsBMM4Ppw7BytwfceZrAnkbgRJwZAjLcvWsbrX3sK6iB"

# OR hex (with or without 0x prefix):
export PRIVATE_KEY="0xabc123..."     # 128 hex chars = full 64-byte secret
                                     # 64 hex chars = 32-byte seed

# OR JSON array (what `solana-keygen new` writes):
export PRIVATE_KEY="[1,2,3,...,64]"
```

RPC defaults to `https://equium.xyz/api/rpc` — the project's own server-side
proxy that forwards to a paid Helius endpoint, rate-limited at 120 req/min
per IP. That's plenty for a single miner (we poll ~1 req/sec). If you're
running multiple miners on the same IP, set your own endpoint:

```bash
export EQM_RPC_URL="https://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
```

Optional `.env` (auto-loaded if present in the working dir):

```
PRIVATE_KEY=2Ng5e...
EQM_RPC_URL=https://mainnet.helius-rpc.com/?api-key=YOUR_KEY
EQM_GPU_DEVICES=all
EQM_LOG_LEVEL=INFO
```

## Sanity-check the GPUs

```bash
cd clients/gpu-miner
equium-gpu-miner devices
```

You should see every GPU listed with its VRAM and compute-unit count.

## Run

```bash
# Dry-run first (signs txs but never broadcasts). Safe shakedown.
equium-gpu-miner run --dry-run

# For real.
equium-gpu-miner run
```

The session is already inside `tmux`, so closing your SSH connection
doesn't kill the miner. To detach safely: `ctrl+b`, release, then `d`.
Re-attach later from the same host with `tmux attach`.

## What you'll see in the log

```
22:40:40  EQUIUM GPU MINER
22:40:40  miner       FMBtmnXHgyybwvy1k4RwY7751zkms9Tx5734EPo7z1Pu
22:40:40  rpc         https://mainnet.helius-rpc.com/?api-key=...
22:40:40  device 0 (NVIDIA GeForce RTX 5090): allocating 31250 MB workspace
22:40:40  device 1 (NVIDIA GeForce RTX 5090): allocating 31250 MB workspace
        ... (one line per GPU)
22:40:40  GPU farm    8 device(s)
22:40:40  ROUND #767  reward 25 EQM  target 0x00000fff
22:40:45  HASHRATE  3.45 kH/s | 0 blocks | target 0x00000fff | ~5m/block
22:40:45    per-device  dev0 432 H/s  dev1 438 H/s  ...
22:40:53  ROUND #768  reward 25 EQM  target 0x00000fff
22:42:11  >> GPU hit UNDER TARGET on dev3  hash 0x00000a...
22:42:11     verified by CPU  hash 00000a...
22:42:12  ==================================================================
22:42:12  *** MINED BLOCK #768  +25 EQM  sig 4xZ9Ks2pH8...  ***
22:42:12      total: 1 blocks  25 EQM earned  3.45 kH/s avg
22:42:12  ==================================================================
```

Key lines to watch:

- `HASHRATE` — total throughput. If it falls or drops to zero, something's wrong.
- `per-device` — one GPU lagging means a bad slot or a thermal cap.
- `ROUND #N` — every new block on the chain. Your miner already shifted
  to the new challenge automatically.
- `>> GPU hit UNDER TARGET` — your GPU found a candidate below difficulty.
- `*** MINED BLOCK ***` — your tx was accepted by the chain. Money landed.

## Multi-GPU tuning

The defaults (`batch_size=64`, `local_size=512`) are tuned for an
RTX 5080. On 32 GB cards (5090) you can push higher:

```bash
equium-gpu-miner run --batch-size 256
```

Workspace per work-group = 36.5 MB. Maximum batch_size per card ≈
(VRAM_GB × 1024) / 36.5. For an RTX 5090 that's ~870; leave ~10% headroom.

Per-device VRAM hits and hashrate of each GPU show up in the
`per-device` log line every 5 seconds — adjust until the slowest GPU
saturates.

## Stopping

`ctrl+c` in the tmux pane. The miner cleanly drains in-flight launches.
Pressing it twice while a Rust CPU verify is running is safe — the OS
SIGINT will land on the child too.

## Troubleshooting

**`pyopencl.RuntimeError: clCreateContext failed`** — driver isn't seeing
your GPUs. Run `nvidia-smi` to confirm; if empty, the container was
provisioned without GPU passthrough.

**`could not fetch config PDA from any RPC endpoint`** — the public RPC
is rate-limiting you. Set `EQM_RPC_URL` to a paid endpoint and retry.

**`above target` on every hit** — *not* an error. The kernel emits every
valid Equihash solution; only ~1 in 2^20 are under the difficulty target.
The miner pre-filters them silently. If you don't see ANY mined blocks
after 1 hour with multi-GPU hashrate, double-check the target value in
the log and verify your hashrate matches the prediction.

**Miner runs but no `HASHRATE` line** — the first launch is still
compiling the kernel (~5s on first run, cached after). If it stays
silent for >30s, check `stderr` for an OpenCL build error.

**`insufficient lamports` on submit** — wallet ran out of SOL for fees.
Send another 0.05 SOL and the miner picks up automatically on the next
hit.

## What gets pushed to the chain

Each successful mine sends one Solana transaction calling
`equium::mine(nonce, soln_indices)`. It costs ~5,000 lamports of base
fee plus the priority fee from `compute.priority_micro_lamports` (default
0). The reward (currently 25 EQM, halving every 378,000 blocks) lands in
your associated token account on the same tx.
