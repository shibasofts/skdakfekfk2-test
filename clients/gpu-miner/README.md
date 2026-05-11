# equium-gpu-miner

OpenCL GPU miner for Equium ($EQM). Mirrors the structure of `shibasofts/hash256-miner` (OpenCL Keccak ERC-20 miner) but the kernel is Equihash (96, 5) and the chain integration is Solana.

> **Status: v1 — kernel correctness over speed.** One nonce per work-item, bucket sort by 16-bit prefix, sequential Wagner inside each work-item. Optimisation pass (one nonce per work-group, persistent kernel, parallel sort) is a follow-up.

## Install

Requires Python ≥ 3.11 and working **OpenCL** (vendor driver + ICD; NVIDIA's CUDA toolkit, AMD's ROCm/Adrenalin, or Intel's runtime all ship one).

```powershell
cd clients/gpu-miner
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .

# Optional but recommended: build the Rust verifier binary (same one the chain
# uses). If absent, verify.py falls back to a slower pure-Python verifier.
cd ..\..
cargo build -p equium-verify --release
```

Then sanity-check OpenCL discovery:

```powershell
equium-gpu-miner devices
```

## Configure

```powershell
cd clients\gpu-miner
copy miner.example.toml miner.toml
# Edit miner.toml. The keypair NEVER goes into the toml — only the path does.
```

Or use env vars:

```powershell
$env:EQM_RPC_URL = "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
$env:EQM_KEYPAIR_BASE58 = "<base58-encoded-64-byte-secret>"
$env:EQM_GPU_DEVICES = "all"
$env:EQM_DRY_RUN = "1"     # build/sign but don't broadcast
```

## Run

```powershell
# Dry run against mainnet — no key needed, just sanity-checks the loop:
equium-gpu-miner run --dry-run

# For real:
equium-gpu-miner run

# Single GPU:
equium-gpu-miner run --devices 0

# Override RPC:
equium-gpu-miner run --rpc "https://mainnet.helius-rpc.com/?api-key=KEY"
```

## How it works

```
chain.py  ──poll EquiumConfig PDA──▶ miner.py ──set_job(challenge,target,height)──▶ gpu.py (one OpenCL worker per device)
                                       ▲                                              │   Hit(nonce, soln)
                                       │                                              ▼
                                  submit.py ◀── verify.py (Rust binary or Python fallback) ◀── results queue
                                       │
                                       └── build / sign / send  mine(nonce, soln_indices)
```

- Each GPU worker runs `equihash_96_5.cl`, one nonce per work-item. Workspace
  (rows + bucket counts) lives in global memory, ~38 MB per work-item. On a
  32 GB 5090, that's room for ~500 concurrent work-items.
- Every GPU candidate is CPU-re-verified before submission. Disagreement
  between GPU and CPU = kernel bug; the hit is logged and dropped, not sent.
- Multi-GPU: one worker thread per device, disjoint random nonce bases.
- Solution binding to miner pubkey means workers can't be sniped — the I-block
  embeds your pubkey so a copyist would have to re-solve under their own key.

## Layout

| path | what |
|---|---|
| `equiumminer/kernels/equihash_96_5.cl` | OpenCL kernel: BLAKE2b leaf gen + 5 Wagner rounds with bucket sort |
| `equiumminer/gpu.py` | OpenCL device discovery, per-device `GpuWorker`, multi-GPU `GpuFarm` |
| `equiumminer/chain.py` | Solana RPC wrapper: EquiumConfig PDA reader, ATA derivation |
| `equiumminer/verify.py` | CPU re-verification — Rust binary if available, else pure Python |
| `equiumminer/submit.py` | `mine` + `advance_empty_round` instruction builders, tx send |
| `equiumminer/miner.py` | orchestrator: poll → set job → drain hits → verify → submit |
| `equiumminer/config.py` | `miner.toml` + env loading; keypair handling |
| `equiumminer/constants.py` | on-chain constants, Equihash params, instruction discriminators |
| `equiumminer/cli.py` | `equium-gpu-miner devices|run|selftest` entry points |
| `../../crates/equium-verify` | Rust binary that re-runs the canonical on-chain verifier from stdin |

## Tuning

- **`--batch-size`** (default 256). Each WI uses ~38 MB workspace; more WI = more parallelism but more VRAM. On 32 GB cards try 512–1024.
- **`--local-size`** (default 64). Work-group size. Equihash here is embarrassingly parallel across nonces so this doesn't matter much; tune ±2× for ~5–10% throughput on different vendors.
- **`compute.priority_micro_lamports`** in `miner.toml` — add a small priority fee if you're competing with other miners for the same round.

## Caveats

- This is v1. The kernel is correct (CPU-verifier catches any disagreement) but not maximally fast — a one-nonce-per-WI design leaves a lot of GPU idle. Expect ~5–20 kH/s per RTX 5090 in this revision; an optimized one-nonce-per-WG design can do 10–50×.
- Never put your private key in `miner.toml`. Use `EQM_KEYPAIR_BASE58` or a path to a JSON keypair file with permissions tightened.
