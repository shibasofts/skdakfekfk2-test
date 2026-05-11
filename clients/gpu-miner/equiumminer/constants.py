"""On-chain + Equihash constants. Mirrors `crates/equihash-core` + `programs/equium`."""

from __future__ import annotations

# ---- Program identifiers ----
PROGRAM_ID = "ZKGMUfxiRCXFPnqz9zgqAnuqJy15jk7fKbR4o6FuEQM"
CONFIG_SEED = b"equium-config"
VAULT_SEED = b"equium-vault"

# ---- Equihash parameters (locked at program init) ----
EQUIHASH_N = 96
EQUIHASH_K = 5
# Personalization for the Equium input block (NOT the BLAKE2b personalization).
PERSONALIZATION = b"Equium-v1"  # 9 bytes
I_LEN = 9 + 32 + 32 + 8  # = 81

# Derived Equihash params for (96, 5):
#   cbits = n / (k+1) = 16  → 16-bit prefix match per round
#   cbytes = ceil(cbits/8) = 2
#   n_init = 2^(cbits+1) = 131072 initial leaves
#   indices_per_hash = 512 / n = 5 → each BLAKE2b output gives 5 leaves
#   blake2b_output_bytes = (512/n) * n / 8 = 60
#   leaf_bytes = n / 8 = 12
CBITS = EQUIHASH_N // (EQUIHASH_K + 1)
CBYTES = (CBITS + 7) // 8
N_INIT = 1 << (CBITS + 1)
INDICES_PER_HASH = 512 // EQUIHASH_N
BLAKE2B_OUTPUT_BYTES = INDICES_PER_HASH * EQUIHASH_N // 8
LEAF_BYTES = EQUIHASH_N // 8
SOLUTION_INDICES_COUNT = 1 << EQUIHASH_K  # = 32

# BLAKE2b personalization for Equihash (compatible with the upstream Zcash scheme).
# Layout: b"ZcashPoW" || n.to_le_bytes(4) || k.to_le_bytes(4)  → 16 bytes
BLAKE2B_PERSONAL = (
    b"ZcashPoW"
    + EQUIHASH_N.to_bytes(4, "little")
    + EQUIHASH_K.to_bytes(4, "little")
)
assert len(BLAKE2B_PERSONAL) == 16

# ---- Compressed solution size (what goes on-chain) ----
# Each index packs (cbits + 1) = 17 bits; total = 32 * 17 = 544 bits = 68 bytes.
COMPRESSED_SOLN_BYTES = (SOLUTION_INDICES_COUNT * (CBITS + 1) + 7) // 8

# ---- Solana program addresses ----
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
SLOT_HASHES_SYSVAR_ID = "SysvarS1otHashes111111111111111111111111111"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# ---- Anchor instruction discriminators (sha256("global:<name>")[:8]) ----
DISCRIMINATOR_MINE = bytes([59, 22, 178, 213, 139, 197, 160, 196])
DISCRIMINATOR_ADVANCE_EMPTY_ROUND = bytes([132, 60, 168, 44, 49, 255, 133, 255])

# ---- Token decimals ----
TOKEN_DECIMALS = 6
