/*
 * Equihash (96, 5) OpenCL kernel for Equium.
 *
 * Strategy: one nonce per work-item. Each work-item runs the full Wagner solve
 * inside its own workspace slice in global memory. Bucket-sort by 16-bit prefix
 * (cbits = 96/(5+1) = 16) replaces a comparison sort.
 *
 * Hardcoded params:
 *   N_INIT          = 131072  (= 2^(cbits+1))
 *   CBITS           = 16
 *   CBYTES          = 2
 *   LEAF_BYTES      = 12      (= 96/8)
 *   INDICES_PER     = 5       (= 512/96)
 *   BLAKE2B_OUT     = 60      (bytes per BLAKE2b call)
 *   SOLN_INDICES    = 32      (= 2^k)
 *   ROW_HASH_BYTES  = 12      (max — shrinks per round)
 *   ROW_IDX_MAX     = 32
 *
 * Per work-item global workspace ≈ 2 buffers × 131072 rows × 144 bytes
 *                                  + 65536 × 4 bucket counts
 *                                  + 65536 × 4 bucket starts
 *                                ≈ 38 MB / WI.
 */

#define N_INIT          131072u
#define CBITS           16u
#define CBYTES          2u
#define LEAF_BYTES      12u
#define INDICES_PER     5u
#define BLAKE2B_OUT     60u
#define BLAKE2B_OUT_64  8u           /* 60 bytes lives in h[0..8] u64 words (with 4 unused trailing bytes) */
#define SOLN_INDICES    32u
#define K_ROUNDS        5u
#define BUCKETS         65536u       /* = 2^CBITS */
#define ROW_HASH_BYTES  12u
#define ROW_IDX_MAX     32u
#define ROW_BYTES       144u         /* 12 hash + 4 count + 128 indices = 144, aligned */
#define MAX_HITS_PER_WI 4u

/* ===================================================================
 * BLAKE2b
 * =================================================================== */

__constant ulong BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL,
};

__constant uchar BLAKE2B_SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
};

#define ROTR64(x, n) rotate((ulong)(x), (ulong)(64 - (n)))

#define G(r, i, a, b, c, d) do {                              \
    a = a + b + m[BLAKE2B_SIGMA[r][2*i  ]];                   \
    d = ROTR64(d ^ a, 32);                                    \
    c = c + d;                                                \
    b = ROTR64(b ^ c, 24);                                    \
    a = a + b + m[BLAKE2B_SIGMA[r][2*i+1]];                   \
    d = ROTR64(d ^ a, 16);                                    \
    c = c + d;                                                \
    b = ROTR64(b ^ c, 63);                                    \
} while (0)

#define ROUND(r) do {                                         \
    G(r, 0, v[ 0], v[ 4], v[ 8], v[12]);                      \
    G(r, 1, v[ 1], v[ 5], v[ 9], v[13]);                      \
    G(r, 2, v[ 2], v[ 6], v[10], v[14]);                      \
    G(r, 3, v[ 3], v[ 7], v[11], v[15]);                      \
    G(r, 4, v[ 0], v[ 5], v[10], v[15]);                      \
    G(r, 5, v[ 1], v[ 6], v[11], v[12]);                      \
    G(r, 6, v[ 2], v[ 7], v[ 8], v[13]);                      \
    G(r, 7, v[ 3], v[ 4], v[ 9], v[14]);                      \
} while (0)

/* Single-block BLAKE2b compression. h[8] is hash state, m[16] is message words
 * (already LE-loaded), t is bytes-counter LOW (we never exceed 128 bytes total),
 * last=1 for the final/only block. */
static inline void blake2b_compress(
    ulong h[8],
    const ulong m[16],
    ulong t,
    int last
) {
    ulong v[16];
    for (int i = 0; i < 8; ++i) v[i] = h[i];
    v[ 8] = BLAKE2B_IV[0];
    v[ 9] = BLAKE2B_IV[1];
    v[10] = BLAKE2B_IV[2];
    v[11] = BLAKE2B_IV[3];
    v[12] = BLAKE2B_IV[4] ^ t;
    v[13] = BLAKE2B_IV[5];
    v[14] = last ? (BLAKE2B_IV[6] ^ 0xFFFFFFFFFFFFFFFFUL) : BLAKE2B_IV[6];
    v[15] = BLAKE2B_IV[7];

    ROUND(0);  ROUND(1);  ROUND(2);  ROUND(3);
    ROUND(4);  ROUND(5);  ROUND(6);  ROUND(7);
    ROUND(8);  ROUND(9);  ROUND(10); ROUND(11);

    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
}

/* ===================================================================
 * Leaf hashing
 * =================================================================== */

/* Pack 8 bytes LE into a ulong. */
static inline ulong load_u64_le(const uchar *p) {
    ulong x = 0;
    for (int i = 7; i >= 0; --i) x = (x << 8) | (ulong)p[i];
    return x;
}

/* Generate one BLAKE2b output (60 bytes, written into out as LE bytes from h[0..8]).
 * input is 113 bytes; we append batch_idx as 4 LE bytes => 117 total, in one block. */
static void blake2b_leaf_batch(
    const ulong h_base[8],     /* precomputed h after IV XOR param */
    const uchar *input113,     /* 113 bytes: persn (9) + challenge (32) + miner (32) + height (8) + nonce (32) */
    uint batch_idx,            /* appended as u32 LE */
    uchar out60[60]
) {
    /* Build the 128-byte block: 113 bytes of input + 4 bytes batch_idx + 11 zero pad. */
    uchar block[128];
    for (uint i = 0; i < 113; ++i) block[i] = input113[i];
    block[113] = (uchar)(batch_idx       & 0xff);
    block[114] = (uchar)((batch_idx >> 8) & 0xff);
    block[115] = (uchar)((batch_idx >> 16) & 0xff);
    block[116] = (uchar)((batch_idx >> 24) & 0xff);
    for (uint i = 117; i < 128; ++i) block[i] = 0;

    ulong m[16];
    for (uint i = 0; i < 16; ++i) {
        m[i] = load_u64_le(&block[i * 8]);
    }

    ulong h[8];
    for (int i = 0; i < 8; ++i) h[i] = h_base[i];

    blake2b_compress(h, m, /*t=*/117UL, /*last=*/1);

    /* Emit 60 bytes LE from h[0..8]. */
    for (uint w = 0; w < 8; ++w) {
        ulong x = h[w];
        for (uint b = 0; b < 8; ++b) {
            uint off = w * 8 + b;
            if (off < 60) out60[off] = (uchar)((x >> (8 * b)) & 0xff);
        }
    }
}

/* ===================================================================
 * Row layout in global workspace
 *
 *   row_bytes = 144
 *   bytes [ 0.. 12)  current hash (left-justified; round r consumes r*CBYTES leading zeros)
 *   bytes [12..16)   index count (uint LE)
 *   bytes [16..144)  indices, each u32 LE, up to 32 entries
 * =================================================================== */

#define ROW_HASH_OFF   0u
#define ROW_COUNT_OFF  12u
#define ROW_IDX_OFF    16u

static inline uint row_count(__global const uchar *row) {
    return (uint)row[ROW_COUNT_OFF]
         | ((uint)row[ROW_COUNT_OFF + 1] << 8)
         | ((uint)row[ROW_COUNT_OFF + 2] << 16)
         | ((uint)row[ROW_COUNT_OFF + 3] << 24);
}

static inline void row_set_count(__global uchar *row, uint c) {
    row[ROW_COUNT_OFF    ] = (uchar)(c        & 0xff);
    row[ROW_COUNT_OFF + 1] = (uchar)((c >> 8) & 0xff);
    row[ROW_COUNT_OFF + 2] = (uchar)((c >> 16) & 0xff);
    row[ROW_COUNT_OFF + 3] = (uchar)((c >> 24) & 0xff);
}

static inline uint row_idx_at(__global const uchar *row, uint i) {
    uint off = ROW_IDX_OFF + i * 4u;
    return (uint)row[off]
         | ((uint)row[off + 1] << 8)
         | ((uint)row[off + 2] << 16)
         | ((uint)row[off + 3] << 24);
}

static inline void row_set_idx(__global uchar *row, uint i, uint v) {
    uint off = ROW_IDX_OFF + i * 4u;
    row[off    ] = (uchar)(v        & 0xff);
    row[off + 1] = (uchar)((v >> 8) & 0xff);
    row[off + 2] = (uchar)((v >> 16) & 0xff);
    row[off + 3] = (uchar)((v >> 24) & 0xff);
}

/* The 16-bit bucket key for the current round (round r consumes hash bytes
 * [r*CBYTES .. r*CBYTES + CBYTES]). For (96,5) cbytes = 2. */
static inline uint row_bucket(__global const uchar *row, uint round) {
    uint off = round * CBYTES;
    return ((uint)row[off] << 8) | (uint)row[off + 1];
}

/* Concatenate index lists in canonical order (min-index subtree first).
 * Matches equihash-core's concat_canonical. */
static void row_concat_indices(
    __global uchar *out_row,
    __global const uchar *a,
    __global const uchar *b,
    uint a_count,
    uint b_count
) {
    uint a0 = row_idx_at(a, 0);
    uint b0 = row_idx_at(b, 0);
    __global const uchar *first  = (a0 < b0) ? a : b;
    __global const uchar *second = (a0 < b0) ? b : a;
    uint first_count  = (a0 < b0) ? a_count : b_count;
    uint second_count = (a0 < b0) ? b_count : a_count;

    for (uint i = 0; i < first_count; ++i)  row_set_idx(out_row, i,                 row_idx_at(first,  i));
    for (uint i = 0; i < second_count; ++i) row_set_idx(out_row, i + first_count, row_idx_at(second, i));

    row_set_count(out_row, first_count + second_count);
}

/* XOR the 12-byte hash of two rows into `out_row`. Round r consumes leading
 * CBYTES, which we then leave as zeros in out_row (so future rounds keep
 * indexing by `round * CBYTES`). */
static void row_xor_hash(
    __global uchar *out_row,
    __global const uchar *a,
    __global const uchar *b
) {
    for (uint i = 0; i < ROW_HASH_BYTES; ++i) {
        out_row[ROW_HASH_OFF + i] = a[ROW_HASH_OFF + i] ^ b[ROW_HASH_OFF + i];
    }
}

/* O(n^2) disjoint-index check (matches equihash-core::distinct_indices). */
static int rows_disjoint(
    __global const uchar *a, uint a_count,
    __global const uchar *b, uint b_count
) {
    for (uint i = 0; i < a_count; ++i) {
        uint ai = row_idx_at(a, i);
        for (uint j = 0; j < b_count; ++j) {
            if (ai == row_idx_at(b, j)) return 0;
        }
    }
    return 1;
}

/* ===================================================================
 * Per-work-item Wagner solve
 * =================================================================== */

/* Encode the 32 solution indices into compressed bytes.
 *
 * Each index uses (cbits + 1) = 17 bits, MSB-first within the byte stream.
 * Total = 32 * 17 = 544 bits = 68 bytes.
 */
#define COMPRESSED_BYTES 68u

static void compress_solution(
    __global const uchar *row,   /* row with SOLN_INDICES = 32 indices */
    __global uchar       *out68
) {
    /* Clear output. */
    for (uint i = 0; i < COMPRESSED_BYTES; ++i) out68[i] = 0;

    uint pos = 0;
    for (uint i = 0; i < SOLN_INDICES; ++i) {
        uint idx = row_idx_at(row, i);
        /* MSB-first emit of 17 bits. */
        for (int b = 16; b >= 0; --b) {
            uint bit = (idx >> b) & 1u;
            uint byte_off = pos >> 3;
            uint shift = 7u - (pos & 7u);
            out68[byte_off] |= (uchar)(bit << shift);
            pos++;
        }
    }
}

/* ===================================================================
 * Main solver kernel.
 *
 * Inputs:
 *   h_base          : 8 ulongs, precomputed BLAKE2b state (IV XOR param)
 *   input113        : 113 bytes (personalization + challenge + miner_pubkey + height + nonce_template)
 *   nonce_offset    : added to gid; final nonce = nonce_offset + gid, written into bytes [105..113]
 *                     of the input copy (first 24 bytes of the 32-byte nonce slot stay as in template).
 *
 * Note on nonce layout: the I-block ends at byte 81. Bytes 81..113 = nonce[0..32].
 * We use the LOW 8 bytes (positions 105..113) as a u64 counter; the rest are
 * set from a random base on the host so workers don't collide.
 *
 * Workspace pointer arrangement per WI:
 *   buf_a:   N_INIT rows of ROW_BYTES → 131072 * 144 = 18,874,368 bytes
 *   buf_b:   same
 *   counts:  BUCKETS * 4 = 262,144 bytes
 *   starts:  same
 *   total ≈ 38 MB per WI
 * =================================================================== */

#define WORKSPACE_PER_WI (2u * N_INIT * ROW_BYTES + 2u * BUCKETS * 4u)

__kernel void equihash_96_5(
    __global const ulong * restrict h_base,    /* [8] precomputed BLAKE2b base state */
    __global const uchar * restrict input113,  /* [113] persn || I-block || nonce template */
    ulong nonce_low_offset,                    /* added to gid → low 8 bytes of nonce */
    __global       uchar * restrict workspace, /* [n_wi * WORKSPACE_PER_WI] */
    __global       uchar * restrict hit_nonces,    /* [MAX_HITS * 32] */
    __global       uchar * restrict hit_solutions, /* [MAX_HITS * COMPRESSED_BYTES] */
    volatile __global uint * restrict hit_count,
    uint max_hits
) {
    const uint gid = (uint)get_global_id(0);

    /* Per-WI workspace slice. */
    __global uchar *ws       = workspace + (ulong)gid * (ulong)WORKSPACE_PER_WI;
    __global uchar *buf_a    = ws;
    __global uchar *buf_b    = ws + (ulong)N_INIT * ROW_BYTES;
    __global uint  *counts   = (__global uint *)(ws + 2UL * (ulong)N_INIT * ROW_BYTES);
    __global uint  *starts   = counts + BUCKETS;

    /* ---- Build the per-WI input (with nonce baked in) ---- */
    uchar input_local[113];
    for (uint i = 0; i < 113; ++i) input_local[i] = input113[i];

    /* Set low 8 bytes of nonce to (nonce_low_offset + gid) LE.
     * Nonce occupies input bytes [81..113]. Low 8 = [105..113]. */
    ulong nonce_low = nonce_low_offset + (ulong)gid;
    for (uint b = 0; b < 8; ++b) {
        input_local[105 + b] = (uchar)((nonce_low >> (8 * b)) & 0xff);
    }

    /* ---- Generate N_INIT leaves into buf_a (one index each) ---- */
    /* 26215 BLAKE2b calls per nonce (= ceil(131072 / 5)). */
    const uint n_batches = (N_INIT + INDICES_PER - 1) / INDICES_PER;
    uchar out60[60];

    for (uint batch = 0; batch < n_batches; ++batch) {
        blake2b_leaf_batch(h_base, input_local, batch, out60);
        const uint base_leaf = batch * INDICES_PER;
        for (uint sub = 0; sub < INDICES_PER; ++sub) {
            const uint leaf_i = base_leaf + sub;
            if (leaf_i >= N_INIT) break;
            __global uchar *row = buf_a + (ulong)leaf_i * ROW_BYTES;
            const uint off = sub * LEAF_BYTES;
            for (uint b = 0; b < LEAF_BYTES; ++b) row[ROW_HASH_OFF + b] = out60[off + b];
            row_set_count(row, 1u);
            row_set_idx(row, 0, leaf_i);
        }
    }

    /* ---- 5 Wagner rounds ---- */
    uint in_count = N_INIT;
    __global uchar *cur = buf_a;
    __global uchar *nxt = buf_b;

    for (uint round = 0; round < K_ROUNDS; ++round) {
        /* (1) Histogram + prefix sum to bucket by first 16 bits of CURRENT hash. */
        for (uint i = 0; i < BUCKETS; ++i) counts[i] = 0u;

        const uint hash_off = round * CBYTES;
        for (uint i = 0; i < in_count; ++i) {
            __global const uchar *row = cur + (ulong)i * ROW_BYTES;
            uint b = ((uint)row[ROW_HASH_OFF + hash_off] << 8)
                   | (uint)row[ROW_HASH_OFF + hash_off + 1];
            counts[b]++;
        }
        uint acc = 0u;
        for (uint i = 0; i < BUCKETS; ++i) {
            starts[i] = acc;
            acc += counts[i];
        }

        /* (2) Scatter: copy rows into bucket-sorted positions in `nxt`. */
        __global uchar *sorted = nxt;
        /* Reuse `counts` as a write-cursor by re-zeroing it. */
        for (uint i = 0; i < BUCKETS; ++i) counts[i] = 0u;
        for (uint i = 0; i < in_count; ++i) {
            __global const uchar *row = cur + (ulong)i * ROW_BYTES;
            uint b = ((uint)row[ROW_HASH_OFF + hash_off] << 8)
                   | (uint)row[ROW_HASH_OFF + hash_off + 1];
            uint dst = starts[b] + counts[b]++;
            __global uchar *out_row = sorted + (ulong)dst * ROW_BYTES;
            for (uint k = 0; k < ROW_BYTES; ++k) out_row[k] = row[k];
        }

        /* Swap roles so we can write the round output into `cur` again. */
        __global uchar *tmp = cur; cur = nxt; nxt = tmp;
        /* Now `cur` holds the bucket-sorted rows; `nxt` is scratch. */

        /* (3) For each bucket, generate all distinct pairs, XOR + concat. */
        uint out_count = 0u;
        for (uint b = 0; b < BUCKETS; ++b) {
            uint b_start = starts[b];
            uint b_size  = counts[b];   /* counts now holds the bucket sizes */
            if (b_size < 2u) continue;
            for (uint ia = 0; ia < b_size; ++ia) {
                __global const uchar *ra = cur + (ulong)(b_start + ia) * ROW_BYTES;
                uint ra_cnt = row_count(ra);
                for (uint ib = ia + 1u; ib < b_size; ++ib) {
                    __global const uchar *rb = cur + (ulong)(b_start + ib) * ROW_BYTES;
                    uint rb_cnt = row_count(rb);
                    if (!rows_disjoint(ra, ra_cnt, rb, rb_cnt)) continue;
                    if (out_count >= N_INIT) goto done_round; /* clamp; in practice ≈ N_INIT */
                    __global uchar *out_row = nxt + (ulong)out_count * ROW_BYTES;
                    row_xor_hash(out_row, ra, rb);
                    row_concat_indices(out_row, ra, rb, ra_cnt, rb_cnt);
                    out_count++;
                }
            }
        }
done_round:
        in_count = out_count;
        __global uchar *tmp2 = cur; cur = nxt; nxt = tmp2;

        if (in_count == 0u) return;   /* dead nonce */
    }

    /* ---- After 5 rounds: filter all-zero-hash candidates of length SOLN_INDICES ---- */
    for (uint i = 0; i < in_count; ++i) {
        __global const uchar *row = cur + (ulong)i * ROW_BYTES;
        if (row_count(row) != SOLN_INDICES) continue;
        /* All 12 hash bytes must be zero after K full rounds (each consumed 2 bytes;
         * but we kept hash padded — check all 12 for safety). */
        int all_zero = 1;
        for (uint b = 0; b < ROW_HASH_BYTES; ++b) {
            if (row[ROW_HASH_OFF + b] != 0u) { all_zero = 0; break; }
        }
        if (!all_zero) continue;

        /* Record the hit. */
        uint slot = atomic_inc(hit_count);
        if (slot >= max_hits) return;

        /* Write nonce[32] (low 8 bytes are LE counter; high 24 from input template). */
        __global uchar *out_nonce = hit_nonces + (ulong)slot * 32u;
        /* High 24 from input_local[81..105]; low 8 from nonce_low. */
        for (uint k = 0; k < 24u; ++k) out_nonce[k] = input_local[81u + k];
        for (uint k = 0; k < 8u;  ++k) out_nonce[24u + k] = (uchar)((nonce_low >> (8u * k)) & 0xff);

        __global uchar *out_soln = hit_solutions + (ulong)slot * COMPRESSED_BYTES;
        compress_solution(row, out_soln);

        return;
    }
}
