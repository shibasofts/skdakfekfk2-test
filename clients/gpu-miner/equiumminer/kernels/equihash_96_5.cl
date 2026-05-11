/*
 * Equihash (96, 5) OpenCL kernel — v2: one nonce per work-group.
 *
 * 256 work-items in a WG cooperate on a single nonce. Parallelizes:
 *  - BLAKE2b leaf generation (each WI hashes ~103 batches → ~512 leaves)
 *  - Histogram (atomic_inc on global per-WG bucket counts)
 *  - Prefix sum (single WI scans 65536 buckets serially — fast enough)
 *  - Scatter (each WI moves ~512 rows)
 *  - Wagner pair-finding (each WI processes ~256 buckets)
 *  - Final all-zero filter
 *
 * Hardcoded params (Equium): N_INIT=131072, CBITS=16, INDICES_PER=5, BLAKE2B_OUT=60.
 * ROW_BYTES = 12 hash + 4 count + 128 indices (32 max) = 144.
 *
 * Workspace per WG = 2*N_INIT*144 + 2*BUCKETS*4 ≈ 38 MB.
 * On 16 GB: ~430 concurrent WGs. On 32 GB: ~870.
 */

#define N_INIT          131072u
#define CBITS           16u
#define CBYTES          2u
#define LEAF_BYTES      12u
#define INDICES_PER     5u
#define BLAKE2B_OUT     60u
#define SOLN_INDICES    32u
#define K_ROUNDS        5u
#define BUCKETS         65536u
#define ROW_HASH_BYTES  12u
#define ROW_BYTES       144u
#define ROW_HASH_OFF    0u
#define ROW_COUNT_OFF   12u
#define ROW_IDX_OFF     16u
#define COMPRESSED_BYTES 68u

#define WORKSPACE_PER_WG (2u * N_INIT * ROW_BYTES + 2u * BUCKETS * 4u)

/* ===================================================================
 * BLAKE2b — single-block compression. Used 26215× per nonce per WG.
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

static inline ulong load_u64_le(const uchar *p) {
    ulong x = 0;
    for (int i = 7; i >= 0; --i) x = (x << 8) | (ulong)p[i];
    return x;
}

static void blake2b_leaf_batch(
    const ulong h_base[8],
    const uchar *input113,
    uint batch_idx,
    uchar out60[60]
) {
    uchar block[128];
    for (uint i = 0; i < 113; ++i) block[i] = input113[i];
    block[113] = (uchar)(batch_idx       & 0xff);
    block[114] = (uchar)((batch_idx >> 8) & 0xff);
    block[115] = (uchar)((batch_idx >> 16) & 0xff);
    block[116] = (uchar)((batch_idx >> 24) & 0xff);
    for (uint i = 117; i < 128; ++i) block[i] = 0;

    ulong m[16];
    for (uint i = 0; i < 16; ++i) m[i] = load_u64_le(&block[i * 8]);

    ulong h[8];
    for (int i = 0; i < 8; ++i) h[i] = h_base[i];
    blake2b_compress(h, m, 117UL, 1);

    for (uint w = 0; w < 8; ++w) {
        ulong x = h[w];
        for (uint b = 0; b < 8; ++b) {
            uint off = w * 8 + b;
            if (off < 60) out60[off] = (uchar)((x >> (8 * b)) & 0xff);
        }
    }
}

/* ===================================================================
 * Row helpers (global memory)
 * =================================================================== */

static inline uint row_count(__global const uchar *row) {
    __global const uint *p = (__global const uint *)(row + ROW_COUNT_OFF);
    return p[0];
}

static inline void row_set_count(__global uchar *row, uint c) {
    __global uint *p = (__global uint *)(row + ROW_COUNT_OFF);
    p[0] = c;
}

static inline uint row_idx_at(__global const uchar *row, uint i) {
    __global const uint *p = (__global const uint *)(row + ROW_IDX_OFF);
    return p[i];
}

static inline void row_set_idx(__global uchar *row, uint i, uint v) {
    __global uint *p = (__global uint *)(row + ROW_IDX_OFF);
    p[i] = v;
}

static inline uint row_bucket16(__global const uchar *row, uint hash_off) {
    return ((uint)row[ROW_HASH_OFF + hash_off] << 8)
         | (uint)row[ROW_HASH_OFF + hash_off + 1];
}

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

    for (uint i = 0; i < first_count; ++i)  row_set_idx(out_row, i,                  row_idx_at(first,  i));
    for (uint i = 0; i < second_count; ++i) row_set_idx(out_row, i + first_count, row_idx_at(second, i));

    row_set_count(out_row, first_count + second_count);
}

/* Vectorized 144-byte row copy: 9 × ulong2. ROW_BYTES = 144 = 9 * 16. */
static inline void row_copy(__global uchar *dst, __global const uchar *src) {
    __global ulong2 *d = (__global ulong2 *)dst;
    __global const ulong2 *s = (__global const ulong2 *)src;
    d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
    d[3] = s[3]; d[4] = s[4]; d[5] = s[5];
    d[6] = s[6]; d[7] = s[7]; d[8] = s[8];
}

static void row_xor_hash(
    __global uchar *out_row,
    __global const uchar *a,
    __global const uchar *b
) {
    /* 12 bytes — do it as one ulong (8) + one uint (4). */
    __global ulong *out_u = (__global ulong *)(out_row + ROW_HASH_OFF);
    __global const ulong *a_u = (__global const ulong *)(a + ROW_HASH_OFF);
    __global const ulong *b_u = (__global const ulong *)(b + ROW_HASH_OFF);
    out_u[0] = a_u[0] ^ b_u[0];
    __global uint *out_v = (__global uint *)(out_row + ROW_HASH_OFF + 8);
    __global const uint *a_v = (__global const uint *)(a + ROW_HASH_OFF + 8);
    __global const uint *b_v = (__global const uint *)(b + ROW_HASH_OFF + 8);
    out_v[0] = a_v[0] ^ b_v[0];
}

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

static void compress_solution(
    __global const uchar *row,
    __global uchar       *out68
) {
    for (uint i = 0; i < COMPRESSED_BYTES; ++i) out68[i] = 0;
    uint pos = 0;
    for (uint i = 0; i < SOLN_INDICES; ++i) {
        uint idx = row_idx_at(row, i);
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
 * MAIN KERNEL — one nonce per work-group.
 *
 * Launch geometry:
 *   global_size = n_wgs * local_size
 *   local_size  = 256 (or 64–512; must divide BUCKETS evenly for cleanliness)
 *
 * One WG = one nonce. WI ids within a WG cooperate via barriers.
 * =================================================================== */

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
void equihash_96_5(
    __global const ulong * restrict h_base,
    __global const uchar * restrict input113,
    ulong nonce_low_offset,
    __global       uchar * restrict workspace,
    __global       uchar * restrict hit_nonces,
    __global       uchar * restrict hit_solutions,
    volatile __global uint * restrict hit_count,
    uint max_hits
) {
    const uint wgid = (uint)get_group_id(0);
    const uint lid  = (uint)get_local_id(0);
    const uint lsz  = (uint)get_local_size(0);

    __global uchar *ws       = workspace + (ulong)wgid * (ulong)WORKSPACE_PER_WG;
    __global uchar *buf_a    = ws;
    __global uchar *buf_b    = ws + (ulong)N_INIT * ROW_BYTES;
    __global uint  *counts   = (__global uint *)(ws + 2UL * (ulong)N_INIT * ROW_BYTES);
    __global uint  *starts   = counts + BUCKETS;

    /* Shared in_count for this round, plus a flag for "row capacity exhausted". */
    __local uint shared_in_count;
    __local uint shared_overflow;

    /* Build per-WG input. Every WI computes it (cheap, avoids a barrier).
     * The nonce is shared by all WIs in the WG (one nonce per WG). */
    uchar input_local[113];
    for (uint i = 0; i < 113; ++i) input_local[i] = input113[i];
    ulong nonce_low = nonce_low_offset + (ulong)wgid;
    for (uint b = 0; b < 8; ++b) {
        input_local[105 + b] = (uchar)((nonce_low >> (8 * b)) & 0xff);
    }

    ulong h_base_priv[8];
    for (uint i = 0; i < 8; ++i) h_base_priv[i] = h_base[i];

    /* ---- 1. Parallel leaf generation ----
     * n_batches = ceil(N_INIT / INDICES_PER) = ceil(131072/5) = 26215
     * Each WI does (26215 + lsz - 1) / lsz batches. */
    const uint n_batches = (N_INIT + INDICES_PER - 1) / INDICES_PER;
    uchar out60[60];
    for (uint batch = lid; batch < n_batches; batch += lsz) {
        blake2b_leaf_batch(h_base_priv, input_local, batch, out60);
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
    barrier(CLK_GLOBAL_MEM_FENCE);

    /* ---- 2. Five Wagner rounds ---- */
    if (lid == 0) shared_in_count = N_INIT;
    barrier(CLK_LOCAL_MEM_FENCE);

    __global uchar *cur = buf_a;
    __global uchar *nxt = buf_b;

    for (uint round = 0; round < K_ROUNDS; ++round) {
        uint in_count = shared_in_count;
        const uint hash_off = round * CBYTES;

        /* Zero counts in parallel. */
        for (uint i = lid; i < BUCKETS; i += lsz) counts[i] = 0u;
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Histogram via global atomic_inc. */
        for (uint i = lid; i < in_count; i += lsz) {
            __global const uchar *row = cur + (ulong)i * ROW_BYTES;
            uint b = row_bucket16(row, hash_off);
            atomic_inc(&counts[b]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Prefix sum — serial on WI 0. 65536 entries × ~2 cycles = ~130k cycles
         * = ~50μs. Negligible vs Wagner cost. */
        if (lid == 0) {
            uint acc = 0u;
            for (uint i = 0; i < BUCKETS; ++i) {
                starts[i] = acc;
                acc += counts[i];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Re-zero counts to use as scatter cursor. */
        for (uint i = lid; i < BUCKETS; i += lsz) counts[i] = 0u;
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Scatter in parallel (vectorized 144-byte row copy). */
        for (uint i = lid; i < in_count; i += lsz) {
            __global const uchar *row = cur + (ulong)i * ROW_BYTES;
            uint b = row_bucket16(row, hash_off);
            uint slot = atomic_inc(&counts[b]);
            uint dst = starts[b] + slot;
            __global uchar *out_row = nxt + (ulong)dst * ROW_BYTES;
            row_copy(out_row, row);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Swap: cur now holds sorted rows, nxt is scratch for round output. */
        { __global uchar *t = cur; cur = nxt; nxt = t; }

        /* Reset out cursor for pair-finding. */
        if (lid == 0) { shared_in_count = 0u; shared_overflow = 0u; }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Pair-finding in parallel — each WI handles a slice of buckets. */
        for (uint b = lid; b < BUCKETS; b += lsz) {
            uint b_start = starts[b];
            uint b_size  = counts[b];
            if (b_size < 2u) continue;
            for (uint ia = 0u; ia < b_size; ++ia) {
                __global const uchar *ra = cur + (ulong)(b_start + ia) * ROW_BYTES;
                uint ra_cnt = row_count(ra);
                for (uint ib = ia + 1u; ib < b_size; ++ib) {
                    __global const uchar *rb = cur + (ulong)(b_start + ib) * ROW_BYTES;
                    uint rb_cnt = row_count(rb);
                    if (!rows_disjoint(ra, ra_cnt, rb, rb_cnt)) continue;
                    uint slot = atomic_inc(&shared_in_count);
                    if (slot >= N_INIT) { shared_overflow = 1u; continue; }
                    __global uchar *out_row = nxt + (ulong)slot * ROW_BYTES;
                    row_xor_hash(out_row, ra, rb);
                    row_concat_indices(out_row, ra, rb, ra_cnt, rb_cnt);
                }
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        /* Clamp in_count to capacity if we overflowed (excess pairs already
         * silently dropped via the slot check above). */
        if (lid == 0 && shared_overflow != 0u) shared_in_count = N_INIT;
        barrier(CLK_LOCAL_MEM_FENCE);

        { __global uchar *t = cur; cur = nxt; nxt = t; }
        /* in_count for next round is whatever we accumulated. */
    }

    /* ---- 3. Final filter ----
     * Valid rows have count == 32 and all 12 hash bytes zero. */
    uint final_count = shared_in_count;
    for (uint i = lid; i < final_count; i += lsz) {
        __global const uchar *row = cur + (ulong)i * ROW_BYTES;
        if (row_count(row) != SOLN_INDICES) continue;
        int all_zero = 1;
        for (uint b = 0; b < ROW_HASH_BYTES; ++b) {
            if (row[ROW_HASH_OFF + b] != 0u) { all_zero = 0; break; }
        }
        if (!all_zero) continue;

        uint slot = atomic_inc(hit_count);
        if (slot >= max_hits) return;

        __global uchar *out_nonce = hit_nonces + (ulong)slot * 32u;
        for (uint k = 0; k < 24u; ++k) out_nonce[k] = input_local[81u + k];
        for (uint k = 0; k < 8u;  ++k) out_nonce[24u + k] = (uchar)((nonce_low >> (8u * k)) & 0xff);

        __global uchar *out_soln = hit_solutions + (ulong)slot * COMPRESSED_BYTES;
        compress_solution(row, out_soln);
        return;
    }
}
