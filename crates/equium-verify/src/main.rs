//! `equium-verify`: a one-shot Equihash CPU verifier driven by stdin/stdout.
//!
//! Used by the OpenCL miner's `verify.py` to CPU-check each GPU candidate
//! against the exact same verifier the chain uses — no chance of solver
//! disagreement causing on-chain reverts.
//!
//! Input (stdin, binary, all little-endian unless noted):
//!   [4 bytes]  n  (u32)
//!   [4 bytes]  k  (u32)
//!   [81 bytes] input block I  (persn + challenge + miner_pubkey + height_le)
//!   [32 bytes] nonce
//!   [32 bytes] target  (big-endian)
//!   [4 bytes]  soln_len  (u32)
//!   [soln_len] soln_indices
//!
//! Output (stdout, ascii line):
//!   "ok <hex32-solution-hash>"   — Equihash valid AND hash < target
//!   "bad_equihash"               — puzzle malformed
//!   "above_target <hex32-hash>"  — puzzle valid but hash >= target
//!   "bad_input"                  — input failed to parse
//!
//! Exit status is 0 on "ok", non-zero otherwise. Callers should check both.

use std::io::{Read, Write};

use equihash_core::challenge::I_LEN;
use equihash_core::verify::{verify, VerifyError};

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<()> {
    r.read_exact(buf)
}

fn read_u32_le<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn main() {
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();

    let result: Result<(), &'static str> = (|| {
        let n = read_u32_le(&mut handle).map_err(|_| "n")?;
        let k = read_u32_le(&mut handle).map_err(|_| "k")?;
        let mut input = [0u8; I_LEN];
        read_exact(&mut handle, &mut input).map_err(|_| "input")?;
        let mut nonce = [0u8; 32];
        read_exact(&mut handle, &mut nonce).map_err(|_| "nonce")?;
        let mut target = [0u8; 32];
        read_exact(&mut handle, &mut target).map_err(|_| "target")?;
        let soln_len = read_u32_le(&mut handle).map_err(|_| "soln_len")? as usize;
        if soln_len > 4096 {
            return Err("soln_len");
        }
        let mut soln = vec![0u8; soln_len];
        read_exact(&mut handle, &mut soln).map_err(|_| "soln")?;

        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        match verify(n, k, &input, &nonce, &soln, &target) {
            Ok(hash) => {
                writeln!(out, "ok {}", hex(&hash)).ok();
                std::process::exit(0);
            }
            Err(VerifyError::InvalidEquihash) => {
                writeln!(out, "bad_equihash").ok();
                std::process::exit(2);
            }
            Err(VerifyError::AboveTarget) => {
                // We still want the caller to know the hash; recompute it.
                let h = equihash_core::challenge::solution_hash(&soln, &input);
                writeln!(out, "above_target {}", hex(&h)).ok();
                std::process::exit(3);
            }
        }
    })();

    if let Err(_what) = result {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "bad_input").ok();
        std::process::exit(1);
    }
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}
