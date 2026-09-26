//! Widen the `dump` f16 expert blobs into the f32 files `compare` reads.
//!
//! `nautiburn-layer0 dump` writes one f16 file per expert per projection:
//! `e{0..63}_{gate,up,down}.f16`, each `N_EMBD * N_FF` values. `compare`
//! instead reads three single f32 blobs holding every expert back to back.
//!
//! This is a pure f16 -> f32 widening. It deliberately does **not** transpose:
//! `compare` and `cpu_expert` both call their own `transpose()` with the
//! shape they expect, so a transpose here would be applied twice and silently
//! produce a plausible-looking but wrong number.
//!
//! Layout, per expert, as read out of the GGUF:
//!
//! ```text
//! gate  [N_EMBD, N_FF]   -> compare reads as [N_FF, N_EMBD]
//! up    [N_EMBD, N_FF]   -> compare reads as [N_FF, N_EMBD]
//! down  [N_FF,  N_EMBD]  -> compare reads as [N_EMBD, N_FF]
//! ```
//!
//! All three hold the same element count, so the only thing that can go wrong
//! is a wrong order, and only a real `compare` run can catch that.

use std::io::Write;
use std::path::{Path, PathBuf};

use half::f16;

/// Model geometry. Must match `nautiburn-layer0/src/scrape_run.rs`.
const N_EMBD: usize = 2048;
const N_FF: usize = 1024;
const N_EXPERT: usize = 64;

/// Values per expert per projection.
const PER_EXPERT: usize = N_EMBD * N_FF;

/// Filename stem for each projection, paired with the blob `compare` reads.
const KINDS: [(&str, &str); 3] = [
    ("gate", "ffn_gate_exps.f32"),
    ("up", "ffn_up_exps.f32"),
    ("down", "ffn_down_exps.f32"),
];

/// Read one f16 file and widen it to f32. Fails on any size mismatch rather
/// than padding, because a short file would shift every later expert.
fn read_f16_widen(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != PER_EXPERT * 2 {
        return Err(format!(
            "{}: {} bytes, expected {} (PER_EXPERT={PER_EXPERT})",
            path.display(),
            bytes.len(),
            PER_EXPERT * 2
        ));
    }
    let mut out = Vec::with_capacity(PER_EXPERT);
    for c in bytes.chunks_exact(2) {
        // GGUF stores f16 little-endian.
        out.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
    }
    Ok(out)
}

/// Write a little-endian f32 blob. `compare` reads it with `f32::from_le_bytes`.
fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);
    // 16 MiB buffer: these blobs run to hundreds of MiB.
    let mut buf = Vec::with_capacity(1 << 24);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
        if buf.len() >= (1 << 24) {
            w.write_all(&buf).map_err(|e| format!("write: {e}"))?;
            buf.clear();
        }
    }
    w.write_all(&buf).map_err(|e| format!("write: {e}"))?;
    w.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

/// Convert `dump_dir`'s 192 f16 files into the three f32 blobs in `out_dir`.
///
/// Also copies `router.f32` to `ffn_gate_inp.f32`: `dump` writes the router
/// under the name `router.f32`, but `compare` reads `ffn_gate_inp.f32`. The two
/// names drifted apart, so the rename happens here rather than in `dump` (which
/// is committed and shared with the other agent).
pub fn convert(dump_dir: &Path, out_dir: &Path) -> Result<Vec<(String, usize)>, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir {}: {e}", out_dir.display()))?;

    let mut written = Vec::new();
    for (kind, blob) in KINDS {
        let mut all: Vec<f32> = Vec::with_capacity(PER_EXPERT * N_EXPERT);
        for e in 0..N_EXPERT {
            let src: PathBuf = dump_dir.join(format!("e{e}_{kind}.f16"));
            all.extend_from_slice(&read_f16_widen(&src)?);
        }
        // Guard the total, so a future expert-count change fails loudly.
        let want = PER_EXPERT * N_EXPERT;
        if all.len() != want {
            return Err(format!("{kind}: built {} values, expected {want}", all.len()));
        }
        let dst = out_dir.join(blob);
        write_f32(&dst, &all)?;
        written.push((blob.to_string(), all.len()));
    }

    // Router under the name `compare` expects.
    let router_src = dump_dir.join("router.f32");
    let router_dst = out_dir.join("ffn_gate_inp.f32");
    if router_src.exists() {
        let bytes = std::fs::read(&router_src)
            .map_err(|e| format!("read {}: {e}", router_src.display()))?;
        let want = N_EXPERT * N_EMBD * 4;
        if bytes.len() != want {
            return Err(format!("router.f32: {} bytes, expected {want}", bytes.len()));
        }
        std::fs::write(&router_dst, &bytes)
            .map_err(|e| format!("write {}: {e}", router_dst.display()))?;
        written.push(("ffn_gate_inp.f32".to_string(), N_EXPERT * N_EMBD));
    }

    Ok(written)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dump_dir, out_dir) = match args.as_slice() {
        [a, b] => (Path::new(a), Path::new(b)),
        [a] => (Path::new(a), Path::new(a)),
        _ => {
            eprintln!("usage: nautiburn-oracle-convert <dump_dir> [out_dir]");
            eprintln!("  converts e{{0..63}}_{{gate,up,down}}.f16 -> ffn_*_exps.f32");
            std::process::exit(2);
        }
    };

    match convert(dump_dir, out_dir) {
        Ok(rows) => {
            for (name, n) in rows {
                println!("wrote {name} values={n}");
            }
            println!(
                "done out={} next: nautiburn-fp16-scrape compare {} f16",
                out_dir.display(),
                out_dir.display()
            );
        }
        Err(e) => {
            eprintln!("convert failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("nautiburn-convert-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("tmpdir");
        d
    }

    fn write_f16(path: &Path, vals: &[f16]) {
        let mut b = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, b).expect("write f16");
    }

    #[test]
    fn constants_match_compare() {
        assert_eq!(PER_EXPERT, N_EMBD * N_FF);
        assert_eq!(PER_EXPERT * 2, 4_194_304, "one f16 expert is 4 MiB");
        assert_eq!(N_EXPERT * N_EMBD * 4, 524_288, "router is 512 KiB");
        // Every projection blob holds the same element count, which is why a
        // wrong transpose cannot be caught by a size check.
        assert_eq!(PER_EXPERT * N_EXPERT, 134_217_728);
    }

    #[test]
    fn widens_f16_to_f32() {
        let dir = tmpdir("widen");
        let src = dir.join("t.f16");

        // Known values, checked at the head where indices are readable.
        let mut vals: Vec<f16> = (0..PER_EXPERT).map(|_| f16::from_f32(0.5)).collect();
        vals[0] = f16::from_f32(1.0);
        vals[1] = f16::from_f32(-2.0);
        write_f16(&src, &vals);

        let got = read_f16_widen(&src).expect("read");
        assert_eq!(got.len(), PER_EXPERT);
        assert_eq!(got[0], 1.0);
        assert_eq!(got[1], -2.0);
        assert!((got[5] - 0.5).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_sized_expert_file() {
        let dir = tmpdir("short");
        let src = dir.join("short.f16");
        write_f16(&src, &[f16::from_f32(1.0); 16]);
        // A short file would shift every later expert, so it must not be padded.
        let err = read_f16_widen(&src).expect_err("must reject");
        assert!(err.contains("expected"), "unhelpful error: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_expert_file_names_the_path() {
        let dir = tmpdir("missing");
        let err = read_f16_widen(&dir.join("e7_gate.f16")).expect_err("must fail");
        assert!(err.contains("e7_gate.f16"), "error should name the file: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

