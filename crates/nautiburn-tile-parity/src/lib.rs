//! Standalone FP16 GEMM tile parity harness.
//!
//! This crate copies the forced CubeCL FP16 tile from `nautiburn-layer0` and
//! runs the same kernel on CUDA and WGPU, checking both against a CPU FP32
//! reference.

use cubecl::prelude::*;
use half::f16;

const THREADS: u32 = 16;
const BM: u32 = 16;
const BN: u32 = 16;
const BK: u32 = 32;

/// One stage of A (BM×BK) or B (BK×BN), in f16 elems.
const STAGE_A: usize = 512;
const STAGE_B: usize = 512;

pub const TILE_SHARED_BYTES: u32 = 4096;
pub const P100_SHARED_CAP_BYTES: u32 = 49152;

pub fn forced_tile_shared_bytes() -> u32 {
    // Two stages, two panels, two bytes per f16.
    2 * 2 * (STAGE_A as u32 + STAGE_B as u32)
}

pub fn assert_tile_fits_p100() {
    let shared = forced_tile_shared_bytes();
    assert_eq!(shared, TILE_SHARED_BYTES);
    assert!(shared <= P100_SHARED_CAP_BYTES);
}

#[cube(launch_unchecked)]
fn gemm_f16_forced_tile(
    lhs: &[f16],
    rhs: &[f16],
    out: &mut [f16],
    m: u32,
    n: u32,
    k: u32,
) {
    let row = CUBE_POS_Y * BM + UNIT_POS_Y;
    let col = CUBE_POS_X * BN + UNIT_POS_X;
    let tid = UNIT_POS_Y * THREADS + UNIT_POS_X;

    let mut a0 = Shared::<[f16]>::new_slice(STAGE_A);
    let mut b0 = Shared::<[f16]>::new_slice(STAGE_B);
    let mut a1 = Shared::<[f16]>::new_slice(STAGE_A);
    let mut b1 = Shared::<[f16]>::new_slice(STAGE_B);

    let mut acc = f32::new(0.0f32);
    let k_tiles = (k + BK - 1) / BK;

    for kt in 0..k_tiles {
        let k0 = kt * BK;
        let even = kt % 2 == 0;

        // 256 threads × 2 elems = 512 = BM*BK = BK*BN.
        for s in 0..2 {
            let idx = tid * 2 + s;
            let a_row = idx / BK;
            let a_col = idx % BK;
            let g_row = CUBE_POS_Y * BM + a_row;
            let g_col = k0 + a_col;
            let aval = if g_row < m && g_col < k {
                lhs[(g_row * k + g_col) as usize]
            } else {
                f16::new(0.0f32)
            };

            let b_row = idx / BN;
            let b_col = idx % BN;
            let r_row = k0 + b_row;
            let r_col = CUBE_POS_X * BN + b_col;
            let bval = if r_row < k && r_col < n {
                rhs[(r_row * n + r_col) as usize]
            } else {
                f16::new(0.0f32)
            };

            if even {
                a0[idx as usize] = aval;
                b0[idx as usize] = bval;
            } else {
                a1[idx as usize] = aval;
                b1[idx as usize] = bval;
            }
        }
        sync_cube();

        for kk in 0..BK {
            let a = if even {
                a0[(UNIT_POS_Y * BK + kk) as usize]
            } else {
                a1[(UNIT_POS_Y * BK + kk) as usize]
            };
            let b = if even {
                b0[(kk * BN + UNIT_POS_X) as usize]
            } else {
                b1[(kk * BN + UNIT_POS_X) as usize]
            };
            acc = acc + f32::cast_from(a) * f32::cast_from(b);
        }
        sync_cube();
    }

    if row < m && col < n {
        out[(row * n + col) as usize] = f16::cast_from(acc);
    }
}

/// Launch the forced-tile FP16 GEMM: `out[m,n] = lhs[m,k] @ rhs[k,n]`.
/// Fails loudly if the tile is above the P100 shared-memory cap. Never demotes to FP32.
pub fn gemm_f16(
    client: &cubecl::client::Client,
    lhs: &[f16],
    rhs: &[f16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f16>, String> {
    assert_tile_fits_p100();
    if lhs.len() != m * k {
        return Err(format!("lhs len {} != {m}*{k}", lhs.len()));
    }
    if rhs.len() != k * n {
        return Err(format!("rhs len {} != {k}*{n}", rhs.len()));
    }
    if m == 0 || n == 0 || k == 0 {
        return Err("empty gemm".to_string());
    }

    let shared = forced_tile_shared_bytes();
    if std::env::var_os("NAUTIBURN_GEMM_LOG").is_some() {
        println!(
            "gemm_f16 tile=BM{BM}_BN{BN}_BK{BK} shared_bytes={shared} cap={P100_SHARED_CAP_BYTES} dtype=f16 shape={m}x{k}x{n}"
        );
    }

    let lhs_h = client.create_from_slice(f16::as_bytes(lhs));
    let rhs_h = client.create_from_slice(f16::as_bytes(rhs));
    let out_h = client.empty(m * n * core::mem::size_of::<f16>());

    let grid_x = (n as u32 + BN - 1) / BN;
    let grid_y = (m as u32 + BM - 1) / BM;

    unsafe {
        gemm_f16_forced_tile::launch_unchecked(
            client,
            CubeCount::Static(grid_x, grid_y, 1),
            CubeDim::new_2d(THREADS, THREADS),
            BufferArg::from_raw_parts(lhs_h, m * k),
            BufferArg::from_raw_parts(rhs_h, k * n),
            BufferArg::from_raw_parts(out_h.clone(), m * n),
            m as u32,
            n as u32,
            k as u32,
        );
    }

    cubecl::future::block_on(client.sync()).map_err(|err| format!("cubecl sync: {err:?}"))?;

    let bytes = client
        .read_one(out_h)
        .map_err(|err| format!("cubecl read: {err:?}"))?;
    Ok(f16::from_bytes(&bytes).to_vec())
}

/// Host FP32 reference for the same layout. Used to check the tile, not to serve.
pub fn gemm_cpu_f32(lhs: &[f16], rhs: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a = lhs[i * k + p].to_f32();
            for j in 0..n {
                out[i * n + j] += a * rhs[p * n + j].to_f32();
            }
        }
    }
    out
}

pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn next_f16(&mut self, scale: f32) -> f16 {
        let raw = self.next_u64();
        let frac = (raw as f64 / u64::MAX as f64) * 2.0 - 1.0;
        f16::from_f32(frac as f32 * scale)
    }
}

pub fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f16> {
    let mut rng = XorShift::new(seed);
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        out.push(rng.next_f16(1.0));
    }
    out
}

pub struct Parity {
    pub max_abs_err: f32,
    pub mean_abs_err: f32,
    pub max_rel_err: f32,
    pub n_bad: usize,
}

pub fn compare(got: &[f16], want: &[f32], atol: f32, rtol: f32) -> Parity {
    let mut max_abs_err = 0.0f32;
    let mut sum_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut n_bad = 0usize;

    for i in 0..want.len() {
        let g = got.get(i).map(|v| v.to_f32()).unwrap_or(0.0);
        let w = want[i];
        let abs = (g - w).abs();

        max_abs_err = max_abs_err.max(abs);
        sum_abs_err += abs;

        let denom = w.abs();
        let rel = if denom > 0.0 {
            abs / denom
        } else if abs > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };
        max_rel_err = max_rel_err.max(rel);

        if abs > atol + rtol * w.abs() {
            n_bad += 1;
        }
    }

    Parity {
        max_abs_err,
        mean_abs_err: if want.is_empty() {
            0.0
        } else {
            sum_abs_err / want.len() as f32
        },
        max_rel_err,
        n_bad,
    }
}

pub enum Backend {
    Cuda(usize),
    Wgpu(String),
}

pub fn open(backend: &Backend) -> Result<cubecl::client::Client, String> {
    match backend {
        Backend::Cuda(i) => {
            let device = cubecl::Device::cuda(*i).map_err(|e| format!("{e:?}"))?;
            Ok(device.client())
        }
        Backend::Wgpu(kind) => {
            let kind = parse_wgpu_kind(kind)?;
            let device = cubecl::Device::wgpu(kind).map_err(|e| format!("{e:?}"))?;
            Ok(device.client())
        }
    }
}

fn parse_wgpu_kind(kind: &str) -> Result<cubecl::device::WgpuDeviceKind, String> {
    use cubecl::device::WgpuDeviceKind;

    match kind {
        "default" => Ok(WgpuDeviceKind::DefaultDevice),
        "cpu" => Ok(WgpuDeviceKind::Cpu),
        s if s.starts_with("discrete:") => {
            let idx = s
                .strip_prefix("discrete:")
                .ok_or_else(|| "missing discrete index".to_string())?
                .parse::<usize>()
                .map_err(|e| format!("invalid discrete index: {e}"))?;
            Ok(WgpuDeviceKind::DiscreteGpu(idx))
        }
        s => Err(format!("unknown wgpu kind: {s}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_shared_bytes_match_p100_cap() {
        assert_eq!(2 * 2 * (STAGE_A + STAGE_B), TILE_SHARED_BYTES as usize);
        assert!(TILE_SHARED_BYTES <= P100_SHARED_CAP_BYTES);
    }

    #[test]
    fn xorshift_is_deterministic() {
        let mut a = XorShift::new(1234);
        let mut b = XorShift::new(1234);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn cpu_gemm_hand_example() {
        let lhs = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        let rhs = vec![
            f16::from_f32(7.0),
            f16::from_f32(8.0),
            f16::from_f32(9.0),
            f16::from_f32(10.0),
            f16::from_f32(11.0),
            f16::from_f32(12.0),
        ];

        let out = gemm_cpu_f32(&lhs, &rhs, 2, 2, 3);
        assert_eq!(out, vec![50.0, 64.0, 115.0, 150.0]);
    }

    #[test]
    fn compare_counts_bad_elements() {
        let got = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let want = vec![1.0, 3.0];
        let p = compare(&got, &want, 0.1, 0.0);
        assert_eq!(p.n_bad, 1);
        assert_eq!(p.max_abs_err, 1.0);
        assert_eq!(p.mean_abs_err, 0.5);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_cpu_parity_matches_reference() {
        let client = open(&Backend::Wgpu("cpu".into())).expect("wgpu cpu backend");

        for (m, n, k) in [(16usize, 16usize, 32usize), (17, 33, 65), (64, 256, 128)] {
            let lhs = random_matrix(m, k, 7);
            let rhs = random_matrix(k, n, 11);
            let got = gemm_f16(&client, &lhs, &rhs, m, n, k).expect("wgpu gemm");
            let want = gemm_cpu_f32(&lhs, &rhs, m, n, k);
            let p = compare(&got, &want, 0.05, 0.02);

            assert_eq!(
                p.n_bad,
                0,
                "shape {m}x{n}x{k}: max_abs={} mean_abs={} max_rel={}",
                p.max_abs_err,
                p.mean_abs_err,
                p.max_rel_err
            );
        }
    }
}
