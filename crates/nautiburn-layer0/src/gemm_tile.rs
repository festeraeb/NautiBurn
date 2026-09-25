//! Forced CubeCL FP16 GEMM tile for Pascal sm_60.
//!
//! Stock CubeCL plane/CMMA asks for 69632 bytes of shared memory. The P100
//! block cap is 49152, and that selector does not read `max_shared_memory_size`.
//! This kernel never asks autotune. It launches BM=BN=16, BK=32 with two FP16
//! stages: `4 * BK * (BM + BN)` = 4096 bytes.

use cubecl::prelude::*;
use half::f16;
use nautiburn_engine::{
    admit_fp16_on_p100, FORCED_TILE_BK, FORCED_TILE_BM, FORCED_TILE_BN, FORCED_TILE_SHARED_BYTES,
    P100_SHARED_CAP_BYTES,
};

const THREADS: u32 = 16;
const BM: u32 = 16;
const BN: u32 = 16;
const BK: u32 = 32;
/// One stage of A (BM×BK) or B (BK×BN), in f16 elems.
const STAGE_A: usize = 512;
const STAGE_B: usize = 512;

pub const TILE_SHARED_BYTES: u32 = FORCED_TILE_SHARED_BYTES;

pub fn forced_tile_shared_bytes() -> u32 {
    // Two stages, two panels, two bytes per f16.
    2 * 2 * (STAGE_A as u32 + STAGE_B as u32)
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

    let mut acc = f32::new(0.0);
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
                f16::new(0.0)
            };

            let b_row = idx / BN;
            let b_col = idx % BN;
            let r_row = k0 + b_row;
            let r_col = CUBE_POS_X * BN + b_col;
            let bval = if r_row < k && r_col < n {
                rhs[(r_row * n + r_col) as usize]
            } else {
                f16::new(0.0)
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

pub fn assert_tile_fits_p100() {
    let shared = forced_tile_shared_bytes();
    assert_eq!(shared, TILE_SHARED_BYTES);
    match admit_fp16_on_p100(shared) {
        nautiburn_engine::Fp16Launch::Admitted => {}
        nautiburn_engine::Fp16Launch::Refused {
            shared_bytes,
            cap_bytes,
        } => {
            panic!("forced tile {shared_bytes} exceeds P100 cap {cap_bytes}");
        }
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
    println!(
        "gemm_f16 tile=BM{BM}_BN{BN}_BK{BK} shared_bytes={shared} cap={P100_SHARED_CAP_BYTES} dtype=f16 shape={m}x{k}x{n}"
    );

    let lhs_h = client.create_from_slice(f16::as_bytes(lhs));
    let rhs_h = client.create_from_slice(f16::as_bytes(rhs));
    let out_h = client.empty(m * n * core::mem::size_of::<f16>());

    let grid_x = (n as u32 + FORCED_TILE_BN - 1) / FORCED_TILE_BN;
    let grid_y = (m as u32 + FORCED_TILE_BM - 1) / FORCED_TILE_BM;

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
