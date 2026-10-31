//! Bridge to `libforge_shim_sm60.so` — Gate 3C.6 (Brief 7).
//!
//! Dynamically loads the FreeTokens-SM60 shim and exposes a safe Rust wrapper
//! around `forge_ggml_moe_vec`. The `.so` path is resolved from:
//! 1. `NAUTIBURN_SHIM_PATH` env var
//! 2. `$CARGO_MANIFEST_DIR/../../FreeTokens-SM60/build/libforge_shim_sm60.so`
//! 3. `/usr/local/lib/libforge_shim_sm60.so`

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use libloading::Library;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ScalarDtype { F32 = 0, F16 = 1 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum GgmlType { Q8_0 = 8 }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShimError {
    NullPointer,        // -1
    InvalidDimensions,  // -2
    CudaAlloc,          // -3
    UnsupportedDtype,   // -5
    Bf16NotSupported,   // -7
    Q40NotImplemented,  // -9
    Unknown(i32),
}

impl std::fmt::Display for ShimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullPointer => write!(f, "null pointer argument"),
            Self::InvalidDimensions => write!(f, "invalid dimensions"),
            Self::CudaAlloc => write!(f, "cudaMallocAsync failed"),
            Self::UnsupportedDtype => write!(f, "unsupported scalar dtype"),
            Self::Bf16NotSupported => write!(f, "BF16 not supported"),
            Self::Q40NotImplemented => write!(f, "Q4_0 not implemented"),
            Self::Unknown(c) => write!(f, "unknown shim error: {c}"),
        }
    }
}

impl std::error::Error for ShimError {}

impl From<i32> for ShimError {
    fn from(code: i32) -> Self {
        match code {
            -1 => Self::NullPointer,  -2 => Self::InvalidDimensions,
            -3 => Self::CudaAlloc,    -5 => Self::UnsupportedDtype,
            -7 => Self::Bf16NotSupported, -9 => Self::Q40NotImplemented,
            other => Self::Unknown(other),
        }
    }
}

/// Signature of `forge_ggml_moe_vec` in the shim.
type ForgeGgmlMoeVec = unsafe extern "C" fn(
    x: *const c_void, packed_w: *const c_void, topk_ids: *const i32,
    output: *mut c_void, scalar_dtype: i32, top_k: i32, ggml_type: i32,
    col: i32, row: i32, tokens: i32, cuda_stream_raw: usize,
) -> i32;

/// Handle to the loaded shim library.
pub struct ForgeShim {
    _lib: Library,
    fn_ptr: ForgeGgmlMoeVec,
}

unsafe impl Send for ForgeShim {}
unsafe impl Sync for ForgeShim {}

impl ForgeShim {
    pub fn load() -> Result<Self, String> {
        let path = find_shim_path()?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<Self, String> {
        let lib = unsafe { Library::new(path) }
            .map_err(|e| format!("load {}: {e}", path.display()))?;
        let fn_ptr: ForgeGgmlMoeVec = unsafe {
            *lib.get(b"forge_ggml_moe_vec\0")
                .map_err(|e| format!("symbol not found: {e}"))?
        };
        Ok(Self { _lib: lib, fn_ptr })
    }

    /// Call `forge_ggml_moe_vec`.
    ///
    /// # Safety
    /// All pointers must be valid CUDA device pointers. `cuda_stream_raw` is a
    /// `cudaStream_t` cast to `usize` (or 0 for the default stream).
    pub unsafe fn gemm_moe_vec(
        &self,
        x: *const c_void, packed_w: *const c_void, topk_ids: *const i32,
        output: *mut c_void, scalar_dtype: ScalarDtype, top_k: i32,
        ggml_type: GgmlType, col: i32, row: i32, tokens: i32,
        cuda_stream_raw: usize,
    ) -> Result<(), ShimError> {
        let rc = (self.fn_ptr)(
            x, packed_w, topk_ids, output,
            scalar_dtype as i32, top_k, ggml_type as i32,
            col, row, tokens, cuda_stream_raw,
        );
        if rc == 0 { Ok(()) } else { Err(ShimError::from(rc)) }
    }

    /// Return raw error code without converting.
    pub unsafe fn gemm_moe_vec_raw(
        &self,
        x: *const c_void, packed_w: *const c_void, topk_ids: *const i32,
        output: *mut c_void, scalar_dtype: i32, top_k: i32, ggml_type: i32,
        col: i32, row: i32, tokens: i32, cuda_stream_raw: usize,
    ) -> i32 {
        (self.fn_ptr)(
            x, packed_w, topk_ids, output, scalar_dtype, top_k, ggml_type,
            col, row, tokens, cuda_stream_raw,
        )
    }
}

/// Singleton shim instance for the process.
static SHIM: OnceLock<Result<ForgeShim, String>> = OnceLock::new();

/// Get or initialize the process-wide shim singleton.
pub fn shim() -> Result<&'static ForgeShim, &'static str> {
    SHIM.get_or_init(|| ForgeShim::load())
        .as_ref()
        .map_err(|e| e.as_str())
}

fn find_shim_path() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("NAUTIBURN_SHIM_PATH") {
        let path = PathBuf::from(&p);
        if path.exists() { return Ok(path); }
        return Err(format!("NAUTIBURN_SHIM_PATH={p} does not exist"));
    }
    // Dev layout: relative to Cargo.toml
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev = manifest.join("../../FreeTokens-SM60/build/libforge_shim_sm60.so");
    if dev.exists() { return Ok(dev); }
    // Known fleet paths
    let known = [
        "/codebase/repos/cesaropsserver/FreeTokens-SM60/build/libforge_shim_sm60.so",
        "/codebase/repos/Freetokens-Pascal/FreeTokens-SM60/build/libforge_shim_sm60.so",
        "/usr/local/lib/libforge_shim_sm60.so",
    ];
    for k in &known {
        let p = PathBuf::from(k);
        if p.exists() { return Ok(p); }
    }
    Err("libforge_shim_sm60.so not found. Set NAUTIBURN_SHIM_PATH".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shim_loads() {
        let s = ForgeShim::load();
        assert!(s.is_ok(), "shim load failed: {:?}", s.err());
    }

    #[test]
    fn shim_singleton_returns_same_instance() {
        let a = shim().unwrap() as *const ForgeShim;
        let b = shim().unwrap() as *const ForgeShim;
        assert_eq!(a, b);
    }

    #[test]
    fn error_codes() {
        let cases = [(-1, "null"), (-2, "dims"), (-3, "alloc"), (-5, "dtype"), (-7, "bf16"), (-9, "q40")];
        for (code, _label) in &cases {
            let err = ShimError::from(*code);
            assert!(!format!("{err}").is_empty());
        }
    }

    #[test]
    fn scalar_dtype_repr() {
        assert_eq!(ScalarDtype::F32 as i32, 0);
        assert_eq!(ScalarDtype::F16 as i32, 1);
    }

    #[test]
    fn ggml_type_repr() {
        assert_eq!(GgmlType::Q8_0 as i32, 8);
    }

    #[test]
    fn null_pointer_returns_error() {
        let s = ForgeShim::load().unwrap();
        unsafe {
            let rc = s.gemm_moe_vec_raw(
                std::ptr::null(), std::ptr::null(), std::ptr::null(),
                std::ptr::null_mut(), 0, 1, 8, 64, 64, 1, 0,
            );
            assert_eq!(rc, -1);
        }
    }

    #[test]
    fn invalid_dims_returns_error() {
        let s = ForgeShim::load().unwrap();
        // top_k=0 should return -2
        unsafe {
            let rc = s.gemm_moe_vec_raw(
                1 as *const _, 1 as *const _, 1 as *const _,
                1 as *mut _, 0, 0, 8, 64, 64, 1, 0,
            );
            assert_eq!(rc, -2);
        }
    }

    #[test]
    fn bf16_rejected() {
        let s = ForgeShim::load().unwrap();
        unsafe {
            let rc = s.gemm_moe_vec_raw(
                1 as *const _, 1 as *const _, 1 as *const _,
                1 as *mut _, 2, 1, 8, 64, 64, 1, 0,
            );
            assert_eq!(rc, -7);
        }
    }

    #[test]
    fn q40_rejected() {
        let s = ForgeShim::load().unwrap();
        unsafe {
            let rc = s.gemm_moe_vec_raw(
                1 as *const _, 1 as *const _, 1 as *const _,
                1 as *mut _, 0, 1, 2, 64, 64, 1, 0,
            );
            assert_eq!(rc, -9);
        }
    }
}