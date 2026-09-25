//! Links Burn's WGPU backend from the checkout under `upstream/burn`.
//!
//! Naming the backend type is enough to compile it. This crate does not
//! create a device and does not launch a kernel.

/// The Burn WGPU backend is linked. No device is opened.
pub fn linked_wgpu_backend() -> &'static str {
    let _ = core::any::type_name::<burn::backend::Wgpu>();
    "wgpu"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_backend_is_linked_without_opening_a_device() {
        assert_eq!(linked_wgpu_backend(), "wgpu");
    }
}
