//! Read selected F16/F32 tensors out of a GGUF file.
//!
//! This is a file parser. It does not run llama.cpp.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct Gguf {
    file: File,
    data_start: u64,
    pub tensors: Vec<TensorInfo>,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let mut file = File::open(path.as_ref()).map_err(|err| err.to_string())?;
        let mut hdr = [0u8; 24];
        file.read_exact(&mut hdr).map_err(|err| err.to_string())?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != 0x4655_4747 {
            return Err(format!("not a gguf file, magic={magic:#x}"));
        }
        let tensor_count = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        let mut alignment = 32u64;
        for _ in 0..kv_count {
            let key = read_string(&mut file)?;
            let typ = read_u32(&mut file)?;
            let value = read_value(&mut file, typ)?;
            if key == "general.alignment" {
                if let Value::U32(v) = value {
                    alignment = v as u64;
                }
            }
        }
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            let n_dims = read_u32(&mut file)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut file)?);
            }
            let ggml_type = read_u32(&mut file)?;
            let offset = read_u64(&mut file)?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }
        let pos = file.stream_position().map_err(|err| err.to_string())?;
        let data_start = (pos + alignment - 1) / alignment * alignment;
        Ok(Self {
            file,
            data_start,
            tensors,
        })
    }

    pub fn find(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("missing tensor {name}"))
    }

    pub fn read_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let info = self.find(name)?.clone();
        let elems = numel(&info.dims);
        let bytes = self.read_raw(&info, elems * 4)?;
        if info.ggml_type != GGML_F32 {
            return Err(format!("{name} is type {} not F32", info.ggml_type));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn read_f16_bytes(&mut self, name: &str) -> Result<Vec<u8>, String> {
        let info = self.find(name)?.clone();
        let elems = numel(&info.dims);
        if info.ggml_type != GGML_F16 {
            return Err(format!("{name} is type {} not F16", info.ggml_type));
        }
        self.read_raw(&info, elems * 2)
    }

    pub fn read_f16_slice(
        &mut self,
        name: &str,
        byte_off: u64,
        nbytes: usize,
    ) -> Result<Vec<u8>, String> {
        let info = self.find(name)?.clone();
        if info.ggml_type != GGML_F16 {
            return Err(format!("{name} is type {} not F16", info.ggml_type));
        }
        let start = self.data_start + info.offset + byte_off;
        self.file
            .seek(SeekFrom::Start(start))
            .map_err(|err| err.to_string())?;
        let mut buf = vec![0u8; nbytes];
        self.file
            .read_exact(&mut buf)
            .map_err(|err| format!("read {name}: {err}"))?;
        Ok(buf)
    }

    fn read_raw(&mut self, info: &TensorInfo, nbytes: usize) -> Result<Vec<u8>, String> {
        self.file
            .seek(SeekFrom::Start(self.data_start + info.offset))
            .map_err(|err| err.to_string())?;
        let mut buf = vec![0u8; nbytes];
        self.file
            .read_exact(&mut buf)
            .map_err(|err| format!("read {}: {err}", info.name))?;
        Ok(buf)
    }
}

fn numel(dims: &[u64]) -> usize {
    dims.iter().fold(1usize, |acc, d| acc.saturating_mul(*d as usize))
}

enum Value {
    U32(u32),
    Other,
}

fn read_u32(f: &mut File) -> Result<u32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).map_err(|err| err.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut File) -> Result<u64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|err| err.to_string())?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(f: &mut File) -> Result<String, String> {
    let n = read_u64(f)? as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(|err| err.to_string())?;
    String::from_utf8(buf).map_err(|err| err.to_string())
}

fn read_value(f: &mut File, typ: u32) -> Result<Value, String> {
    match typ {
        8 => {
            let _ = read_string(f)?;
            Ok(Value::Other)
        }
        4 => Ok(Value::U32(read_u32(f)?)),
        5 => {
            let _ = read_u32(f)?;
            Ok(Value::Other)
        }
        6 => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).map_err(|err| err.to_string())?;
            Ok(Value::Other)
        }
        7 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b).map_err(|err| err.to_string())?;
            Ok(Value::Other)
        }
        10 => {
            let _ = read_u64(f)?;
            Ok(Value::Other)
        }
        11 => {
            let _ = read_u64(f)?;
            Ok(Value::Other)
        }
        12 => {
            let mut b = [0u8; 8];
            f.read_exact(&mut b).map_err(|err| err.to_string())?;
            Ok(Value::Other)
        }
        0 | 1 => {
            let mut b = [0u8; 1];
            f.read_exact(&mut b).map_err(|err| err.to_string())?;
            Ok(Value::Other)
        }
        2 | 3 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b).map_err(|err| err.to_string())?;
            Ok(Value::Other)
        }
        9 => {
            let elem = read_u32(f)?;
            let count = read_u64(f)?;
            for _ in 0..count {
                let _ = read_value(f, elem)?;
            }
            Ok(Value::Other)
        }
        other => Err(format!("unsupported gguf value type {other}")),
    }
}
