//! GGUF binary format reader with quantized tensor support.
//!
//! Supports GGUF versions 2 and 3. Provides zero-copy mmap access to raw
//! quantized weight data plus an on-demand `get_f32()` dequant path.

use std::collections::HashMap;

// ========================================================================
// GGML quantization type enum
// ========================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    BF16,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0  => Self::F32,
            1  => Self::F16,
            2  => Self::Q4_0,
            3  => Self::Q4_1,
            6  => Self::Q5_0,
            7  => Self::Q5_1,
            8  => Self::Q8_0,
            9  => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            30 => Self::BF16,
            x  => Self::Unknown(x),
        }
    }

    /// Elements per quantization block (1 for float types).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            Self::Unknown(_) => 1,
        }
    }

    /// Bytes per block (or per element for float types).
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32  => 4,
            Self::F16  => 2,
            Self::BF16 => 2,
            Self::Q4_0 => 18,   // 2 (f16 d) + 16 (qs)
            Self::Q4_1 => 20,   // 2 (f16 d) + 2 (f16 m) + 16 (qs)
            Self::Q5_0 => 22,   // 2 + 4 + 16
            Self::Q5_1 => 24,   // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,   // 2 (f16 d) + 32 (qs)
            Self::Q8_1 => 40,   // 4 (f32 d) + 4 (f32 s) + 32 (qs)
            Self::Q2_K => 84,   // 16 + 64 + 2 + 2
            Self::Q3_K => 110,  // 32 + 64 + 12 + 2
            Self::Q4_K => 144,  // 2 + 2 + 12 + 128
            Self::Q5_K => 176,  // 2 + 2 + 12 + 32 + 128
            Self::Q6_K => 210,  // 128 + 64 + 16 + 2
            Self::Q8_K => 292,  // 4 + 256 + 2*16 + 4
            Self::Unknown(_) => 0,
        }
    }

    pub fn n_blocks(&self, n_elements: usize) -> usize {
        let bs = self.block_size();
        if bs == 1 { n_elements } else { (n_elements + bs - 1) / bs }
    }

    /// Total byte count for n_elements of this type.
    pub fn storage_bytes(&self, n_elements: usize) -> usize {
        self.n_blocks(n_elements) * self.type_size()
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::Unknown(_))
    }
}

// ========================================================================
// Tensor metadata
// ========================================================================

#[derive(Clone, Debug)]
pub struct GgufTensorMeta {
    pub name: String,
    pub shape: Vec<u64>,   // innermost-first (GGUF convention)
    pub ggml_type: GgmlType,
    pub offset: u64,       // byte offset within tensor data section
}

impl GgufTensorMeta {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
}

// ========================================================================
// KV metadata value
// ========================================================================

#[derive(Debug, Clone)]
pub enum GgufMetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufMetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

// ========================================================================
// GGUF file
// ========================================================================

pub struct GgufFile {
    data: *mut u8,
    file_size: usize,
    pub data_start: usize,     // byte offset where tensor data begins (32-byte aligned)
    pub tensors: Vec<GgufTensorMeta>,
    tensor_map: HashMap<String, usize>,
    pub metadata: HashMap<String, GgufMetaValue>,
}

unsafe impl Send for GgufFile {}
unsafe impl Sync for GgufFile {}

impl Drop for GgufFile {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe { libc::munmap(self.data as *mut _, self.file_size); }
        }
    }
}

impl GgufFile {
    pub fn open(path: &str) -> Option<Self> {
        use libc::*;
        use std::ffi::CString;

        let c_path = CString::new(path).ok()?;
        let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
        if fd < 0 { return None; }

        let mut stat_buf = unsafe { std::mem::zeroed::<stat>() };
        if unsafe { fstat(fd, &mut stat_buf) } < 0 {
            unsafe { close(fd); }
            return None;
        }
        let file_size = stat_buf.st_size as usize;
        if file_size < 24 {
            unsafe { close(fd); }
            return None;
        }

        let data = unsafe {
            mmap(std::ptr::null_mut(), file_size, PROT_READ, MAP_PRIVATE, fd, 0)
        };
        unsafe { close(fd); }

        if data == MAP_FAILED { return None; }
        let data = data as *mut u8;

        let slice = unsafe { std::slice::from_raw_parts(data, file_size) };
        let mut r = Reader::new(slice);

        // Magic
        let magic = r.read_bytes(4)?;
        if magic != b"GGUF" {
            unsafe { munmap(data as *mut _, file_size); }
            return None;
        }

        let version = r.read_u32()?;
        if version < 2 || version > 3 {
            eprintln!("gguf: unsupported version {}", version);
            unsafe { munmap(data as *mut _, file_size); }
            return None;
        }

        let n_tensors = r.read_u64()? as usize;
        let n_kv = r.read_u64()? as usize;

        // Parse KV metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = r.read_string()?;
            let value_type = r.read_u32()?;
            let value = r.read_meta_value(value_type)?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_map = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.read_string()?;
            let n_dims = r.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64()?);
            }
            let ggml_type = GgmlType::from_u32(r.read_u32()?);
            let offset = r.read_u64()?;
            let idx = tensors.len();
            tensor_map.insert(name.clone(), idx);
            tensors.push(GgufTensorMeta { name, shape, ggml_type, offset });
        }

        // Data section starts at next 32-byte alignment after header
        let data_start = align_up(r.pos(), 32);
        if data_start > file_size {
            unsafe { munmap(data as *mut _, file_size); }
            return None;
        }

        Some(GgufFile { data, file_size, data_start, tensors, tensor_map, metadata })
    }

    pub fn find(&self, name: &str) -> Option<&GgufTensorMeta> {
        self.tensor_map.get(name).map(|&i| &self.tensors[i])
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_map.contains_key(name)
    }

    /// Raw pointer to the start of this tensor's data in the mmap.
    pub fn tensor_data(&self, t: &GgufTensorMeta) -> *const u8 {
        unsafe { self.data.add(self.data_start + t.offset as usize) }
    }

    /// Dequantize any tensor type to a Vec<f32>.
    pub fn get_f32(&self, t: &GgufTensorMeta) -> Option<Vec<f32>> {
        let n = t.numel();
        if n == 0 { return None; }
        let ptr = self.tensor_data(t);
        Some(match t.ggml_type {
            GgmlType::F32 => {
                let mut out = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), n);
                }
                out
            }
            GgmlType::F16 => {
                unsafe {
                    let src = std::slice::from_raw_parts(ptr as *const u16, n);
                    src.iter().map(|&h| f16_to_f32(h)).collect()
                }
            }
            GgmlType::BF16 => {
                unsafe {
                    let src = std::slice::from_raw_parts(ptr as *const u16, n);
                    src.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
                }
            }
            GgmlType::Q8_0 => dequant_q8_0(ptr, n),
            GgmlType::Q4_K => dequant_q4_k(ptr, n),
            GgmlType::Q4_0 => dequant_q4_0(ptr, n),
            _ => {
                eprintln!("gguf: get_f32: unsupported type {:?}", t.ggml_type);
                return None;
            }
        })
    }

    // ---- KV accessors ----

    pub fn get_kv_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufMetaValue::U32(v)) => Some(*v),
            Some(GgufMetaValue::U64(v)) => Some(*v as u32),
            Some(GgufMetaValue::I32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_kv_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufMetaValue::U64(v)) => Some(*v),
            Some(GgufMetaValue::U32(v)) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_kv_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufMetaValue::F32(v)) => Some(*v),
            Some(GgufMetaValue::F64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn get_kv_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufMetaValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Add an alias: if `from` exists in tensor_map, also register it as `to`.
    pub fn add_alias(&mut self, from: &str, to: &str) {
        if let Some(&idx) = self.tensor_map.get(from) {
            self.tensor_map.entry(to.to_string()).or_insert(idx);
        }
    }

    /// Remap ggml-org llama.cpp decoder tensor names to internal HuggingFace-style names.
    /// Detects by presence of `token_embd.weight`.
    pub fn remap_decoder_names(&mut self) {
        if !self.tensor_map.contains_key("token_embd.weight") {
            return;
        }
        self.add_alias("token_embd.weight", "thinker.model.embed_tokens.weight");
        self.add_alias("output.weight",      "thinker.lm_head.weight");
        self.add_alias("output_norm.weight", "thinker.model.norm.weight");
        // How many layers? Use tensor count to find max blk.N
        let n_layers = {
            let mut max = 0usize;
            for name in self.tensor_map.keys() {
                if let Some(rest) = name.strip_prefix("blk.") {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(n) = rest[..dot].parse::<usize>() {
                            if n + 1 > max { max = n + 1; }
                        }
                    }
                }
            }
            max
        };
        for n in 0..n_layers {
            let src = format!("blk.{}", n);
            let dst = format!("thinker.model.layers.{}", n);
            self.add_alias(&format!("{}.attn_q.weight",    src), &format!("{}.self_attn.q_proj.weight",          dst));
            self.add_alias(&format!("{}.attn_k.weight",    src), &format!("{}.self_attn.k_proj.weight",          dst));
            self.add_alias(&format!("{}.attn_v.weight",    src), &format!("{}.self_attn.v_proj.weight",          dst));
            self.add_alias(&format!("{}.attn_output.weight", src), &format!("{}.self_attn.o_proj.weight",        dst));
            self.add_alias(&format!("{}.attn_norm.weight", src), &format!("{}.input_layernorm.weight",           dst));
            self.add_alias(&format!("{}.ffn_norm.weight",  src), &format!("{}.post_attention_layernorm.weight",  dst));
            self.add_alias(&format!("{}.attn_q_norm.weight", src), &format!("{}.self_attn.q_norm.weight",        dst));
            self.add_alias(&format!("{}.attn_k_norm.weight", src), &format!("{}.self_attn.k_norm.weight",        dst));
            self.add_alias(&format!("{}.ffn_gate.weight",  src), &format!("{}.mlp.gate_proj.weight",            dst));
            self.add_alias(&format!("{}.ffn_up.weight",    src), &format!("{}.mlp.up_proj.weight",              dst));
            self.add_alias(&format!("{}.ffn_down.weight",  src), &format!("{}.mlp.down_proj.weight",            dst));
        }
    }

    /// Remap ggml-org mmproj encoder tensor names to internal HuggingFace-style names.
    /// Detects by presence of `a.conv2d.1.weight`.
    pub fn remap_encoder_names(&mut self) {
        if !self.tensor_map.contains_key("a.conv2d.1.weight") {
            return;
        }
        let p = "thinker.audio_tower.";
        self.add_alias("a.conv2d.1.weight", &format!("{}conv2d1.weight",  p));
        self.add_alias("a.conv2d.1.bias",   &format!("{}conv2d1.bias",    p));
        self.add_alias("a.conv2d.2.weight", &format!("{}conv2d2.weight",  p));
        self.add_alias("a.conv2d.2.bias",   &format!("{}conv2d2.bias",    p));
        self.add_alias("a.conv2d.3.weight", &format!("{}conv2d3.weight",  p));
        self.add_alias("a.conv2d.3.bias",   &format!("{}conv2d3.bias",    p));
        self.add_alias("a.conv_out.weight", &format!("{}conv_out.weight", p));
        self.add_alias("a.post_ln.weight",  &format!("{}ln_post.weight",  p));
        self.add_alias("a.post_ln.bias",    &format!("{}ln_post.bias",    p));
        self.add_alias("mm.a.mlp.1.weight", &format!("{}proj1.weight",    p));
        self.add_alias("mm.a.mlp.1.bias",   &format!("{}proj1.bias",      p));
        self.add_alias("mm.a.mlp.2.weight", &format!("{}proj2.weight",    p));
        self.add_alias("mm.a.mlp.2.bias",   &format!("{}proj2.bias",      p));
        // layers
        let n_layers = {
            let mut max = 0usize;
            for name in self.tensor_map.keys() {
                if let Some(rest) = name.strip_prefix("a.blk.") {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(n) = rest[..dot].parse::<usize>() {
                            if n + 1 > max { max = n + 1; }
                        }
                    }
                }
            }
            max
        };
        for n in 0..n_layers {
            let src = format!("a.blk.{}", n);
            let lp  = format!("{}layers.{}", p, n);
            self.add_alias(&format!("{}.attn_q.weight",  src), &format!("{}.self_attn.q_proj.weight",      lp));
            self.add_alias(&format!("{}.attn_q.bias",    src), &format!("{}.self_attn.q_proj.bias",        lp));
            self.add_alias(&format!("{}.attn_k.weight",  src), &format!("{}.self_attn.k_proj.weight",      lp));
            self.add_alias(&format!("{}.attn_k.bias",    src), &format!("{}.self_attn.k_proj.bias",        lp));
            self.add_alias(&format!("{}.attn_v.weight",  src), &format!("{}.self_attn.v_proj.weight",      lp));
            self.add_alias(&format!("{}.attn_v.bias",    src), &format!("{}.self_attn.v_proj.bias",        lp));
            self.add_alias(&format!("{}.attn_out.weight",src), &format!("{}.self_attn.out_proj.weight",    lp));
            self.add_alias(&format!("{}.attn_out.bias",  src), &format!("{}.self_attn.out_proj.bias",      lp));
            self.add_alias(&format!("{}.ln1.weight",     src), &format!("{}.self_attn_layer_norm.weight",  lp));
            self.add_alias(&format!("{}.ln1.bias",       src), &format!("{}.self_attn_layer_norm.bias",    lp));
            self.add_alias(&format!("{}.ffn_up.weight",  src), &format!("{}.fc1.weight",                   lp));
            self.add_alias(&format!("{}.ffn_up.bias",    src), &format!("{}.fc1.bias",                     lp));
            self.add_alias(&format!("{}.ffn_down.weight",src), &format!("{}.fc2.weight",                   lp));
            self.add_alias(&format!("{}.ffn_down.bias",  src), &format!("{}.fc2.bias",                     lp));
            self.add_alias(&format!("{}.ln2.weight",     src), &format!("{}.final_layer_norm.weight",      lp));
            self.add_alias(&format!("{}.ln2.bias",       src), &format!("{}.final_layer_norm.bias",        lp));
        }
    }
}

// ========================================================================
// Binary reader
// ========================================================================

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Reader { data, pos: 0 } }
    fn pos(&self) -> usize { self.pos }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.data.len() { return None; }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }

    fn read_u8(&mut self) -> Option<u8> {
        let b = self.read_bytes(1)?;
        Some(b[0])
    }

    fn read_u16(&mut self) -> Option<u16> {
        let b = self.read_bytes(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let b = self.read_bytes(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let b = self.read_bytes(8)?;
        Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_i8(&mut self) -> Option<i8> { Some(self.read_u8()? as i8) }
    fn read_i16(&mut self) -> Option<i16> { Some(self.read_u16()? as i16) }
    fn read_i32(&mut self) -> Option<i32> { Some(self.read_u32()? as i32) }
    fn read_i64(&mut self) -> Option<i64> { Some(self.read_u64()? as i64) }

    fn read_f32(&mut self) -> Option<f32> { Some(f32::from_bits(self.read_u32()?)) }
    fn read_f64(&mut self) -> Option<f64> { Some(f64::from_bits(self.read_u64()?)) }

    fn read_bool(&mut self) -> Option<bool> { Some(self.read_u8()? != 0) }

    fn read_string(&mut self) -> Option<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).ok()
    }

    fn read_meta_value(&mut self, vtype: u32) -> Option<GgufMetaValue> {
        match vtype {
            0  => Some(GgufMetaValue::U8(self.read_u8()?)),
            1  => Some(GgufMetaValue::I8(self.read_i8()?)),
            2  => Some(GgufMetaValue::U16(self.read_u16()?)),
            3  => Some(GgufMetaValue::I16(self.read_i16()?)),
            4  => Some(GgufMetaValue::U32(self.read_u32()?)),
            5  => Some(GgufMetaValue::I32(self.read_i32()?)),
            6  => Some(GgufMetaValue::F32(self.read_f32()?)),
            7  => Some(GgufMetaValue::Bool(self.read_bool()?)),
            8  => Some(GgufMetaValue::Str(self.read_string()?)),
            9  => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_meta_value(elem_type)?);
                }
                Some(GgufMetaValue::Array(arr))
            }
            10 => Some(GgufMetaValue::U64(self.read_u64()?)),
            11 => Some(GgufMetaValue::I64(self.read_i64()?)),
            12 => Some(GgufMetaValue::F64(self.read_f64()?)),
            _  => {
                eprintln!("gguf: unknown metadata value type {}", vtype);
                None
            }
        }
    }
}

// ========================================================================
// Dequantization helpers
// ========================================================================

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let sign = (h >> 15) as u32;
    if exp == 0 {
        f32::from_bits((sign << 31) | (mant << 13))
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Q8_0: 34 bytes per block — [d: f16][qs: i8×32]
pub fn dequant_q8_0(src: *const u8, n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut out = vec![0.0f32; n_blocks * 32];
    for b in 0..n_blocks {
        let block = unsafe { src.add(b * 34) };
        let d_bits = unsafe { u16::from_le_bytes([*block, *block.add(1)]) };
        let d = f16_to_f32(d_bits);
        for i in 0..32 {
            let q = unsafe { *block.add(2 + i) as i8 };
            out[b * 32 + i] = q as f32 * d;
        }
    }
    out.truncate(n_elements);
    out
}

/// Q4_0: 18 bytes per block — [d: f16][qs: u8×16] (32 nibbles, zero-point = 8)
pub fn dequant_q4_0(src: *const u8, n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 31) / 32;
    let mut out = vec![0.0f32; n_blocks * 32];
    for b in 0..n_blocks {
        let block = unsafe { src.add(b * 18) };
        let d_bits = unsafe { u16::from_le_bytes([*block, *block.add(1)]) };
        let d = f16_to_f32(d_bits);
        for i in 0..16 {
            let byte = unsafe { *block.add(2 + i) };
            let lo = (byte & 0xF) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            out[b * 32 + i]      = lo as f32 * d;
            out[b * 32 + 16 + i] = hi as f32 * d;
        }
    }
    out.truncate(n_elements);
    out
}

/// Unpack 6-bit scale and min for Q4_K sub-block j (0..8).
/// Mirrors llama.cpp `get_scale_min_k4`.
#[inline]
fn get_scale_min_k4(j: usize, scales: *const u8) -> (u8, u8) {
    unsafe {
        if j < 4 {
            let d = *scales.add(j) & 63;
            let m = *scales.add(j + 4) & 63;
            (d, m)
        } else {
            let d = (*scales.add(j + 4) & 0xF) | ((*scales.add(j - 4) >> 6) << 4);
            let m = (*scales.add(j + 4) >> 4)  | ((*scales.add(j    ) >> 6) << 4);
            (d, m)
        }
    }
}

/// Q4_K: 144 bytes per super-block — [d: f16][dmin: f16][scales: u8×12][qs: u8×128]
/// Each super-block covers 256 elements, split into 8 sub-blocks of 32.
pub fn dequant_q4_k(src: *const u8, n_elements: usize) -> Vec<f32> {
    let n_blocks = (n_elements + 255) / 256;
    let mut out = vec![0.0f32; n_blocks * 256];
    for b in 0..n_blocks {
        let p = unsafe { src.add(b * 144) };
        let d_bits    = unsafe { u16::from_le_bytes([*p,        *p.add(1)]) };
        let dmin_bits = unsafe { u16::from_le_bytes([*p.add(2), *p.add(3)]) };
        let d    = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dmin_bits);
        let scales_ptr = unsafe { p.add(4) };    // 12 bytes
        let qs_ptr     = unsafe { p.add(16) };   // 128 bytes

        let base = b * 256;
        let mut q_off = 0usize;
        let mut is = 0usize;

        // 4 outer iterations × 64 elements = 256 elements
        for j_step in 0..4 {
            let (sc0, mc0) = get_scale_min_k4(is, scales_ptr);
            let sc0f = d * sc0 as f32;
            let mc0f = dmin * mc0 as f32;
            let (sc1, mc1) = get_scale_min_k4(is + 1, scales_ptr);
            let sc1f = d * sc1 as f32;
            let mc1f = dmin * mc1 as f32;

            for l in 0..32 {
                let byte = unsafe { *qs_ptr.add(q_off + l) };
                out[base + j_step * 64 + l]      = sc0f * (byte & 0xF) as f32 - mc0f;
                out[base + j_step * 64 + 32 + l] = sc1f * (byte >> 4)  as f32 - mc1f;
            }
            q_off += 32;
            is += 2;
        }
    }
    out.truncate(n_elements);
    out
}

/// Public re-export of the F16→F32 conversion for use in kernels.
#[inline]
pub fn f16_to_f32_pub(h: u16) -> f32 { f16_to_f32(h) }

/// Public re-export of Q4_K scale/min unpacker for use in kernels.
#[inline]
pub fn get_scale_min_k4_pub(j: usize, scales: *const u8) -> (u8, u8) {
    get_scale_min_k4(j, scales)
}

/// Dequantize a single row from a Q8_0 matrix (for token embedding lookup).
/// `row_elements` is the number of elements in one row (= hidden_dim).
/// `row_idx` is the row to extract (= token_id).
pub fn dequant_q8_0_row(dst: &mut [f32], src: *const u8, row_idx: usize, row_elements: usize) {
    debug_assert_eq!(row_elements % 32, 0);
    let blocks_per_row = row_elements / 32;
    let row_ptr = unsafe { src.add(row_idx * blocks_per_row * 34) };
    for b in 0..blocks_per_row {
        let block = unsafe { row_ptr.add(b * 34) };
        let d_bits = unsafe { u16::from_le_bytes([*block, *block.add(1)]) };
        let d = f16_to_f32(d_bits);
        for i in 0..32 {
            let q = unsafe { *block.add(2 + i) as i8 };
            dst[b * 32 + i] = q as f32 * d;
        }
    }
}

/// Dequantize a single row from a Q4_K matrix.
pub fn dequant_q4k_row(dst: &mut [f32], src: *const u8, row_idx: usize, row_elements: usize) {
    debug_assert_eq!(row_elements % 256, 0);
    let blocks_per_row = row_elements / 256;
    let row_ptr = unsafe { src.add(row_idx * blocks_per_row * 144) };
    let tmp = dequant_q4_k(row_ptr, row_elements);
    dst.copy_from_slice(&tmp[..row_elements]);
}

/// Dequantize a single row from a Q4_0 matrix.
pub fn dequant_q4_0_row(dst: &mut [f32], src: *const u8, row_idx: usize, row_elements: usize) {
    debug_assert_eq!(row_elements % 32, 0);
    let blocks_per_row = row_elements / 32;
    let row_ptr = unsafe { src.add(row_idx * blocks_per_row * 18) };
    let tmp = dequant_q4_0(row_ptr, row_elements);
    dst.copy_from_slice(&tmp[..row_elements]);
}

// ========================================================================
// Utilities
// ========================================================================

fn align_up(v: usize, align: usize) -> usize {
    (v + align - 1) & !(align - 1)
}
