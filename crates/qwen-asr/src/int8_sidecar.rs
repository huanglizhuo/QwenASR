//! Memory-mappable INT8 weight sidecar for the decoder (R12-H1).
//!
//! At load the decoder quantizes all non-aligner decode weights to INT8 with
//! per-row scales (see `decoder::quantize_to_superpage`). That BF16->INT8 pass
//! plus the interleaved gate_up fusion costs measurable startup time and holds
//! the INT8 copies as anonymous RSS. This module persists the quantized buffers
//! into a single sidecar file laid out so it can be `mmap`'d and consumed **in
//! place**: on a warm run the decoder's INT8 weight structs point at slices INTO
//! the mmap with zero bulk copies.
//!
//! This is deliberately different from the previously-rejected weight caches
//! (Round 3 "A1", F19/F20, G23): those read the cache back into owned `Vec`s
//! (a multi-GB copy) which is slower than the existing mmap + on-demand
//! conversion of the ~1.2 GB safetensors. Here the sidecar is never copied in
//! bulk — [`WeightBuf::Mapped`] borrows the mmap region directly and the hot
//! kernels are unchanged (they take `*const i8`/`&[i8]`, obtained via `Deref`).
//!
//! Layout (little-endian):
//!   [0..8)   magic
//!   [8..12)  version
//!   [12..16) num_buffers
//!   [16..24) identity_hash (fnv-1a over model files' name/size/mtime)
//!   [24..32) model_total_size
//!   [32..36) dec_layers      [36..40) dec_hidden     [40..44) dec_intermediate
//!   [44..48) dec_heads       [48..52) dec_kv_heads   [52..56) dec_head_dim
//!   [56..60) lm_out_dim      [60..64) reserved
//!   [64..)   table: num_buffers * (offset u64, len u64)
//!   data:    each buffer padded to `BUF_ALIGN`
//!
//! Buffer order: for each layer `i` in `0..dec_layers`, 12 buffers
//!   [wq, wq_scales, wk, wk_scales, wv, wv_scales, wo, wo_scales,
//!    gate_up, gate_up_scales, down, down_scales]
//! then `[lm_head, lm_head_scales]`. i8 buffers store `len` bytes; f32 scale
//! buffers store `len` bytes = 4 * element count.

use crate::config::QwenConfig;

const MAGIC: u64 = 0x3854_4e49_5253_4151; // "QASRINT8" little-endian
const VERSION: u32 = 1;
/// Per-buffer alignment inside the file. mmap returns a page-aligned base, so a
/// page-aligned in-file offset yields a page-aligned pointer for every buffer.
/// The NEON INT8 kernels only require natural/element alignment (`vld1`/`ldr`
/// have no alignment requirement on AArch64); the superpage backing used by the
/// owned `superpage_vec` path is a TLB hint, not a correctness requirement, so
/// page alignment here is safe. 16 KiB matches the Apple-Silicon page size.
const BUF_ALIGN: u64 = 16384;
const BUFS_PER_LAYER: usize = 12;
const HEADER_FIXED: u64 = 64;

const fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

/// A weight buffer that is either owned (freshly quantized this run) or a
/// borrowed slice into the mmap'd sidecar. Both variants deref to `&[T]`, so
/// call sites (`.as_ptr()`, `&buf` coerced to `&[T]`) are identical and the hot
/// kernels are unchanged.
pub enum WeightBuf<T: Copy> {
    Owned(Vec<T>),
    /// Points into a `SidecarMmap` that outlives this buffer (kept alive in the
    /// same `Decoder`). Never freed here.
    Mapped {
        ptr: *const T,
        len: usize,
    },
}

impl<T: Copy> WeightBuf<T> {
    pub fn empty() -> Self {
        WeightBuf::Owned(Vec::new())
    }
}

impl<T: Copy> From<Vec<T>> for WeightBuf<T> {
    fn from(v: Vec<T>) -> Self {
        WeightBuf::Owned(v)
    }
}

impl<T: Copy> std::ops::Deref for WeightBuf<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        match self {
            WeightBuf::Owned(v) => v.as_slice(),
            // Safety: `ptr`/`len` describe a live, immutable range inside the
            // owning `SidecarMmap`, which outlives every `WeightBuf::Mapped`.
            WeightBuf::Mapped { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

/// Deterministic byte lengths of every sidecar buffer, in canonical order.
fn buffer_byte_lens(cfg: &QwenConfig) -> Vec<u64> {
    let hidden = cfg.dec_hidden;
    let inter = cfg.dec_intermediate;
    let q_dim = cfg.dec_heads * cfg.dec_head_dim;
    let kv_dim = cfg.dec_kv_heads * cfg.dec_head_dim;
    let lm = cfg.lm_head_dim();
    let mut v = Vec::with_capacity(cfg.dec_layers * BUFS_PER_LAYER + 2);
    for _ in 0..cfg.dec_layers {
        v.push((q_dim * hidden) as u64); // wq (i8)
        v.push((q_dim * 4) as u64); // wq_scales (f32)
        v.push((kv_dim * hidden) as u64); // wk
        v.push((kv_dim * 4) as u64);
        v.push((kv_dim * hidden) as u64); // wv
        v.push((kv_dim * 4) as u64);
        v.push((hidden * q_dim) as u64); // wo
        v.push((hidden * 4) as u64);
        v.push((2 * inter * hidden) as u64); // gate_up
        v.push((2 * inter * 4) as u64);
        v.push((hidden * inter) as u64); // down
        v.push((hidden * 4) as u64);
    }
    v.push((lm * hidden) as u64); // lm_head (i8)
    v.push((lm * 4) as u64); // lm_head_scales (f32)
    v
}

/// Deterministic file layout derived purely from config dims.
pub struct SidecarLayout {
    pub offsets: Vec<u64>,
    pub lens: Vec<u64>,
    pub total_size: u64,
}

impl SidecarLayout {
    pub fn compute(cfg: &QwenConfig) -> SidecarLayout {
        let lens = buffer_byte_lens(cfg);
        let num_buffers = lens.len() as u64;
        let table_end = HEADER_FIXED + num_buffers * 16;
        let mut cursor = align_up(table_end, BUF_ALIGN);
        let mut offsets = Vec::with_capacity(lens.len());
        for &len in &lens {
            offsets.push(cursor);
            cursor = align_up(cursor + len, BUF_ALIGN);
        }
        SidecarLayout {
            offsets,
            lens,
            total_size: cursor,
        }
    }

    /// Global buffer index for layer `i`, sub-buffer `j` (0..12).
    pub fn layer_buf(&self, i: usize, j: usize) -> usize {
        i * BUFS_PER_LAYER + j
    }

    pub fn lm_head_idx(&self, nlayers: usize) -> usize {
        nlayers * BUFS_PER_LAYER
    }
}

/// fnv-1a over the model's safetensors files (name + size + mtime). Returns
/// `(hash, total_size)`. Matches the file scan order used by `MultiSafetensors`.
fn model_identity(model_dir: &str) -> (u64, u64) {
    use std::path::Path;
    let mut names: Vec<String> = Vec::new();
    let single = Path::new(model_dir).join("model.safetensors");
    if single.exists() {
        names.push("model.safetensors".to_string());
    } else if let Ok(entries) = std::fs::read_dir(model_dir) {
        for e in entries.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if n.starts_with("model-") && n.ends_with(".safetensors") {
                names.push(n);
            }
        }
    }
    names.sort();

    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut total: u64 = 0;
    let mix = |bytes: &[u8], h: &mut u64| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for name in &names {
        let path = Path::new(model_dir).join(name);
        let (size, mtime_ns) = match std::fs::metadata(&path) {
            Ok(m) => {
                let size = m.len();
                let mtime = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                (size, mtime)
            }
            Err(_) => (0, 0),
        };
        total = total.wrapping_add(size);
        mix(name.as_bytes(), &mut hash);
        mix(&size.to_le_bytes(), &mut hash);
        mix(&mtime_ns.to_le_bytes(), &mut hash);
    }
    (hash, total)
}

/// A read-only mmap of a validated sidecar. Kept alive in the `Decoder`; every
/// `WeightBuf::Mapped` handed out borrows from `base`.
pub struct SidecarMmap {
    base: *mut u8,
    size: usize,
    offsets: Vec<u64>,
    lens: Vec<u64>,
}

// Safety: `base` is an immutable mmap; the raw pointer is only read. The owning
// `Decoder` is already `unsafe impl Send + Sync` for its BF16 mmap pointers.
unsafe impl Send for SidecarMmap {}
unsafe impl Sync for SidecarMmap {}

impl Drop for SidecarMmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base as *mut _, self.size);
        }
    }
}

impl SidecarMmap {
    /// mmap and validate an existing sidecar. Returns `None` (caller rebuilds)
    /// on any mismatch: missing file, bad magic/version, changed model, or
    /// changed config dims.
    pub fn open_valid(
        path: &str,
        model_dir: &str,
        cfg: &QwenConfig,
        layout: &SidecarLayout,
    ) -> Option<SidecarMmap> {
        use libc::*;
        use std::ffi::CString;

        let c_path = CString::new(path).ok()?;
        let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return None;
        }
        let mut st = unsafe { std::mem::zeroed::<stat>() };
        if unsafe { fstat(fd, &mut st) } < 0 {
            unsafe { close(fd) };
            return None;
        }
        let size = st.st_size as usize;
        if (size as u64) < layout.total_size {
            unsafe { close(fd) };
            return None;
        }
        let base = unsafe { mmap(std::ptr::null_mut(), size, PROT_READ, MAP_PRIVATE, fd, 0) };
        unsafe { close(fd) };
        if base == libc::MAP_FAILED {
            return None;
        }
        let base = base as *mut u8;

        let ok = validate_header(base, size, model_dir, cfg, layout);
        if !ok {
            unsafe { munmap(base as *mut _, size) };
            return None;
        }

        Some(SidecarMmap {
            base,
            size,
            offsets: layout.offsets.clone(),
            lens: layout.lens.clone(),
        })
    }

    #[inline]
    pub fn i8_buf(&self, idx: usize) -> WeightBuf<i8> {
        let ptr = unsafe { self.base.add(self.offsets[idx] as usize) } as *const i8;
        WeightBuf::Mapped {
            ptr,
            len: self.lens[idx] as usize,
        }
    }

    #[inline]
    pub fn f32_buf(&self, idx: usize) -> WeightBuf<f32> {
        let ptr = unsafe { self.base.add(self.offsets[idx] as usize) } as *const f32;
        WeightBuf::Mapped {
            ptr,
            len: (self.lens[idx] / 4) as usize,
        }
    }
}

fn read_u32(base: *const u8, off: usize) -> u32 {
    let mut b = [0u8; 4];
    unsafe { std::ptr::copy_nonoverlapping(base.add(off), b.as_mut_ptr(), 4) };
    u32::from_le_bytes(b)
}
fn read_u64(base: *const u8, off: usize) -> u64 {
    let mut b = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(base.add(off), b.as_mut_ptr(), 8) };
    u64::from_le_bytes(b)
}

fn validate_header(
    base: *const u8,
    size: usize,
    model_dir: &str,
    cfg: &QwenConfig,
    layout: &SidecarLayout,
) -> bool {
    if size < HEADER_FIXED as usize {
        return false;
    }
    if read_u64(base, 0) != MAGIC || read_u32(base, 8) != VERSION {
        return false;
    }
    let num_buffers = read_u32(base, 12) as usize;
    if num_buffers != layout.lens.len() {
        return false;
    }
    let (id_hash, total_size) = model_identity(model_dir);
    if read_u64(base, 16) != id_hash || read_u64(base, 24) != total_size {
        return false;
    }
    let lm = cfg.lm_head_dim() as u32;
    let dims = [
        cfg.dec_layers as u32,
        cfg.dec_hidden as u32,
        cfg.dec_intermediate as u32,
        cfg.dec_heads as u32,
        cfg.dec_kv_heads as u32,
        cfg.dec_head_dim as u32,
        lm,
    ];
    for (k, &d) in dims.iter().enumerate() {
        if read_u32(base, 32 + k * 4) != d {
            return false;
        }
    }
    // Verify the stored offset/len table matches the recomputed layout so a
    // format/alignment change can never be silently misread.
    for (k, (&off, &len)) in layout.offsets.iter().zip(layout.lens.iter()).enumerate() {
        let base_off = HEADER_FIXED as usize + k * 16;
        if read_u64(base, base_off) != off || read_u64(base, base_off + 8) != len {
            return false;
        }
    }
    true
}

/// Serialize the freshly-quantized buffers (in canonical order) to a new sidecar
/// at `path`. Written to a temp file then atomically renamed, so a crash mid
/// write never leaves a file that would pass validation. `bufs[k]` must be the
/// raw bytes of buffer `k` with `bufs[k].len() == layout.lens[k]`. Best-effort:
/// returns `false` on any I/O error (caller keeps its owned Vecs regardless).
pub fn write_sidecar(
    path: &str,
    model_dir: &str,
    cfg: &QwenConfig,
    layout: &SidecarLayout,
    bufs: &[&[u8]],
) -> bool {
    use std::io::Write;
    use std::os::unix::fs::FileExt;

    if bufs.len() != layout.lens.len() {
        return false;
    }
    let tmp = format!("{}.tmp.{}", path, std::process::id());
    let file = match std::fs::File::create(&tmp) {
        Ok(f) => f,
        Err(_) => return false,
    };
    if file.set_len(layout.total_size).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }

    // Header + table.
    let (id_hash, total_size) = model_identity(model_dir);
    let num_buffers = layout.lens.len() as u32;
    let lm = cfg.lm_head_dim() as u32;
    let mut header = vec![0u8; layout.offsets[0] as usize];
    header[0..8].copy_from_slice(&MAGIC.to_le_bytes());
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&num_buffers.to_le_bytes());
    header[16..24].copy_from_slice(&id_hash.to_le_bytes());
    header[24..32].copy_from_slice(&total_size.to_le_bytes());
    let dims = [
        cfg.dec_layers as u32,
        cfg.dec_hidden as u32,
        cfg.dec_intermediate as u32,
        cfg.dec_heads as u32,
        cfg.dec_kv_heads as u32,
        cfg.dec_head_dim as u32,
        lm,
    ];
    for (k, &d) in dims.iter().enumerate() {
        header[32 + k * 4..36 + k * 4].copy_from_slice(&d.to_le_bytes());
    }
    for (k, (&off, &len)) in layout.offsets.iter().zip(layout.lens.iter()).enumerate() {
        let b = HEADER_FIXED as usize + k * 16;
        header[b..b + 8].copy_from_slice(&off.to_le_bytes());
        header[b + 8..b + 16].copy_from_slice(&len.to_le_bytes());
    }
    if file.write_all_at(&header, 0).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }

    // Data buffers at their aligned offsets.
    for (k, chunk) in bufs.iter().enumerate() {
        if chunk.len() as u64 != layout.lens[k] {
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        if file.write_all_at(chunk, layout.offsets[k]).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
    }
    drop({
        let mut f = file;
        let _ = f.flush();
        f
    });

    if std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

/// Default sidecar path next to the model directory.
pub fn sidecar_path(model_dir: &str) -> String {
    format!("{}/qwen-asr-int8.sidecar", model_dir)
}

/// Whether the sidecar is enabled. Disabled via `QWEN_ASR_SIDECAR=0`.
pub fn enabled() -> bool {
    std::env::var("QWEN_ASR_SIDECAR")
        .map(|v| v != "0")
        .unwrap_or(true)
}
