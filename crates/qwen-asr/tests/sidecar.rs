//! R12-H1: mmap-backed INT8 weight sidecar round-trip + invalidation.
//!
//! Verifies that a warm run which mmaps the sidecar produces INT8 weights/scales
//! byte-identical to a cold run that freshly quantizes them, and that a
//! corrupted or mismatched header triggers a rebuild (never a stale read).

use qwen_asr::context::QwenModel;
use qwen_asr::kernels;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

mod common;

fn model_dir() -> Option<String> {
    match common::model_dir() {
        Some(d) => Some(d),
        None => {
            eprintln!("Skipping sidecar test: model not downloaded");
            None
        }
    }
}

fn i8_bytes(s: &[i8]) -> Vec<u8> {
    s.iter().map(|&b| b as u8).collect()
}
fn f32_bytes(s: &[f32]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Capture bytes of representative INT8 buffers across first/middle/last layers
/// plus lm_head, so byte-equality covers the full weight set.
fn snapshot(m: &Arc<QwenModel>) -> Vec<Vec<u8>> {
    let d = &m.decoder;
    let n = d.layers.len();
    let mut v = Vec::new();
    for li in [0usize, n / 2, n - 1] {
        let l = &d.layers[li];
        v.push(i8_bytes(&l.wq_int8));
        v.push(f32_bytes(&l.wq_int8_scales));
        v.push(i8_bytes(&l.wk_int8));
        v.push(i8_bytes(&l.wv_int8));
        v.push(i8_bytes(&l.wo_int8));
        v.push(i8_bytes(&l.gate_up_int8));
        v.push(f32_bytes(&l.gate_up_int8_scales));
        v.push(i8_bytes(&l.down_int8));
        v.push(f32_bytes(&l.down_int8_scales));
    }
    v.push(i8_bytes(d.lm_head_int8.as_ref().unwrap()));
    v.push(f32_bytes(d.lm_head_int8_scales.as_ref().unwrap()));
    v
}

#[test]
fn sidecar_roundtrip_and_invalidation() {
    let Some(dir) = model_dir() else {
        return;
    };
    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());

    let sc = format!("{}/qwen-asr-int8.sidecar", dir);
    let _ = std::fs::remove_file(&sc);

    // Cold run: quantizes into owned Vecs and writes the sidecar.
    let a = QwenModel::load(&dir).expect("cold load");
    assert!(
        a.decoder._int8_sidecar.is_none(),
        "cold run must own freshly-quantized Vecs, not mmap"
    );
    assert!(
        std::path::Path::new(&sc).exists(),
        "cold run must write the sidecar"
    );
    let snap_a = snapshot(&a);

    // Warm run: mmaps the sidecar and borrows INT8 weights in place.
    let b = QwenModel::load(&dir).expect("warm load");
    assert!(
        b.decoder._int8_sidecar.is_some(),
        "warm run must mmap the sidecar"
    );
    let snap_b = snapshot(&b);
    assert_eq!(
        snap_a, snap_b,
        "mmap'd INT8 weights/scales must be byte-identical to freshly quantized"
    );

    // Corrupted magic -> rebuild (fall back to owned) and still match.
    {
        let f = std::fs::OpenOptions::new().write(true).open(&sc).unwrap();
        f.write_all_at(&[0xffu8; 8], 0).unwrap();
    }
    let c = QwenModel::load(&dir).expect("corrupt-magic load");
    assert!(
        c.decoder._int8_sidecar.is_none(),
        "corrupted magic must trigger a rebuild, never a stale mmap read"
    );
    assert_eq!(
        snapshot(&c),
        snap_a,
        "rebuild must reproduce identical weights"
    );

    // The rebuild rewrote a valid sidecar -> warm mmap works again.
    let d = QwenModel::load(&dir).expect("re-warm load");
    assert!(
        d.decoder._int8_sidecar.is_some(),
        "sidecar must be valid again after rebuild"
    );

    // Mismatched model identity hash (offset 16) -> rebuild.
    {
        let f = std::fs::OpenOptions::new().write(true).open(&sc).unwrap();
        f.write_all_at(&[0xaau8; 8], 16).unwrap();
    }
    let e = QwenModel::load(&dir).expect("identity-mismatch load");
    assert!(
        e.decoder._int8_sidecar.is_none(),
        "model-identity mismatch must trigger a rebuild"
    );

    // Leave a valid warm sidecar behind (rebuild above rewrote it).
    let _ = std::fs::remove_file(&sc);
}
