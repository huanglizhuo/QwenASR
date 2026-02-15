/// C-FFI API for iOS (and other platforms).
/// Compile with: cargo build --release --target aarch64-apple-ios --features ios

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::audio;
use crate::context::QwenCtx;
use crate::kernels;
use crate::transcribe;

/// Opaque handle to the ASR engine.
pub struct QasrEngine {
    ctx: QwenCtx,
}

/// Load model from a directory path. Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn qasr_load_model(
    model_dir: *const c_char,
    n_threads: i32,
    verbosity: i32,
) -> *mut QasrEngine {
    if model_dir.is_null() {
        return std::ptr::null_mut();
    }
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    kernels::set_verbose(verbosity);

    let threads = if n_threads <= 0 {
        kernels::get_num_cpus()
    } else {
        n_threads as usize
    };
    kernels::set_threads(threads);

    match QwenCtx::load(dir) {
        Some(ctx) => Box::into_raw(Box::new(QasrEngine { ctx })),
        None => std::ptr::null_mut(),
    }
}

/// Transcribe a WAV file. Returns a heap-allocated C string (caller must free with qasr_free_string).
#[no_mangle]
pub unsafe extern "C" fn qasr_transcribe_file(
    engine: *mut QasrEngine,
    wav_path: *const c_char,
) -> *mut c_char {
    if engine.is_null() || wav_path.is_null() {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let path = match CStr::from_ptr(wav_path).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match transcribe::transcribe(&mut eng.ctx, path) {
        Some(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Transcribe raw PCM samples (f32, 16kHz, mono).
/// Returns a heap-allocated C string (caller must free with qasr_free_string).
#[no_mangle]
pub unsafe extern "C" fn qasr_transcribe_pcm(
    engine: *mut QasrEngine,
    samples: *const f32,
    n_samples: i32,
) -> *mut c_char {
    if engine.is_null() || samples.is_null() || n_samples <= 0 {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let pcm = std::slice::from_raw_parts(samples, n_samples as usize);

    match transcribe::transcribe_audio(&mut eng.ctx, pcm) {
        Some(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Transcribe raw WAV buffer (entire file contents including header).
/// Returns a heap-allocated C string (caller must free with qasr_free_string).
#[no_mangle]
pub unsafe extern "C" fn qasr_transcribe_wav_buffer(
    engine: *mut QasrEngine,
    wav_data: *const u8,
    wav_len: i32,
) -> *mut c_char {
    if engine.is_null() || wav_data.is_null() || wav_len <= 0 {
        return std::ptr::null_mut();
    }
    let eng = &mut *engine;
    let data = std::slice::from_raw_parts(wav_data, wav_len as usize);

    let samples = match audio::parse_wav_buffer(data) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };

    match transcribe::transcribe_audio(&mut eng.ctx, &samples) {
        Some(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

/// Set segmentation seconds (0 = no segmentation).
#[no_mangle]
pub unsafe extern "C" fn qasr_set_segment_sec(engine: *mut QasrEngine, sec: f32) {
    if !engine.is_null() {
        (*engine).ctx.segment_sec = sec;
    }
}

/// Set language (e.g. "English", "Chinese"). Empty string = auto-detect.
#[no_mangle]
pub unsafe extern "C" fn qasr_set_language(
    engine: *mut QasrEngine,
    language: *const c_char,
) -> i32 {
    if engine.is_null() || language.is_null() {
        return -1;
    }
    let lang = match CStr::from_ptr(language).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match (*engine).ctx.set_force_language(lang) {
        Ok(()) => 0,
        Err(()) => -1,
    }
}

/// Free a string returned by qasr_transcribe_*.
#[no_mangle]
pub unsafe extern "C" fn qasr_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free the engine.
#[no_mangle]
pub unsafe extern "C" fn qasr_free(engine: *mut QasrEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}
