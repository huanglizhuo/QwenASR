//! CPU-only Qwen3-ASR speech recognition in pure Rust.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use qwen_asr::context::QwenCtx;
//! use qwen_asr::transcribe;
//!
//! let mut ctx = QwenCtx::load("qwen3-asr-0.6b").expect("model not found");
//! let text = transcribe::transcribe(&mut ctx, "audio.wav").unwrap();
//! println!("{text}");
//! ```
//!
//! # Forced Alignment
//!
//! With the aligner model variant you can obtain word-level timestamps for a
//! known transcript:
//!
//! ```rust,no_run
//! use qwen_asr::context::QwenCtx;
//! use qwen_asr::align;
//!
//! let mut ctx = QwenCtx::load("qwen3-aligner-0.6b").expect("aligner model not found");
//! let samples: Vec<f32> = vec![]; // 16 kHz mono f32 PCM
//! let results = align::forced_align(&mut ctx, &samples, "Hello world", "English").unwrap();
//! for r in &results {
//!     println!("{}: {:.0} – {:.0} ms", r.text, r.start_ms, r.end_ms);
//! }
//! ```
//!
//! # Module Guide
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`context`] | Engine state — start here with [`context::QwenCtx::load`] |
//! | [`transcribe`] | Offline, segmented, and streaming transcription |
//! | [`audio`] | WAV loading, resampling, mel spectrogram |
//! | [`align`] | Forced alignment (word/character timestamps) |
//! | [`config`] | Model configuration and variant detection |
//! | [`tokenizer`] | GPT-2 byte-level BPE tokenizer |
//!
//! The remaining modules (`encoder`, `decoder`, `kernels`, `safetensors`) are
//! implementation details and not intended for direct use.

#![allow(dead_code)]

pub mod config;
pub mod safetensors;
pub mod audio;
pub mod tokenizer;
pub mod kernels;
pub mod encoder;
pub mod decoder;
pub mod context;
pub mod transcribe;
pub mod align;
#[cfg(any(feature = "ios", feature = "android"))]
pub mod c_api;
#[cfg(feature = "android")]
pub mod jni_api;
