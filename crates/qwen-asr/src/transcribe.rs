//! Offline, segmented, and streaming transcription orchestration.

use crate::align;
use crate::audio;
use crate::config::*;
use crate::context::QwenCtx;
use crate::decoder::{self, tok_embed_bf16_to_f32, Decoder, DecoderBuffers, KvCache, RopeCache};
use crate::encoder::{Encoder, EncoderBuffers};
use crate::kernels;
use crate::output::{SegmentResult, TranscriptionResult, WordTimestamp};
use crate::subtitle::{format_vtt, group_words_to_cues, segment_to_cue, Cue};
use crate::tokenizer::QwenTokenizer;

use std::time::Instant;

// Prompt token sequences
const PREFIX_HEAD: &[i32] = &[151644, 8948, 198];
const PREFIX_TAIL: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
const SUFFIX_BASE: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

// Streaming robustness constants (matching C reference)
const STREAM_DEGEN_MAX_PERIOD: usize = 6;
const STREAM_DEGEN_MIN_REPEATS: usize = 4;
const STREAM_STALE_CHUNKS: i32 = 4;
const STREAM_RESET_INTERVAL_CHUNKS: i32 = 45;
const STREAM_RESET_CARRY_TOKENS: usize = 24;
const STREAM_MAX_ENC_WINDOWS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
struct PrefillRowKey {
    a: u64,
    b: u64,
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[inline]
fn prefill_token_key(token: i32) -> PrefillRowKey {
    let x = token as u32 as u64;
    PrefillRowKey {
        a: mix64(0x544f_4b45_4e00_0000 ^ x),
        b: mix64(0x454d_4245_4400_0000 ^ x.rotate_left(17)),
    }
}

fn prefill_embed_key(row: &[f32]) -> PrefillRowKey {
    let mut a = 0xcbf29ce484222325u64;
    let mut b = 0x9e3779b97f4a7c15u64 ^ row.len() as u64;
    for &v in row {
        let bits = v.to_bits() as u64;
        a ^= bits;
        a = a.wrapping_mul(0x100000001b3);
        b ^= bits
            .wrapping_add(0x9e3779b97f4a7c15)
            .wrapping_add(b << 6)
            .wrapping_add(b >> 2);
    }
    PrefillRowKey { a, b }
}

fn prefill_embed_keys(data: &[f32], seq_len: usize, dim: usize) -> Vec<PrefillRowKey> {
    let mut keys = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        keys.push(prefill_embed_key(&data[i * dim..(i + 1) * dim]));
    }
    keys
}

fn has_prefilled_asr_text(ctx: &QwenCtx) -> bool {
    ctx.force_prompt_tokens
        .as_ref()
        .is_some_and(|tokens| tokens.contains(&TOKEN_ASR_TEXT))
}

fn prefill_lcp_len(prev: &[PrefillRowKey], current: &[PrefillRowKey], prefill_len: usize) -> usize {
    let cmp_len = prefill_len.min(prev.len()).min(current.len());
    let mut reused = 0usize;
    while reused < cmp_len && prev[reused] == current[reused] {
        reused += 1;
    }
    reused
}

fn get_time_ms() -> f64 {
    // Use monotonic clock
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms(t0: f64) -> f64 {
    get_time_ms() - t0
}

/// Returns `(best_reps, best_period)`: how many times a block of `best_period`
/// tokens repeats at the tail of `tokens`. Used for streaming degeneracy detection.
fn stream_tail_repeat_blocks(tokens: &[i32], max_period: usize) -> (usize, usize) {
    let n = tokens.len();
    if n < 2 {
        return (1, 0);
    }
    let period_cap = (n / 2).min(if max_period > 0 { max_period } else { n / 2 });
    let mut best_reps = 1usize;
    let mut best_period = 0usize;
    for p in 1..=period_cap {
        let mut reps = 1usize;
        while (reps + 1) * p <= n {
            let a = &tokens[n - (reps + 1) * p..n - reps * p];
            let b = &tokens[n - reps * p..n - (reps - 1) * p];
            if a != b {
                break;
            }
            reps += 1;
        }
        if reps > best_reps {
            best_reps = reps;
            best_period = p;
        }
    }
    (best_reps, best_period)
}

fn load_tokenizer(model_dir: &str) -> Option<QwenTokenizer> {
    let _pg = kernels::ProfileGuard::new(&kernels::PROF.tokenizer_load);
    let vocab_path = format!("{}/vocab.json", model_dir);
    QwenTokenizer::load(&vocab_path)
}

/// Immutable, thread-shareable model weights plus the prepared prompt token
/// state needed to decode one segment. Every field is read-only during decode,
/// so a single `SegWeights` can be shared by `&` across concurrent segment
/// workers (see [`transcribe_audio`]'s parallel path).
struct SegWeights<'a> {
    cfg: &'a QwenConfig,
    encoder: &'a Encoder,
    decoder: &'a Decoder,
    prompt_tokens: Option<&'a [i32]>,
    force_prompt_tokens: Option<&'a [i32]>,
    has_prefilled_asr_text: bool,
}

/// Per-segment mutable session buffers. Each concurrent worker owns its own set
/// so no cross-thread synchronization is needed inside a decode.
struct SegSession {
    kv_cache: KvCache,
    rope_cache: RopeCache,
    dec_bufs: DecoderBuffers,
    enc_bufs: EncoderBuffers,
}

impl SegSession {
    fn new(cfg: &QwenConfig) -> Self {
        SegSession {
            kv_cache: KvCache::new(cfg.dec_layers, 2048, cfg.dec_kv_heads, cfg.dec_head_dim),
            rope_cache: RopeCache::new(),
            dec_bufs: DecoderBuffers::new(cfg),
            enc_bufs: EncoderBuffers::new(),
        }
    }
}

/// Output of decoding one segment, with the ctx-side side effects deferred so
/// the core is a pure function of `(weights, session, samples)`.
struct SegCore {
    /// Whitespace-trimmed transcript text for this segment.
    text: String,
    /// Raw header bytes (pre-`<asr_text>` tokens) for language detection.
    header_bytes: Vec<u8>,
    n_text_tokens: i32,
    /// Mel + encoder time (ms).
    encode_ms: f64,
    /// Prefill + autoregressive decode time (ms).
    decode_ms: f64,
    /// Wall time of the whole segment decode (ms).
    total_ms: f64,
}

/// Decode a single segment against shared `weights` using a worker-owned
/// `session`. `on_piece` receives the raw bytes of each decoded *text* token as
/// it is produced (the serial path streams these through `token_cb`; the
/// parallel path captures them for ordered replay). This function touches no
/// [`QwenCtx`] state, so it is safe to run on many threads concurrently as long
/// as each call has its own `session`.
#[allow(clippy::too_many_arguments)]
fn decode_segment_core(
    weights: &SegWeights,
    kv_cache: &mut KvCache,
    rope_cache: &mut RopeCache,
    dec_bufs: &mut DecoderBuffers,
    enc_bufs: &mut EncoderBuffers,
    tokenizer: &QwenTokenizer,
    samples: &[f32],
    past_tokens: Option<&[i32]>,
    on_piece: &mut dyn FnMut(&[u8]),
) -> Option<SegCore> {
    let cfg = weights.cfg;
    let dim = cfg.dec_hidden;
    let seg_t0 = get_time_ms();
    let mut n_text_tokens = 0i32;

    // Mel spectrogram
    let t0 = get_time_ms();
    let (mel, mel_frames) = {
        let _pg = kernels::ProfileGuard::new(&kernels::PROF.mel_compute);
        audio::mel_spectrogram(samples)?
    };
    let mel_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Mel: {} frames ({:.0} ms)", mel_frames, mel_ms);
    }

    // Encoder
    let t0 = get_time_ms();
    let (enc_output, enc_seq_len) =
        weights
            .encoder
            .forward(cfg, &mel, mel_frames, Some(enc_bufs))?;
    let enc_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Encoder: {} tokens ({:.0} ms)", enc_seq_len, enc_ms);
    }

    // Build input embeddings
    let n_prompt_tokens = weights.prompt_tokens.map_or(0, |t| t.len());
    let n_force_prompt_tokens = weights.force_prompt_tokens.map_or(0, |t| t.len());
    let n_past = past_tokens.map_or(0, |t| t.len());
    let n_past_prompt_tokens = if n_past > 0 { n_past + 1 } else { 0 }; // +1 for <asr_text>

    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let tok_emb = weights.decoder.tok_embeddings_bf16;

    // Embed prefix head
    let mut off = 0;
    for &tok in PREFIX_HEAD {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok_emb,
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Optional prompt
    if let Some(ptoks) = weights.prompt_tokens {
        for &tok in ptoks {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            off += 1;
        }
    }

    // Prefix tail
    for &tok in PREFIX_TAIL {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds[off * dim..(off + 1) * dim],
                tok_emb,
                tok,
                dim,
            )
        };
        off += 1;
    }

    // Encoder output
    for i in 0..enc_seq_len {
        input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
            .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
    }

    // Suffix base
    let suffix_off = prefix_len + enc_seq_len;
    for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                tok_emb,
                tok,
                dim,
            )
        };
    }

    // Force language tokens
    if let Some(ftoks) = weights.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                        ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
        }
    }

    // Past text conditioning tokens
    let past_off = suffix_off + suffix_len;
    if let Some(ptoks) = past_tokens {
        for (i, &tok) in ptoks.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(past_off + i) * dim..(past_off + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
        }
        unsafe {
            tok_embed_bf16_to_f32(
                &mut input_embeds
                    [(past_off + ptoks.len()) * dim..(past_off + ptoks.len() + 1) * dim],
                tok_emb,
                TOKEN_ASR_TEXT,
                dim,
            )
        };
    }

    // Decoder prefill
    let t0 = get_time_ms();
    kv_cache.len = 0;
    let prefill_len = total_seq - 1;
    decoder::decoder_prefill(
        weights.decoder,
        cfg,
        kv_cache,
        rope_cache,
        dec_bufs,
        &input_embeds,
        prefill_len,
    );

    // First token from last prefill position
    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = decoder::decoder_forward(
        weights.decoder,
        cfg,
        kv_cache,
        rope_cache,
        dec_bufs,
        last_embed,
    );

    let prefill_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!("  Prefill: {} tokens ({:.0} ms)", total_seq, prefill_ms);
    }

    // Autoregressive decode
    let t0 = get_time_ms();
    let max_tokens = 2048;
    let mut n_generated = 0;
    let mut past_asr_text = weights.has_prefilled_asr_text || n_past > 0;

    let mut text_bytes: Vec<u8> = Vec::new();
    let mut header_bytes: Vec<u8> = Vec::new();
    let mut tmp_embed = vec![0.0f32; dim];

    while n_generated < max_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        if token == TOKEN_ASR_TEXT {
            past_asr_text = true;
        } else if past_asr_text {
            let piece_bytes = tokenizer.decode_bytes(token);
            text_bytes.extend_from_slice(piece_bytes);
            n_text_tokens += 1;
            on_piece(piece_bytes);
        } else if token < 151643 {
            let piece_bytes = tokenizer.decode_bytes(token);
            header_bytes.extend_from_slice(piece_bytes);
        }

        unsafe { tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim) };
        token = decoder::decoder_forward(
            weights.decoder,
            cfg,
            kv_cache,
            rope_cache,
            dec_bufs,
            &tmp_embed,
        );
    }

    let decode_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!(
            "  Decode: {} tokens ({:.0} ms, {:.1} ms/token)",
            n_generated,
            decode_ms,
            if n_generated > 0 {
                decode_ms / n_generated as f64
            } else {
                0.0
            }
        );
    }

    let text = String::from_utf8_lossy(&text_bytes);
    let trimmed = text.trim().to_string();

    Some(SegCore {
        text: trimmed,
        header_bytes,
        n_text_tokens,
        encode_ms: mel_ms + enc_ms,
        decode_ms: prefill_ms + decode_ms,
        total_ms: elapsed_ms(seg_t0),
    })
}

/// Fold a decoded [`SegCore`] back into `ctx`: language detection from the
/// header and perf-counter accumulation. Shared by the serial and parallel
/// segment paths so both update `ctx` identically.
fn apply_seg_core(ctx: &mut QwenCtx, core: &SegCore) {
    if ctx.force_language.is_none() {
        if let Some(language) = parse_language_header(&String::from_utf8_lossy(&core.header_bytes)) {
            ctx.detected_language = Some(language);
        }
    }
    ctx.perf_text_tokens += core.n_text_tokens;
    ctx.perf_encode_ms += core.encode_ms;
    ctx.perf_decode_ms += core.decode_ms;
}

/// Transcribe a single segment against `ctx`, streaming text tokens through
/// `ctx.token_cb` as they are produced. Returns (text, n_text_tokens).
fn transcribe_segment(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    past_tokens: Option<&[i32]>,
) -> Option<(String, i32)> {
    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    let cfg = ctx.config.clone();
    let weights = SegWeights {
        cfg: &cfg,
        encoder: &ctx.encoder,
        decoder: &ctx.decoder,
        prompt_tokens: ctx.prompt_tokens.as_deref(),
        force_prompt_tokens: ctx.force_prompt_tokens.as_deref(),
        has_prefilled_asr_text: has_prefilled_asr_text(ctx),
    };

    // Disjoint field borrows: `token_cb` (shared) is a different field from the
    // four mutable session buffers, so the closure and the `&mut` args coexist.
    let token_cb = ctx.token_cb.as_ref();
    let mut on_piece = |bytes: &[u8]| {
        if let Some(cb) = token_cb {
            // For the callback, provide lossy UTF-8 for display purposes
            cb(&String::from_utf8_lossy(bytes));
        }
    };

    let core = decode_segment_core(
        &weights,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &mut ctx.enc_bufs,
        tokenizer,
        samples,
        past_tokens,
        &mut on_piece,
    )?;

    apply_seg_core(ctx, &core);
    ctx.perf_total_ms += core.total_ms;
    Some((core.text, core.n_text_tokens))
}

fn parse_language_header(header: &str) -> Option<String> {
    let trimmed = header.trim();
    if let Some(language) = normalize_language(trimmed) {
        return Some(language);
    }
    let (prefix, rest) = trimmed.split_once(char::is_whitespace)?;
    if !prefix.eq_ignore_ascii_case("language") {
        return None;
    }
    normalize_language(rest)
}

// ========================================================================
// Segment-based splitting
// ========================================================================

fn find_split_point(samples: &[f32], target_sample: usize, search_sec: f32) -> usize {
    let search_half = (search_sec * SAMPLE_RATE as f32) as usize;
    let lo = target_sample.saturating_sub(search_half);
    let hi = (target_sample + search_half).min(samples.len());

    let win_samples = 1600; // 100ms at 16kHz
    let mut best_energy = 1e30f32;
    let mut best_center = target_sample;

    let mut pos = lo;
    while pos + win_samples <= hi {
        let end = (pos + win_samples).min(samples.len());
        let mut energy = 0.0f32;
        for &s in samples.iter().take(end).skip(pos) {
            energy += s * s;
        }
        energy /= (end - pos) as f32;
        if energy < best_energy {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
        pos += win_samples / 2;
    }

    best_center
}

fn should_insert_boundary_space(prev_ch: u8, next_ch: u8) -> bool {
    if prev_ch == 0 || next_ch == 0 {
        return false;
    }
    if (prev_ch as char).is_whitespace() {
        return false;
    }
    if (next_ch as char).is_whitespace() {
        return false;
    }
    if (next_ch as char).is_ascii_punctuation() {
        return false;
    }
    true
}

// ========================================================================
// Public API
// ========================================================================

/// A single transcribed segment with wall-clock timestamps.
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// Transcribe audio and return per-segment results with timestamps.
///
/// Unlike [`transcribe_audio`], this function preserves the original audio
/// timeline (no silence compaction) so that `start_ms`/`end_ms` are accurate
/// for subtitle generation.  Each segment's duration is used to set the token
/// budget, so accuracy is not sacrificed for long files.
///
/// Uses `ctx.segment_sec` for splitting; falls back to 30 s if unset.
pub fn transcribe_segmented(ctx: &mut QwenCtx, samples: &[f32]) -> Option<Vec<TranscriptSegment>> {
    ctx.reset_perf();

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let mut segments: Vec<TranscriptSegment> = Vec::new();
    transcribe_splits(ctx, samples, &tokenizer, &mut |_, seg_start, seg_end, text| {
        segments.push(TranscriptSegment {
            start_ms: samples_to_ms(seg_start),
            end_ms: samples_to_ms(seg_end),
            text,
        });
    });

    Some(segments)
}

/// Build low-energy split points for segmented transcription. Always contains
/// a leading `0`; further splits are added only when the audio exceeds one
/// segment (`target_samples + margin_samples`).
fn build_split_points(
    samples: &[f32],
    target_samples: usize,
    margin_samples: usize,
    search_sec: f32,
) -> Vec<usize> {
    let mut splits = vec![0usize];
    let mut pos = 0;
    while pos + target_samples + margin_samples < samples.len() {
        let split = find_split_point(samples, pos + target_samples, search_sec);
        splits.push(split);
        pos = split;
    }
    splits
}

/// Shared segmentation driver for timestamp-preserving transcription: split
/// `samples` at low-energy boundaries (`ctx.segment_sec`, 30 s fallback),
/// transcribe each split, and invoke `on_text` with
/// `(ctx, seg_start, seg_end, text)` for every non-empty segment transcript.
fn transcribe_splits(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    on_text: &mut dyn FnMut(&mut QwenCtx, usize, usize, String),
) {
    let segment_sec = if ctx.segment_sec > 0.0 {
        ctx.segment_sec
    } else {
        30.0
    };
    let search_sec = ctx.search_sec.min(segment_sec / 2.0);
    let target_samples = (segment_sec * SAMPLE_RATE as f32) as usize;
    let margin_samples = (search_sec * SAMPLE_RATE as f32) as usize;
    let min_samples = SAMPLE_RATE as usize / 2;

    let splits = build_split_points(samples, target_samples, margin_samples, search_sec);
    let n_splits = splits.len();

    for s in 0..n_splits {
        let seg_start = splits[s];
        let seg_end = if s + 1 < n_splits {
            splits[s + 1]
        } else {
            samples.len()
        };
        let seg_len = seg_end - seg_start;

        // Set perf_audio_ms to this segment's duration so any per-segment
        // token budgets are relative to the segment, not the full file.
        ctx.perf_audio_ms = 1000.0 * seg_len as f64 / SAMPLE_RATE as f64;

        let seg_buf: Vec<f32>;
        let seg_ptr = if seg_len < min_samples {
            seg_buf = {
                let mut buf = vec![0.0f32; min_samples];
                buf[..seg_len].copy_from_slice(&samples[seg_start..seg_end]);
                buf
            };
            &seg_buf[..]
        } else {
            &samples[seg_start..seg_end]
        };

        let (text, _) = match transcribe_segment(ctx, seg_ptr, tokenizer, None) {
            Some(r) => r,
            None => continue,
        };

        if text.is_empty() {
            continue;
        }

        on_text(ctx, seg_start, seg_end, text);
    }
}

/// Callback invoked after each transcribed segment with the segment result and
/// its subtitle cues.
pub type OnSegmentFn<'a> = &'a mut dyn FnMut(&SegmentResult, &[Cue]);

pub fn transcribe_full(
    ctx: &mut QwenCtx,
    mut aligner: Option<&mut QwenCtx>,
    samples: &[f32],
    mut on_segment: Option<OnSegmentFn<'_>>,
) -> Option<TranscriptionResult> {
    ctx.reset_perf();

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let audio_end_ms = samples_to_ms(samples.len());
    let mut segments = Vec::new();
    let mut all_cues: Vec<Cue> = Vec::new();
    let mut text = String::new();

    transcribe_splits(ctx, samples, &tokenizer, &mut |ctx, seg_start, seg_end, seg_text| {
        let start_ms = samples_to_ms(seg_start);
        let end_ms = samples_to_ms(seg_end);

        if !text.is_empty() {
            let prev = *text.as_bytes().last().unwrap_or(&0);
            let next = *seg_text.as_bytes().first().unwrap_or(&0);
            if should_insert_boundary_space(prev, next) {
                text.push(' ');
            }
        }
        text.push_str(&seg_text);

        let language = ctx
            .force_language
            .as_deref()
            .or(ctx.detected_language.as_deref())
            .unwrap_or("English")
            .to_string();

        let mut words = Vec::new();
        let mut cue_words = Vec::new();
        if let Some(aligner_ctx) = aligner.as_deref_mut() {
            if let Some(aligned) = align::forced_align(
                aligner_ctx,
                &samples[seg_start..seg_end],
                &seg_text,
                &language,
            ) {
                for word in aligned {
                    let word_start = start_ms + word.start_ms.max(0.0).round() as u64;
                    let word_end = start_ms + word.end_ms.max(0.0).round() as u64;
                    cue_words.push(align::AlignResult {
                        text: word.text.clone(),
                        start_ms: word_start as f32,
                        end_ms: word_end as f32,
                    });
                    words.push(WordTimestamp {
                        word: word.text,
                        start_ms: word_start,
                        end_ms: word_end,
                    });
                }
            }
        }

        // Compute this segment's cues exactly once; the on_segment callback
        // (incremental SRT/VTT writing) and the final `vtt` field both see
        // the same cue list, keeping streamed and one-shot output identical.
        let seg_cues = if cue_words.is_empty() {
            vec![segment_to_cue(&TranscriptSegment {
                start_ms,
                end_ms,
                text: seg_text.clone(),
            })]
        } else {
            group_words_to_cues(&cue_words, audio_end_ms)
        };

        let segment = SegmentResult {
            start_ms,
            end_ms,
            text: seg_text,
            words,
        };

        if let Some(callback) = on_segment.as_deref_mut() {
            callback(&segment, &seg_cues);
        }
        all_cues.extend(seg_cues);
        segments.push(segment);
    });

    let language = ctx
        .force_language
        .as_deref()
        .or(ctx.detected_language.as_deref())
        .and_then(language_to_iso639)
        .unwrap_or("")
        .to_string();

    Some(TranscriptionResult {
        language,
        duration_ms: samples_to_ms(samples.len()),
        text,
        segments,
        vtt: format_vtt(&all_cues),
    })
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples as u64 * 1000) / SAMPLE_RATE as u64
}

/// Transcribe audio samples (f32, 16 kHz, mono, range [-1, 1]).
///
/// When `ctx.segment_sec > 0` and the audio exceeds that duration, it is
/// automatically split at low-energy boundaries.
/// Returns `None` if the tokenizer or encoder fails to initialize.
pub fn transcribe_audio(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    let audio_samples = if ctx.skip_silence {
        let compacted = if ctx.segment_sec > 0.0 {
            audio::compact_silence_fast(samples)
        } else {
            audio::compact_silence(samples)
        };
        if kernels::verbose() >= 1 {
            let used_pct = 100.0 * compacted.len() as f32 / samples.len().max(1) as f32;
            eprintln!(
                "Silence skip: used {:.1}%, skipped {:.1}% ({} -> {} samples)",
                used_pct,
                100.0 - used_pct,
                samples.len(),
                compacted.len()
            );
        }
        compacted
    } else {
        samples.to_vec()
    };

    if kernels::verbose() >= 2 {
        eprintln!(
            "Audio: {} samples ({:.1} seconds)",
            audio_samples.len(),
            audio_samples.len() as f32 / SAMPLE_RATE as f32
        );
    }

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let target_samples = (ctx.segment_sec * SAMPLE_RATE as f32) as usize;
    let search = ctx.search_sec.min(ctx.segment_sec / 2.0);
    let margin_samples = (search * SAMPLE_RATE as f32) as usize;

    // No splitting if segment_sec is 0 or audio fits in one segment
    if ctx.segment_sec <= 0.0 || audio_samples.len() <= target_samples + margin_samples {
        let (text, _) = transcribe_segment(ctx, &audio_samples, &tokenizer, None)?;
        return Some(text);
    }

    // Build split points
    let splits = build_split_points(&audio_samples, target_samples, margin_samples, search);
    let n_splits = splits.len();

    if kernels::verbose() >= 2 {
        eprintln!("Splitting into {} segments", n_splits);
    }

    let mut result = String::new();
    let min_samples = SAMPLE_RATE as usize / 2;
    let use_past_text = ctx.past_text_conditioning;

    // Parallel fast path: when segments are independent (no past-text
    // conditioning) and there is more than one, decode them concurrently across
    // K single-threaded workers (see `transcribe_splits_parallel`). Each
    // segment's decode is bit-identical to the serial path, so the assembled
    // transcript is byte-identical. Past-text conditioning creates a serial
    // dependency (segment N's prompt = segment N-1's text) and stays serial.
    if !use_past_text && n_splits > 1 {
        let k = segment_worker_count(n_splits);
        if k >= 2 {
            return transcribe_splits_parallel(ctx, &audio_samples, &splits, &tokenizer, k);
        }
    }

    for s in 0..n_splits {
        let core_start = splits[s];
        let core_end = if s + 1 < n_splits {
            splits[s + 1]
        } else {
            audio_samples.len()
        };
        let seg_start = core_start;
        let seg_end = core_end;
        let seg_samples = seg_end - seg_start;

        if kernels::verbose() >= 2 {
            eprintln!(
                "Segment {}/{}: {:.1}-{:.1}s ({} samples)",
                s + 1,
                n_splits,
                seg_start as f32 / SAMPLE_RATE as f32,
                seg_end as f32 / SAMPLE_RATE as f32,
                seg_samples
            );
        }

        let seg_buf: Vec<f32>;
        let seg_ptr = if seg_samples < min_samples {
            seg_buf = {
                let mut buf = vec![0.0f32; min_samples];
                buf[..seg_samples].copy_from_slice(&audio_samples[seg_start..seg_end]);
                buf
            };
            &seg_buf[..]
        } else {
            &audio_samples[seg_start..seg_end]
        };

        let past_tokens: Option<Vec<i32>> = if use_past_text && !result.is_empty() {
            tokenizer.encode(&result)
        } else {
            None
        };

        let (seg_text, _seg_text_tokens) =
            match transcribe_segment(ctx, seg_ptr, &tokenizer, past_tokens.as_deref()) {
                Some(r) => r,
                None => continue,
            };

        if seg_text.is_empty() {
            continue;
        }

        let need_space = if !result.is_empty() {
            let prev = *result.as_bytes().last().unwrap_or(&0);
            let next = *seg_text.as_bytes().first().unwrap_or(&0);
            should_insert_boundary_space(prev, next)
        } else {
            false
        };

        if need_space {
            result.push(' ');
            if let Some(ref cb) = ctx.token_cb {
                cb(" ");
            }
        }

        if let Some(ref cb) = ctx.token_cb {
            if ctx.past_text_conditioning {
                cb(&seg_text);
            }
        }

        result.push_str(&seg_text);
    }

    Some(result)
}

/// Number of concurrent single-threaded segment workers to use. Defaults to the
/// configured hot-kernel thread count plus up to two spare CPUs, capped at the
/// number of segments; a `1` count means "stay serial". The
/// `QWEN_ASR_SEG_WORKERS` environment variable overrides the default (internal
/// tuning knob used for the K-sweep; not part of any public API).
fn segment_worker_count(n_splits: usize) -> usize {
    let base = std::env::var("QWEN_ASR_SEG_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            let hot_threads = kernels::get_num_threads();
            let spare_cpus = kernels::get_num_cpus().saturating_sub(hot_threads);
            hot_threads + spare_cpus.min(2)
        });
    base.clamp(1, n_splits.max(1))
}

/// Decode independent segments concurrently. Each of `k` workers runs its
/// kernels single-threaded (`set_thread_override(1)`) and owns its session
/// buffers, so `k` segments occupy `k` cores with no shared thread-pool
/// barriers — the in-process analogue of L2's independent processes, but
/// sharing one immutable weight set. Segments are assigned round-robin for load
/// balance; results are reassembled in segment order so the transcript (and the
/// streamed `token_cb` byte order) is identical to the serial path.
fn transcribe_splits_parallel(
    ctx: &mut QwenCtx,
    audio_samples: &[f32],
    splits: &[usize],
    tokenizer: &QwenTokenizer,
    k: usize,
) -> Option<String> {
    let n_splits = splits.len();
    let min_samples = SAMPLE_RATE as usize / 2;

    // Per-segment owned sample buffers (mirrors the serial short-segment
    // zero-padding). Owned so each worker can borrow its input across the scope.
    let mut seg_inputs: Vec<Vec<f32>> = Vec::with_capacity(n_splits);
    for s in 0..n_splits {
        let seg_start = splits[s];
        let seg_end = if s + 1 < n_splits {
            splits[s + 1]
        } else {
            audio_samples.len()
        };
        let seg_samples = seg_end - seg_start;
        let buf = if seg_samples < min_samples {
            let mut b = vec![0.0f32; min_samples];
            b[..seg_samples].copy_from_slice(&audio_samples[seg_start..seg_end]);
            b
        } else {
            audio_samples[seg_start..seg_end].to_vec()
        };
        seg_inputs.push(buf);
    }

    if kernels::verbose() >= 2 {
        eprintln!(
            "Parallel segments: {} segments across {} single-threaded workers",
            n_splits, k
        );
    }

    let cfg = ctx.config.clone();
    let weights = SegWeights {
        cfg: &cfg,
        encoder: &ctx.encoder,
        decoder: &ctx.decoder,
        prompt_tokens: ctx.prompt_tokens.as_deref(),
        force_prompt_tokens: ctx.force_prompt_tokens.as_deref(),
        has_prefilled_asr_text: has_prefilled_asr_text(ctx),
    };

    struct SegOut {
        core: SegCore,
        streamed: String,
    }

    let weights_ref = &weights;
    let seg_inputs_ref = &seg_inputs;

    let region_t0 = get_time_ms();
    let collected: Vec<(usize, SegOut)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(k);
        for w in 0..k {
            let indices: Vec<usize> = (w..n_splits).step_by(k).collect();
            let handle = scope.spawn(move || {
                // Force every kernel this worker invokes onto the inline
                // single-thread path so the K workers don't oversubscribe or
                // barrier-share the one global pool.
                kernels::set_thread_override(1);
                let mut session = SegSession::new(weights_ref.cfg);
                let mut out: Vec<(usize, SegOut)> = Vec::with_capacity(indices.len());
                for idx in indices {
                    let mut streamed = String::new();
                    let mut on_piece = |bytes: &[u8]| {
                        streamed.push_str(&String::from_utf8_lossy(bytes));
                    };
                    let core = decode_segment_core(
                        weights_ref,
                        &mut session.kv_cache,
                        &mut session.rope_cache,
                        &mut session.dec_bufs,
                        &mut session.enc_bufs,
                        tokenizer,
                        &seg_inputs_ref[idx],
                        None,
                        &mut on_piece,
                    );
                    if let Some(core) = core {
                        out.push((idx, SegOut { core, streamed }));
                    }
                }
                kernels::set_thread_override(0);
                out
            });
            handles.push(handle);
        }
        let mut all: Vec<(usize, SegOut)> = Vec::with_capacity(n_splits);
        for handle in handles {
            match handle.join() {
                Ok(mut v) => all.append(&mut v),
                Err(e) => std::panic::resume_unwind(e),
            }
        }
        all
    });
    let region_ms = elapsed_ms(region_t0);

    // Reassemble into segment-indexed slots (None = segment returned nothing).
    let mut slots: Vec<Option<SegOut>> = (0..n_splits).map(|_| None).collect();
    for (idx, out) in collected {
        slots[idx] = Some(out);
    }

    // Ordered assembly — byte-for-byte equivalent to the serial loop. The
    // streamed pieces (step 1) are what the serial path emits live through
    // `token_cb` during each segment's decode; replaying them in segment order
    // reproduces the exact stdout byte stream.
    let mut result = String::new();
    for slot in slots.into_iter() {
        let out = match slot {
            Some(o) => o,
            None => continue,
        };
        if !out.streamed.is_empty() {
            if let Some(ref cb) = ctx.token_cb {
                cb(&out.streamed);
            }
        }
        apply_seg_core(ctx, &out.core);

        let seg_text = out.core.text;
        if seg_text.is_empty() {
            continue;
        }
        let need_space = if !result.is_empty() {
            let prev = *result.as_bytes().last().unwrap_or(&0);
            let next = *seg_text.as_bytes().first().unwrap_or(&0);
            should_insert_boundary_space(prev, next)
        } else {
            false
        };
        if need_space {
            result.push(' ');
            if let Some(ref cb) = ctx.token_cb {
                cb(" ");
            }
        }
        if let Some(ref cb) = ctx.token_cb {
            if ctx.past_text_conditioning {
                cb(&seg_text);
            }
        }
        result.push_str(&seg_text);
    }

    // Report the parallel region's wall time as the inference time: the honest
    // latency of decoding all segments (per-segment `encode_ms`/`decode_ms`
    // remain component sums and now overlap in wall time).
    ctx.perf_total_ms += region_ms;
    Some(result)
}

/// Convenience wrapper: load a WAV file and transcribe it.
///
/// Equivalent to [`audio::load_wav`] followed by
/// [`transcribe_audio`].
pub fn transcribe(ctx: &mut QwenCtx, wav_path: &str) -> Option<String> {
    let samples = audio::load_wav(wav_path)?;
    transcribe_audio(ctx, &samples)
}

/// Transcribe from stdin.
pub fn transcribe_stdin(ctx: &mut QwenCtx) -> Option<String> {
    let samples = audio::read_pcm_stdin()?;
    transcribe_audio(ctx, &samples)
}

/// Streaming transcription: processes audio in chunks, emitting tokens via
/// `ctx.token_cb` as they become stable.
///
/// Trades throughput for lower latency compared to offline mode. If no
/// `token_cb` is set, falls back to a single offline decode of the full audio.
pub fn transcribe_stream(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 {
        ctx.stream_max_new_tokens
    } else {
        32
    };

    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    // If no token callback, fall back to offline decode
    if ctx.token_cb.is_none() {
        if kernels::verbose() >= 2 {
            eprintln!("Streaming: no token callback, using direct final refinement");
        }
        let audio_samples = if ctx.skip_silence {
            audio::compact_silence(samples)
        } else {
            samples.to_vec()
        };
        let tokenizer = load_tokenizer(&ctx.model_dir)?;
        ctx.prepare_prompt_tokens(&tokenizer);
        let (text, _) = transcribe_segment(ctx, &audio_samples, &tokenizer, None)?;
        return Some(text);
    }

    let audio_samples = if ctx.skip_silence {
        audio::compact_silence_fast(samples)
    } else {
        samples.to_vec()
    };

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;

    let tok_emb = ctx.decoder.tok_embeddings_bf16;

    let mut raw_tokens: Vec<i32> = Vec::new();
    let mut stable_text_tokens: Vec<i32> = Vec::new();
    let mut result_bytes: Vec<u8> = Vec::new();
    let mut tmp_embed = vec![0.0f32; dim];

    let mut chunk_idx = 0i32;
    let mut audio_cursor = 0usize;

    // Encoder window cache
    struct EncWindow {
        seq_len: usize,
        enc_output: Vec<f32>,
        row_keys: Vec<PrefillRowKey>,
    }
    let mut enc_cache: Vec<EncWindow> = Vec::new();
    let mut enc_cached_seq_total = 0usize;

    // Previous prefill row keys for LCP reuse
    let mut prev_prefill_keys: Vec<PrefillRowKey> = Vec::new();

    // Streaming robustness state
    let mut stale_count = 0i32;
    let mut prev_tail_snapshot: Vec<i32> = Vec::new();
    let mut enc_cache_base_windows = 0usize;

    while audio_cursor < audio_samples.len() {
        let chunk_t0 = get_time_ms();
        audio_cursor = (audio_cursor + chunk_samples).min(audio_samples.len());
        let is_final = audio_cursor >= audio_samples.len();

        // Encoder
        let t0 = get_time_ms();
        let full_end = (audio_cursor / enc_window_samples) * enc_window_samples;

        // Cache completed windows (base offset accounts for windows cleared on re-anchor)
        while (enc_cache_base_windows + enc_cache.len()) * enc_window_samples < full_end {
            let ws = (enc_cache_base_windows + enc_cache.len()) * enc_window_samples;
            let (mel, mel_frames) =
                audio::mel_spectrogram(&audio_samples[ws..ws + enc_window_samples])?;
            let (win_enc, win_seq) =
                ctx.encoder
                    .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
            let row_keys = prefill_embed_keys(&win_enc, win_seq, dim);
            enc_cached_seq_total += win_seq;
            enc_cache.push(EncWindow {
                seq_len: win_seq,
                enc_output: win_enc,
                row_keys,
            });
        }

        // Encode partial tail
        let mut partial_seq = 0;
        let mut partial_enc: Vec<f32> = Vec::new();
        let mut partial_keys: Vec<PrefillRowKey> = Vec::new();
        if is_final && full_end < audio_cursor {
            let _partial_samples = audio_cursor - full_end;
            if let Some((mel, mel_frames)) =
                audio::mel_spectrogram(&audio_samples[full_end..audio_cursor])
            {
                if let Some((enc, seq)) =
                    ctx.encoder
                        .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))
                {
                    partial_seq = seq;
                    partial_keys = prefill_embed_keys(&enc, seq, dim);
                    partial_enc = enc;
                }
            }
        }

        let enc_seq_len = enc_cached_seq_total + partial_seq;
        if enc_seq_len == 0 {
            chunk_idx += 1;
            continue;
        }

        let enc_ms = elapsed_ms(t0);
        ctx.perf_encode_ms += enc_ms;

        if !is_final && chunk_idx < unfixed_chunks {
            prev_prefill_keys.clear();
            ctx.perf_total_ms += elapsed_ms(chunk_t0);
            chunk_idx += 1;
            continue;
        }

        // Prefix rollback
        let n_prefix_tokens = if ctx.past_text_conditioning
            && chunk_idx >= unfixed_chunks
            && !raw_tokens.is_empty()
        {
            (raw_tokens.len() as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // Build input embeddings
        let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
        let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
        let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
        let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
        let total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;

        let mut input_embeds = vec![0.0f32; total_seq * dim];
        let mut prefill_keys = vec![PrefillRowKey { a: 0, b: 0 }; total_seq];
        let mut off = 0;

        for &tok in PREFIX_HEAD {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }
        if let Some(ref ptoks) = ctx.prompt_tokens {
            for &tok in ptoks {
                unsafe {
                    tok_embed_bf16_to_f32(
                        &mut input_embeds[off * dim..(off + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    )
                };
                prefill_keys[off] = prefill_token_key(tok);
                off += 1;
            }
        }
        for &tok in PREFIX_TAIL {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }

        let mut enc_key_off = 0;
        for w in &enc_cache {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + w.seq_len]
                .copy_from_slice(&w.row_keys);
            enc_key_off += w.seq_len;
        }
        if partial_seq > 0 {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + partial_seq]
                .copy_from_slice(&partial_keys);
        }

        let mut enc_embed_off = 0;
        for w in &enc_cache {
            let n = w.seq_len * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&w.enc_output);
            enc_embed_off += w.seq_len;
        }
        if partial_seq > 0 {
            let n = partial_seq * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&partial_enc);
        }

        let suffix_off = prefix_len + enc_seq_len;
        for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                )
            };
            prefill_keys[suffix_off + i] = prefill_token_key(tok);
        }
        if let Some(ref ftoks) = ctx.force_prompt_tokens {
            for (i, &tok) in ftoks.iter().enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(
                        &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                            ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    )
                };
                prefill_keys[suffix_off + SUFFIX_BASE.len() + i] = prefill_token_key(tok);
            }
        }

        let text_off = suffix_off + suffix_len;
        for i in 0..n_prefix_tokens {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
                    tok_emb,
                    raw_tokens[i],
                    dim,
                )
            };
            prefill_keys[text_off + i] = prefill_token_key(raw_tokens[i]);
        }

        // Decoder prefill with LCP reuse
        let t0 = get_time_ms();
        let prefill_len = total_seq - 1;

        let reused_prefill = prefill_lcp_len(&prev_prefill_keys, &prefill_keys, prefill_len);

        ctx.kv_cache.len = reused_prefill;
        let delta_prefill = prefill_len - reused_prefill;
        if delta_prefill > 0 {
            decoder::decoder_prefill(
                &ctx.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &input_embeds[reused_prefill * dim..],
                delta_prefill,
            );
        }

        // Save for next chunk
        prev_prefill_keys.clear();
        prev_prefill_keys.extend_from_slice(&prefill_keys[..prefill_len]);

        let prefill_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += prefill_ms;

        let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
        let mut token = decoder::decoder_forward(
            &ctx.decoder,
            &cfg,
            &mut ctx.kv_cache,
            &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            last_embed,
        );

        // Autoregressive decode
        let t0 = get_time_ms();
        let mut chunk_tokens: Vec<i32> = Vec::new();
        let mut n_generated = 0;

        while n_generated < max_new_tokens {
            n_generated += 1;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                break;
            }
            chunk_tokens.push(token);
            unsafe {
                tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim);
            }
            token = decoder::decoder_forward(
                &ctx.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &tmp_embed,
            );
        }

        let decode_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += decode_ms;

        // Update raw token history
        raw_tokens.truncate(n_prefix_tokens);
        raw_tokens.extend_from_slice(&chunk_tokens);

        // Streaming degeneracy detection
        if raw_tokens == prev_tail_snapshot {
            stale_count += 1;
        } else {
            stale_count = 0;
            prev_tail_snapshot = raw_tokens.clone();
        }
        let (best_reps, _) = stream_tail_repeat_blocks(&raw_tokens, STREAM_DEGEN_MAX_PERIOD);
        let is_degen = stale_count >= STREAM_STALE_CHUNKS || best_reps >= STREAM_DEGEN_MIN_REPEATS;

        if is_degen {
            if kernels::verbose() >= 2 {
                eprintln!(
                    "[stream degen] reset at chunk {} (stale={}, reps={})",
                    chunk_idx, stale_count, best_reps
                );
            }
            let carry = stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = stable_text_tokens.len() - carry;
            raw_tokens.clear();
            if carry > 0 {
                raw_tokens.push(TOKEN_ASR_TEXT);
                raw_tokens.extend_from_slice(&stable_text_tokens[carry_start..]);
            }
            prev_prefill_keys.clear();
            stale_count = 0;
            prev_tail_snapshot.clear();
        }

        // Periodic re-anchor: reset context every STREAM_RESET_INTERVAL_CHUNKS chunks
        if chunk_idx > 0 && chunk_idx % STREAM_RESET_INTERVAL_CHUNKS == 0 {
            if kernels::verbose() >= 2 {
                eprintln!("[stream reanchor] at chunk {}", chunk_idx);
            }
            let carry = stable_text_tokens.len().min(STREAM_RESET_CARRY_TOKENS);
            let carry_start = stable_text_tokens.len() - carry;
            raw_tokens.clear();
            if carry > 0 {
                raw_tokens.push(TOKEN_ASR_TEXT);
                raw_tokens.extend_from_slice(&stable_text_tokens[carry_start..]);
            }
            prev_prefill_keys.clear();
            stale_count = 0;
            prev_tail_snapshot.clear();
            enc_cache_base_windows += enc_cache.len();
            enc_cache.clear();
            enc_cached_seq_total = 0;
        }

        // Parse text region
        let text_start = if !has_prefilled_asr_text(ctx) {
            raw_tokens
                .iter()
                .position(|&t| t == TOKEN_ASR_TEXT)
                .map(|p| p + 1)
                .unwrap_or(0)
        } else {
            0
        };
        let n_text_tokens = raw_tokens.len().saturating_sub(text_start);

        // Fixed frontier
        let candidate_len = if is_final {
            n_text_tokens
        } else if chunk_idx >= unfixed_chunks {
            (n_text_tokens as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // Monotonic commit
        let candidate_tokens = &raw_tokens[text_start..];
        let _lcp = stable_text_tokens
            .iter()
            .zip(candidate_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let emit_from = stable_text_tokens.len();
        let emit_to = candidate_len.max(emit_from);

        for i in emit_from..emit_to {
            if i < candidate_tokens.len() {
                if i >= stable_text_tokens.len() {
                    stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(&String::from_utf8_lossy(piece_bytes));
                }
                ctx.perf_text_tokens += 1;
                result_bytes.extend_from_slice(piece_bytes);
            }
        }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        chunk_idx += 1;
    }

    Some(String::from_utf8_lossy(&result_bytes).trim().to_string())
} // ========================================================================
  // Incremental Streaming API
  // ========================================================================

/// Encoder window cached output.
struct EncWindow {
    seq_len: usize,
    enc_output: Vec<f32>,
    row_keys: Vec<PrefillRowKey>,
}

/// Persistent state for incremental streaming transcription.
///
/// Create once, then call [`stream_push_audio`] each time new audio
/// arrives.  The state keeps encoder caches, token history, and decoder
/// prefill row keys so that only *new* work is performed per call.
pub struct StreamState {
    // Encoder
    enc_cache: Vec<EncWindow>,
    enc_cached_seq_total: usize,
    enc_cache_base_windows: usize,

    // Decoder token history
    raw_tokens: Vec<i32>,
    stable_text_tokens: Vec<i32>,
    result_bytes: Vec<u8>,

    // Prefill LCP reuse
    prev_prefill_keys: Vec<PrefillRowKey>,

    // Streaming robustness
    prev_tail_snapshot: Vec<i32>,
    stale_count: i32,

    // Lazy partial encoding: skip re-encoding every other chunk
    last_partial_cursor: usize,
    last_partial_enc: Vec<f32>,
    last_partial_keys: Vec<PrefillRowKey>,
    last_partial_seq: usize,

    // Audio cursor
    audio_cursor: usize,
    chunk_idx: i32,

    // Tokenizer (loaded once)
    tokenizer: Option<QwenTokenizer>,
    prompt_prepared: bool,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamState {
    /// Create a new empty streaming state.
    pub fn new() -> Self {
        StreamState {
            enc_cache: Vec::new(),
            enc_cached_seq_total: 0,
            enc_cache_base_windows: 0,
            raw_tokens: Vec::new(),
            stable_text_tokens: Vec::new(),
            result_bytes: Vec::new(),
            prev_prefill_keys: Vec::new(),
            prev_tail_snapshot: Vec::new(),
            stale_count: 0,
            last_partial_cursor: 0,
            last_partial_enc: Vec::new(),
            last_partial_keys: Vec::new(),
            last_partial_seq: 0,
            audio_cursor: 0,
            chunk_idx: 0,
            tokenizer: None,
            prompt_prepared: false,
        }
    }

    /// Reset state for a new streaming window (e.g., after 30s limit).
    pub fn reset(&mut self) {
        self.enc_cache.clear();
        self.enc_cached_seq_total = 0;
        self.enc_cache_base_windows = 0;
        self.raw_tokens.clear();
        self.stable_text_tokens.clear();
        self.result_bytes.clear();
        self.prev_prefill_keys.clear();
        self.prev_tail_snapshot.clear();
        self.stale_count = 0;
        self.last_partial_cursor = 0;
        self.last_partial_enc.clear();
        self.last_partial_keys.clear();
        self.last_partial_seq = 0;
        self.audio_cursor = 0;
        self.chunk_idx = 0;
        // Keep tokenizer and prompt_prepared
    }

    /// Get the current stable transcription result.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.result_bytes).into_owned()
    }

    /// Get how many samples have been processed so far.
    pub fn audio_cursor(&self) -> usize {
        self.audio_cursor
    }
}

/// Process all available new audio incrementally.
///
/// `samples` is the **full** audio buffer accumulated so far (16 kHz mono f32).
/// Processes all full chunks from `state.audio_cursor` to end of `samples`.
/// When `finalize` is true, also processes any remaining partial chunk and
/// emits all rollback-buffered tokens.
/// Returns the newly emitted text delta (if any).
pub fn stream_push_audio(
    ctx: &mut QwenCtx,
    samples: &[f32],
    state: &mut StreamState,
    finalize: bool,
) -> Option<String> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 {
        ctx.stream_max_new_tokens
    } else {
        32
    };

    // Lazy-init tokenizer
    if state.tokenizer.is_none() {
        state.tokenizer = load_tokenizer(&ctx.model_dir);
    }
    let tokenizer = state.tokenizer.as_ref()?;

    if !state.prompt_prepared {
        if !ctx.prepare_prompt_tokens(tokenizer) {
            return None;
        }
        state.prompt_prepared = true;
    }

    // Check if we have enough audio for at least one chunk (or finalizing)
    let available = samples.len().saturating_sub(state.audio_cursor);
    if available < chunk_samples && !finalize {
        return Some(String::new());
    }
    if available == 0 {
        return Some(String::new());
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;
    let tok_emb = ctx.decoder.tok_embeddings_bf16;
    let mut tmp_embed = vec![0.0f32; dim];
    let mut delta_bytes: Vec<u8> = Vec::new();

    // ---- Process full chunks, plus remainder if finalizing ----
    while state.audio_cursor < samples.len() {
        let remaining = samples.len() - state.audio_cursor;
        if remaining < chunk_samples && !finalize {
            break; // Wait for more audio
        }

        let chunk_t0 = get_time_ms();
        state.audio_cursor = (state.audio_cursor + chunk_samples).min(samples.len());
        let is_final = finalize && state.audio_cursor >= samples.len();

        // ---- Encoder: only encode new windows ----
        let t0 = get_time_ms();
        let full_end = (state.audio_cursor / enc_window_samples) * enc_window_samples;

        // Cache newly completed windows (base offset accounts for windows cleared on re-anchor)
        while (state.enc_cache_base_windows + state.enc_cache.len()) * enc_window_samples < full_end
        {
            let ws = (state.enc_cache_base_windows + state.enc_cache.len()) * enc_window_samples;
            let (mel, mel_frames) = audio::mel_spectrogram(&samples[ws..ws + enc_window_samples])?;
            let (win_enc, win_seq) =
                ctx.encoder
                    .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))?;
            let row_keys = prefill_embed_keys(&win_enc, win_seq, dim);
            state.enc_cached_seq_total += win_seq;
            state.enc_cache.push(EncWindow {
                seq_len: win_seq,
                enc_output: win_enc,
                row_keys,
            });
        }

        // Encode partial tail — with lazy re-encoding for LCP optimization.
        // Only re-encode when enough new audio has accumulated (every 2 chunks),
        // on the first chunk, or when finalizing. On skip chunks, the reused
        // encoder output gives near-perfect LCP matching, cutting prefill cost.
        let enc_update_threshold = chunk_samples * 2;
        let partial_age = state.audio_cursor.saturating_sub(state.last_partial_cursor);
        let need_encode =
            state.last_partial_cursor == 0 || partial_age >= enc_update_threshold || is_final;

        let partial_seq;
        let partial_enc;
        let partial_keys;
        if need_encode && full_end < state.audio_cursor {
            if let Some((mel, mel_frames)) =
                audio::mel_spectrogram(&samples[full_end..state.audio_cursor])
            {
                if let Some((enc, seq)) =
                    ctx.encoder
                        .forward(&cfg, &mel, mel_frames, Some(&mut ctx.enc_bufs))
                {
                    partial_seq = seq;
                    partial_keys = prefill_embed_keys(&enc, seq, dim);
                    partial_enc = enc;
                    state.last_partial_cursor = state.audio_cursor;
                    state.last_partial_enc = partial_enc.clone();
                    state.last_partial_keys = partial_keys.clone();
                    state.last_partial_seq = partial_seq;
                } else {
                    partial_seq = state.last_partial_seq;
                    partial_enc = state.last_partial_enc.clone();
                    partial_keys = state.last_partial_keys.clone();
                }
            } else {
                partial_seq = state.last_partial_seq;
                partial_enc = state.last_partial_enc.clone();
                partial_keys = state.last_partial_keys.clone();
            }
        } else if full_end < state.audio_cursor {
            // Reuse previous partial encoding (skip chunk)
            partial_seq = state.last_partial_seq;
            partial_enc = state.last_partial_enc.clone();
            partial_keys = state.last_partial_keys.clone();
        } else {
            partial_seq = 0;
            partial_enc = Vec::new();
            partial_keys = Vec::new();
        }

        let enc_seq_len = state.enc_cached_seq_total + partial_seq;
        if enc_seq_len == 0 {
            state.chunk_idx += 1;
            return Some(String::new());
        }

        let enc_ms = elapsed_ms(t0);
        ctx.perf_encode_ms += enc_ms;

        // ---- Prefix rollback ----
        let n_prefix_tokens = if ctx.past_text_conditioning
            && state.chunk_idx >= unfixed_chunks
            && !state.raw_tokens.is_empty()
        {
            (state.raw_tokens.len() as i32 - rollback).max(0) as usize
        } else {
            0
        };

        // ---- Build input embeddings ----
        let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
        let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
        let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
        let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
        let total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;

        let mut input_embeds = vec![0.0f32; total_seq * dim];
        let mut prefill_keys = vec![PrefillRowKey { a: 0, b: 0 }; total_seq];
        let mut off = 0;

        for &tok in PREFIX_HEAD {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                );
            }
            prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }
        if let Some(ref ptoks) = ctx.prompt_tokens {
            for &tok in ptoks {
                unsafe {
                    tok_embed_bf16_to_f32(
                        &mut input_embeds[off * dim..(off + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    );
                }
                prefill_keys[off] = prefill_token_key(tok);
                off += 1;
            }
        }
        for &tok in PREFIX_TAIL {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[off * dim..(off + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                );
            }
            prefill_keys[off] = prefill_token_key(tok);
            off += 1;
        }

        let mut enc_key_off = 0;
        for w in &state.enc_cache {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + w.seq_len]
                .copy_from_slice(&w.row_keys);
            enc_key_off += w.seq_len;
        }
        if partial_seq > 0 {
            prefill_keys[prefix_len + enc_key_off..prefix_len + enc_key_off + partial_seq]
                .copy_from_slice(&partial_keys);
        }

        let mut enc_embed_off = 0;
        for w in &state.enc_cache {
            let n = w.seq_len * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&w.enc_output);
            enc_embed_off += w.seq_len;
        }
        if partial_seq > 0 {
            let n = partial_seq * dim;
            input_embeds
                [(prefix_len + enc_embed_off) * dim..(prefix_len + enc_embed_off) * dim + n]
                .copy_from_slice(&partial_enc);
        }

        let suffix_off = prefix_len + enc_seq_len;
        for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                    tok_emb,
                    tok,
                    dim,
                );
            }
            prefill_keys[suffix_off + i] = prefill_token_key(tok);
        }
        if let Some(ref ftoks) = ctx.force_prompt_tokens {
            for (i, &tok) in ftoks.iter().enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(
                        &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                            ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                        tok_emb,
                        tok,
                        dim,
                    );
                }
                prefill_keys[suffix_off + SUFFIX_BASE.len() + i] = prefill_token_key(tok);
            }
        }

        let text_off = suffix_off + suffix_len;
        for i in 0..n_prefix_tokens {
            unsafe {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
                    tok_emb,
                    state.raw_tokens[i],
                    dim,
                );
            }
            prefill_keys[text_off + i] = prefill_token_key(state.raw_tokens[i]);
        }

        // ---- Decoder prefill with LCP reuse ----
        let t0 = get_time_ms();
        let prefill_len = total_seq - 1;

        let reused_prefill = prefill_lcp_len(&state.prev_prefill_keys, &prefill_keys, prefill_len);

        ctx.kv_cache.len = reused_prefill;
        let delta_prefill = prefill_len - reused_prefill;
        if delta_prefill > 0 {
            decoder::decoder_prefill(
                &ctx.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &input_embeds[reused_prefill * dim..],
                delta_prefill,
            );
        }

        let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
        let mut token = decoder::decoder_forward(
            &ctx.decoder,
            &cfg,
            &mut ctx.kv_cache,
            &mut ctx.rope_cache,
            &mut ctx.dec_bufs,
            last_embed,
        );

        // Save for next chunk
        state.prev_prefill_keys.clear();
        state
            .prev_prefill_keys
            .extend_from_slice(&prefill_keys[..prefill_len]);

        let prefill_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += prefill_ms;

        if kernels::verbose() >= 2 {
            eprintln!(
                "  [stream chunk {}] encoder: {:.0}ms, prefill: {}/{} reused ({:.0}ms, delta={})",
                state.chunk_idx, enc_ms, reused_prefill, prefill_len, prefill_ms, delta_prefill
            );
        }

        // ---- Autoregressive decode ----
        let t0 = get_time_ms();
        let mut chunk_tokens: Vec<i32> = Vec::new();
        let mut n_generated = 0;

        while n_generated < max_new_tokens {
            n_generated += 1;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                break;
            }
            chunk_tokens.push(token);
            unsafe {
                tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim);
            }
            token = decoder::decoder_forward(
                &ctx.decoder,
                &cfg,
                &mut ctx.kv_cache,
                &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &tmp_embed,
            );
        }

        let decode_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += decode_ms;

        // ---- Detect speech end (decoder produced EOT on silence) ----
        // When chunk_tokens is empty, the decoder saw silence/end-of-speech.
        // Commit ALL remaining rollback-buffered tokens BEFORE truncation,
        // since truncate will remove them.
        let speech_ended = chunk_tokens.is_empty()
            && !state.raw_tokens.is_empty()
            && state.chunk_idx >= unfixed_chunks;

        if speech_ended {
            // Emit remaining rollback tokens from current raw_tokens (before truncation)
            let text_start = if !has_prefilled_asr_text(ctx) {
                state
                    .raw_tokens
                    .iter()
                    .position(|&t| t == TOKEN_ASR_TEXT)
                    .map(|p| p + 1)
                    .unwrap_or(0)
            } else {
                0
            };
            let candidate_tokens = &state.raw_tokens[text_start..];
            let n_text = candidate_tokens.len();
            let emit_from = state.stable_text_tokens.len();
            for i in emit_from..n_text {
                if i < candidate_tokens.len() {
                    if i >= state.stable_text_tokens.len() {
                        state.stable_text_tokens.push(candidate_tokens[i]);
                    }
                    let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                    if let Some(ref cb) = ctx.token_cb {
                        cb(&String::from_utf8_lossy(piece_bytes));
                    }
                    ctx.perf_text_tokens += 1;
                    state.result_bytes.extend_from_slice(piece_bytes);
                    delta_bytes.extend_from_slice(piece_bytes);
                }
            }
        }

        // ---- Update raw token history ----
        state.raw_tokens.truncate(n_prefix_tokens);
        state.raw_tokens.extend_from_slice(&chunk_tokens);

        // ---- Streaming degeneracy detection ----
        if !speech_ended {
            if state.raw_tokens == state.prev_tail_snapshot {
                state.stale_count += 1;
            } else {
                state.stale_count = 0;
                state.prev_tail_snapshot = state.raw_tokens.clone();
            }
            let (best_reps, _) =
                stream_tail_repeat_blocks(&state.raw_tokens, STREAM_DEGEN_MAX_PERIOD);
            let is_degen =
                state.stale_count >= STREAM_STALE_CHUNKS || best_reps >= STREAM_DEGEN_MIN_REPEATS;

            if is_degen {
                if kernels::verbose() >= 2 {
                    eprintln!(
                        "[stream degen] reset at chunk {} (stale={}, reps={})",
                        state.chunk_idx, state.stale_count, best_reps
                    );
                }
                let carry = state
                    .stable_text_tokens
                    .len()
                    .min(STREAM_RESET_CARRY_TOKENS);
                let carry_start = state.stable_text_tokens.len() - carry;
                state.raw_tokens.clear();
                if carry > 0 {
                    state.raw_tokens.push(TOKEN_ASR_TEXT);
                    state
                        .raw_tokens
                        .extend_from_slice(&state.stable_text_tokens[carry_start..]);
                }
                state.prev_prefill_keys.clear();
                state.stale_count = 0;
                state.prev_tail_snapshot.clear();
                if state.enc_cache.len() >= STREAM_MAX_ENC_WINDOWS {
                    state.enc_cache_base_windows += state.enc_cache.len();
                    state.enc_cache.clear();
                    state.enc_cached_seq_total = 0;
                }
            }

            // Periodic re-anchor: reset context every STREAM_RESET_INTERVAL_CHUNKS chunks
            if state.chunk_idx > 0 && state.chunk_idx % STREAM_RESET_INTERVAL_CHUNKS == 0 {
                if kernels::verbose() >= 2 {
                    eprintln!("[stream reanchor] at chunk {}", state.chunk_idx);
                }
                let carry = state
                    .stable_text_tokens
                    .len()
                    .min(STREAM_RESET_CARRY_TOKENS);
                let carry_start = state.stable_text_tokens.len() - carry;
                state.raw_tokens.clear();
                if carry > 0 {
                    state.raw_tokens.push(TOKEN_ASR_TEXT);
                    state
                        .raw_tokens
                        .extend_from_slice(&state.stable_text_tokens[carry_start..]);
                }
                state.prev_prefill_keys.clear();
                state.stale_count = 0;
                state.prev_tail_snapshot.clear();
                if state.enc_cache.len() >= STREAM_MAX_ENC_WINDOWS {
                    state.enc_cache_base_windows += state.enc_cache.len();
                    state.enc_cache.clear();
                    state.enc_cached_seq_total = 0;
                }
            }
        }

        // ---- Parse text region and emit stable tokens (non-speech-ended case) ----
        if !speech_ended {
            let text_start = if !has_prefilled_asr_text(ctx) {
                state
                    .raw_tokens
                    .iter()
                    .position(|&t| t == TOKEN_ASR_TEXT)
                    .map(|p| p + 1)
                    .unwrap_or(0)
            } else {
                0
            };
            let n_text_tokens = state.raw_tokens.len().saturating_sub(text_start);

            let candidate_len = if is_final {
                n_text_tokens
            } else if state.chunk_idx >= unfixed_chunks {
                (n_text_tokens as i32 - rollback).max(0) as usize
            } else {
                0
            };

            let candidate_tokens = &state.raw_tokens[text_start..];
            let emit_from = state.stable_text_tokens.len();
            let emit_to = candidate_len.max(emit_from);

            for i in emit_from..emit_to {
                if i < candidate_tokens.len() {
                    if i >= state.stable_text_tokens.len() {
                        state.stable_text_tokens.push(candidate_tokens[i]);
                    }
                    let piece_bytes = tokenizer.decode_bytes(candidate_tokens[i]);
                    if let Some(ref cb) = ctx.token_cb {
                        cb(&String::from_utf8_lossy(piece_bytes));
                    }
                    ctx.perf_text_tokens += 1;
                    state.result_bytes.extend_from_slice(piece_bytes);
                    delta_bytes.extend_from_slice(piece_bytes);
                }
            }
        }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        state.chunk_idx += 1;

        // Stop processing after speech ends — no point encoding more silence
        if speech_ended {
            break;
        }
    } // end while loop

    Some(String::from_utf8_lossy(&delta_bytes).into_owned())
}
