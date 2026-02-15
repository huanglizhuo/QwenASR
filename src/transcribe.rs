/// Offline/segmented/streaming transcription orchestration.

use crate::audio;
use crate::config::*;
use crate::context::QwenCtx;
use crate::decoder::{self, tok_embed_bf16_to_f32};
use crate::kernels;
use crate::tokenizer::QwenTokenizer;

use std::time::Instant;

// Prompt token sequences
const PREFIX_HEAD: &[i32] = &[151644, 8948, 198];
const PREFIX_TAIL: &[i32] = &[151645, 198, 151644, 872, 198, 151669];
const SUFFIX_BASE: &[i32] = &[151670, 151645, 198, 151644, 77091, 198];

fn get_time_ms() -> f64 {
    // Use monotonic clock
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms(t0: f64) -> f64 {
    get_time_ms() - t0
}

fn load_tokenizer(model_dir: &str) -> Option<QwenTokenizer> {
    let vocab_path = format!("{}/vocab.json", model_dir);
    QwenTokenizer::load(&vocab_path)
}

/// Transcribe a single segment. Returns (text, n_text_tokens).
fn transcribe_segment(
    ctx: &mut QwenCtx,
    samples: &[f32],
    tokenizer: &QwenTokenizer,
    past_tokens: Option<&[i32]>,
) -> Option<(String, i32)> {
    let cfg = &ctx.config.clone();
    let dim = cfg.dec_hidden;
    let seg_t0 = get_time_ms();
    let mut n_text_tokens = 0i32;

    // Mel spectrogram
    let t0 = get_time_ms();
    let (mel, mel_frames) = audio::mel_spectrogram(samples)?;
    let mel_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Mel: {} frames ({:.0} ms)", mel_frames, mel_ms);
    }

    // Encoder
    let t0 = get_time_ms();
    let (enc_output, enc_seq_len) = ctx.encoder.forward(cfg, &mel, mel_frames)?;
    let enc_ms = elapsed_ms(t0);

    if kernels::verbose() >= 2 {
        eprintln!("  Encoder: {} tokens ({:.0} ms)", enc_seq_len, enc_ms);
    }

    if !ctx.prepare_prompt_tokens(tokenizer) {
        return None;
    }

    // Build input embeddings
    let n_prompt_tokens = ctx.prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_force_prompt_tokens = ctx.force_prompt_tokens.as_ref().map_or(0, |t| t.len());
    let n_past = past_tokens.map_or(0, |t| t.len());
    let n_past_prompt_tokens = if n_past > 0 { n_past + 1 } else { 0 }; // +1 for <asr_text>

    let prefix_len = PREFIX_HEAD.len() + n_prompt_tokens + PREFIX_TAIL.len();
    let suffix_len = SUFFIX_BASE.len() + n_force_prompt_tokens;
    let total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;

    let mut input_embeds = vec![0.0f32; total_seq * dim];
    let tok_emb = ctx.decoder.tok_embeddings_bf16;

    // Embed prefix head
    let mut off = 0;
    for &tok in PREFIX_HEAD {
        tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim);
        off += 1;
    }

    // Optional prompt
    if let Some(ref ptoks) = ctx.prompt_tokens {
        for &tok in ptoks {
            tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim);
            off += 1;
        }
    }

    // Prefix tail
    for &tok in PREFIX_TAIL {
        tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim);
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
        tok_embed_bf16_to_f32(
            &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
            tok_emb, tok, dim,
        );
    }

    // Force language tokens
    if let Some(ref ftoks) = ctx.force_prompt_tokens {
        for (i, &tok) in ftoks.iter().enumerate() {
            tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                    ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                tok_emb, tok, dim,
            );
        }
    }

    // Past text conditioning tokens
    let past_off = suffix_off + suffix_len;
    if let Some(ptoks) = past_tokens {
        for (i, &tok) in ptoks.iter().enumerate() {
            tok_embed_bf16_to_f32(
                &mut input_embeds[(past_off + i) * dim..(past_off + i + 1) * dim],
                tok_emb, tok, dim,
            );
        }
        tok_embed_bf16_to_f32(
            &mut input_embeds[(past_off + ptoks.len()) * dim..(past_off + ptoks.len() + 1) * dim],
            tok_emb, TOKEN_ASR_TEXT, dim,
        );
    }

    // Decoder prefill
    let t0 = get_time_ms();
    ctx.kv_cache.len = 0;
    let prefill_len = total_seq - 1;
    decoder::decoder_prefill(
        &ctx.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
        &mut ctx.dec_bufs, &input_embeds, prefill_len,
    );

    // First token from last prefill position
    let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
    let mut token = decoder::decoder_forward(
        &ctx.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
        &mut ctx.dec_bufs, last_embed,
    );

    let prefill_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!("  Prefill: {} tokens ({:.0} ms)", total_seq, prefill_ms);
    }

    // Autoregressive decode
    let t0 = get_time_ms();
    let max_tokens = 2048;
    let mut n_generated = 0;
    let mut past_asr_text = n_force_prompt_tokens > 0 || n_past > 0;

    let mut text = String::new();
    let mut tmp_embed = vec![0.0f32; dim];

    while n_generated < max_tokens {
        n_generated += 1;

        if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
            break;
        }

        if token == TOKEN_ASR_TEXT {
            past_asr_text = true;
        } else if past_asr_text {
            let piece = tokenizer.decode(token);
            text.push_str(piece);
            n_text_tokens += 1;

            if let Some(ref cb) = ctx.token_cb {
                cb(piece);
            }
        }

        tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim);
        token = decoder::decoder_forward(
            &ctx.decoder, cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
            &mut ctx.dec_bufs, &tmp_embed,
        );
    }

    let decode_ms = elapsed_ms(t0);
    if kernels::verbose() >= 2 {
        eprintln!(
            "  Decode: {} tokens ({:.0} ms, {:.1} ms/token)",
            n_generated, decode_ms,
            if n_generated > 0 { decode_ms / n_generated as f64 } else { 0.0 }
        );
    }

    // Trim whitespace
    let trimmed = text.trim().to_string();

    ctx.perf_total_ms += elapsed_ms(seg_t0);
    ctx.perf_text_tokens += n_text_tokens;
    ctx.perf_encode_ms += mel_ms + enc_ms;
    ctx.perf_decode_ms += prefill_ms + decode_ms;

    Some((trimmed, n_text_tokens))
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
        for j in pos..end {
            energy += samples[j] * samples[j];
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
    if prev_ch == 0 || next_ch == 0 { return false; }
    if (prev_ch as char).is_whitespace() { return false; }
    if (next_ch as char).is_whitespace() { return false; }
    if (next_ch as char).is_ascii_punctuation() { return false; }
    true
}

// ========================================================================
// Public API
// ========================================================================

/// Transcribe audio samples (offline or segmented).
pub fn transcribe_audio(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    let audio_samples = if ctx.skip_silence {
        let compacted = audio::compact_silence(samples);
        if kernels::verbose() >= 1 {
            let used_pct = 100.0 * compacted.len() as f32 / samples.len().max(1) as f32;
            eprintln!(
                "Silence skip: used {:.1}%, skipped {:.1}% ({} -> {} samples)",
                used_pct, 100.0 - used_pct, samples.len(), compacted.len()
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
    let mut splits = vec![0usize];
    let mut pos = 0;
    while pos + target_samples + margin_samples < audio_samples.len() {
        let split = find_split_point(&audio_samples, pos + target_samples, search);
        splits.push(split);
        pos = split;
        if splits.len() >= 127 { break; }
    }
    let n_splits = splits.len();

    if kernels::verbose() >= 2 {
        eprintln!("Splitting into {} segments", n_splits);
    }

    let mut result = String::new();
    let min_samples = SAMPLE_RATE as usize / 2;
    let use_past_text = ctx.past_text_conditioning;

    for s in 0..n_splits {
        let core_start = splits[s];
        let core_end = if s + 1 < n_splits { splits[s + 1] } else { audio_samples.len() };
        let seg_start = core_start;
        let seg_end = core_end;
        let seg_samples = seg_end - seg_start;

        if kernels::verbose() >= 2 {
            eprintln!(
                "Segment {}/{}: {:.1}-{:.1}s ({} samples)",
                s + 1, n_splits,
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

        let (seg_text, _seg_text_tokens) = match transcribe_segment(
            ctx, seg_ptr, &tokenizer, past_tokens.as_deref(),
        ) {
            Some(r) => r,
            None => continue,
        };

        if seg_text.is_empty() { continue; }

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

/// Transcribe a WAV file.
pub fn transcribe(ctx: &mut QwenCtx, wav_path: &str) -> Option<String> {
    let samples = audio::load_wav(wav_path)?;
    transcribe_audio(ctx, &samples)
}

/// Transcribe from stdin.
pub fn transcribe_stdin(ctx: &mut QwenCtx) -> Option<String> {
    let samples = audio::read_pcm_stdin()?;
    transcribe_audio(ctx, &samples)
}

/// Streaming transcription.
pub fn transcribe_stream(ctx: &mut QwenCtx, samples: &[f32]) -> Option<String> {
    let cfg = ctx.config.clone();
    let dim = cfg.dec_hidden;
    let chunk_samples = (ctx.stream_chunk_sec * SAMPLE_RATE as f32) as usize;
    let rollback = ctx.stream_rollback;
    let unfixed_chunks = ctx.stream_unfixed_chunks;
    let max_new_tokens = if ctx.stream_max_new_tokens > 0 { ctx.stream_max_new_tokens } else { 32 };

    let audio_samples = if ctx.skip_silence {
        audio::compact_silence(samples)
    } else {
        samples.to_vec()
    };

    ctx.reset_perf();
    ctx.perf_audio_ms = 1000.0 * samples.len() as f64 / SAMPLE_RATE as f64;

    // If no token callback, fall back to offline decode
    if ctx.token_cb.is_none() {
        if kernels::verbose() >= 2 {
            eprintln!("Streaming: no token callback, using direct final refinement");
        }
        let tokenizer = load_tokenizer(&ctx.model_dir)?;
        ctx.prepare_prompt_tokens(&tokenizer);
        let (text, _) = transcribe_segment(ctx, &audio_samples, &tokenizer, None)?;
        return Some(text);
    }

    let tokenizer = load_tokenizer(&ctx.model_dir)?;
    if !ctx.prepare_prompt_tokens(&tokenizer) {
        return None;
    }

    let enc_window_frames = cfg.enc_n_window_infer.clamp(100, 800);
    let enc_window_samples = enc_window_frames * HOP_LENGTH;

    let tok_emb = ctx.decoder.tok_embeddings_bf16;

    let mut raw_tokens: Vec<i32> = Vec::new();
    let mut stable_text_tokens: Vec<i32> = Vec::new();
    let mut result = String::new();
    let mut tmp_embed = vec![0.0f32; dim];

    let mut chunk_idx = 0i32;
    let mut audio_cursor = 0usize;

    // Encoder window cache
    struct EncWindow {
        seq_len: usize,
        enc_output: Vec<f32>,
    }
    let mut enc_cache: Vec<EncWindow> = Vec::new();
    let mut enc_cached_seq_total = 0usize;

    // Previous prefill embeddings for LCP reuse
    let mut prev_prefill_embeds: Vec<f32> = Vec::new();
    let mut prev_prefill_len = 0usize;

    while audio_cursor < audio_samples.len() {
        let chunk_t0 = get_time_ms();
        audio_cursor = (audio_cursor + chunk_samples).min(audio_samples.len());
        let is_final = audio_cursor >= audio_samples.len();

        // Encoder
        let t0 = get_time_ms();
        let full_end = (audio_cursor / enc_window_samples) * enc_window_samples;

        // Cache completed windows
        while enc_cache.len() * enc_window_samples < full_end {
            let ws = enc_cache.len() * enc_window_samples;
            let (mel, mel_frames) = audio::mel_spectrogram(&audio_samples[ws..ws + enc_window_samples])?;
            let (win_enc, win_seq) = ctx.encoder.forward(&cfg, &mel, mel_frames)?;
            enc_cached_seq_total += win_seq;
            enc_cache.push(EncWindow { seq_len: win_seq, enc_output: win_enc });
        }

        // Encode partial tail
        let mut partial_seq = 0;
        let mut partial_enc: Vec<f32> = Vec::new();
        if full_end < audio_cursor {
            let _partial_samples = audio_cursor - full_end;
            if let Some((mel, mel_frames)) = audio::mel_spectrogram(&audio_samples[full_end..audio_cursor]) {
                if let Some((enc, seq)) = ctx.encoder.forward(&cfg, &mel, mel_frames) {
                    partial_seq = seq;
                    partial_enc = enc;
                }
            }
        }

        let enc_seq_len = enc_cached_seq_total + partial_seq;
        if enc_seq_len == 0 {
            chunk_idx += 1;
            continue;
        }

        // Assemble encoder output
        let mut enc_output = vec![0.0f32; enc_seq_len * dim];
        let mut enc_off = 0;
        for w in &enc_cache {
            enc_output[enc_off * dim..(enc_off + w.seq_len) * dim]
                .copy_from_slice(&w.enc_output);
            enc_off += w.seq_len;
        }
        if partial_seq > 0 {
            enc_output[enc_off * dim..(enc_off + partial_seq) * dim]
                .copy_from_slice(&partial_enc);
        }

        let enc_ms = elapsed_ms(t0);
        ctx.perf_encode_ms += enc_ms;

        // Prefix rollback
        let n_prefix_tokens = if ctx.past_text_conditioning && chunk_idx >= unfixed_chunks && !raw_tokens.is_empty() {
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
        let mut off = 0;

        for &tok in PREFIX_HEAD {
            tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim);
            off += 1;
        }
        if let Some(ref ptoks) = ctx.prompt_tokens {
            for &tok in ptoks {
                tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim);
                off += 1;
            }
        }
        for &tok in PREFIX_TAIL {
            tok_embed_bf16_to_f32(&mut input_embeds[off * dim..(off + 1) * dim], tok_emb, tok, dim);
            off += 1;
        }

        for i in 0..enc_seq_len {
            input_embeds[(prefix_len + i) * dim..(prefix_len + i + 1) * dim]
                .copy_from_slice(&enc_output[i * dim..(i + 1) * dim]);
        }

        let suffix_off = prefix_len + enc_seq_len;
        for (i, &tok) in SUFFIX_BASE.iter().enumerate() {
            tok_embed_bf16_to_f32(
                &mut input_embeds[(suffix_off + i) * dim..(suffix_off + i + 1) * dim],
                tok_emb, tok, dim,
            );
        }
        if let Some(ref ftoks) = ctx.force_prompt_tokens {
            for (i, &tok) in ftoks.iter().enumerate() {
                tok_embed_bf16_to_f32(
                    &mut input_embeds[(suffix_off + SUFFIX_BASE.len() + i) * dim
                        ..(suffix_off + SUFFIX_BASE.len() + i + 1) * dim],
                    tok_emb, tok, dim,
                );
            }
        }

        let text_off = suffix_off + suffix_len;
        for i in 0..n_prefix_tokens {
            tok_embed_bf16_to_f32(
                &mut input_embeds[(text_off + i) * dim..(text_off + i + 1) * dim],
                tok_emb, raw_tokens[i], dim,
            );
        }

        // Decoder prefill with LCP reuse
        let t0 = get_time_ms();
        let prefill_len = total_seq - 1;

        let mut reused_prefill = 0;
        if !prev_prefill_embeds.is_empty() && prev_prefill_len > 0 {
            let cmp_len = prefill_len.min(prev_prefill_len);
            let _row_bytes = dim * std::mem::size_of::<f32>();
            while reused_prefill < cmp_len {
                let a = &prev_prefill_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
                let b = &input_embeds[reused_prefill * dim..(reused_prefill + 1) * dim];
                if a != b { break; }
                reused_prefill += 1;
            }
        }

        ctx.kv_cache.len = reused_prefill;
        let delta_prefill = prefill_len - reused_prefill;
        if delta_prefill > 0 {
            decoder::decoder_prefill(
                &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
                &mut ctx.dec_bufs,
                &input_embeds[reused_prefill * dim..],
                delta_prefill,
            );
        }

        let last_embed = &input_embeds[prefill_len * dim..(prefill_len + 1) * dim];
        let mut token = decoder::decoder_forward(
            &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
            &mut ctx.dec_bufs, last_embed,
        );

        // Save for next chunk
        prev_prefill_embeds = input_embeds[..prefill_len * dim].to_vec();
        prev_prefill_len = prefill_len;

        let prefill_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += prefill_ms;

        // Autoregressive decode
        let t0 = get_time_ms();
        let mut chunk_tokens: Vec<i32> = Vec::new();
        let mut n_generated = 0;

        while n_generated < max_new_tokens {
            n_generated += 1;
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END { break; }
            chunk_tokens.push(token);
            tok_embed_bf16_to_f32(&mut tmp_embed, tok_emb, token, dim);
            token = decoder::decoder_forward(
                &ctx.decoder, &cfg, &mut ctx.kv_cache, &mut ctx.rope_cache,
                &mut ctx.dec_bufs, &tmp_embed,
            );
        }

        let decode_ms = elapsed_ms(t0);
        ctx.perf_decode_ms += decode_ms;

        // Update raw token history
        raw_tokens.truncate(n_prefix_tokens);
        raw_tokens.extend_from_slice(&chunk_tokens);

        // Parse text region
        let text_start = if n_force_prompt_tokens == 0 {
            raw_tokens.iter().position(|&t| t == TOKEN_ASR_TEXT)
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
        let _lcp = stable_text_tokens.iter().zip(candidate_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let emit_from = stable_text_tokens.len();
        let emit_to = candidate_len.max(emit_from);

        for i in emit_from..emit_to {
            if i < candidate_tokens.len() {
                if i >= stable_text_tokens.len() {
                    stable_text_tokens.push(candidate_tokens[i]);
                }
                let piece = tokenizer.decode(candidate_tokens[i]);
                if let Some(ref cb) = ctx.token_cb {
                    cb(piece);
                }
                ctx.perf_text_tokens += 1;
                result.push_str(piece);
            }
        }

        ctx.perf_total_ms += elapsed_ms(chunk_t0);
        chunk_idx += 1;
    }

    Some(result.trim().to_string())
}
