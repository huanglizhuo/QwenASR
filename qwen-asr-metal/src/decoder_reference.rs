//! Development-only CPU intermediate oracle; never linked into GPU inference.
use qwen_asr::{audio, context::QwenCtx, decoder, kernels, tokenizer::QwenTokenizer};
use serde_json::json;
use std::{fs::File, io::Write, time::Instant};
fn write_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(File::create(path)?);
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: decoder-reference <model> <wav> <text-file> <output-prefix>".into());
    }
    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());
    let mut ctx = QwenCtx::load(&args[1]).ok_or("model load failed")?;
    let tok =
        QwenTokenizer::load(&format!("{}/vocab.json", args[1])).ok_or("tokenizer load failed")?;
    let pcm = audio::load_wav(&args[2]).ok_or("audio load failed")?;
    let (mel, frames) = audio::mel_spectrogram(&pcm).ok_or("mel failed")?;
    let (encoded, audio_tokens) = ctx
        .model
        .encoder
        .forward(&ctx.config, &mel, frames, Some(&mut ctx.enc_bufs))
        .ok_or("encoder failed")?;
    let mut tokens = vec![151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669];
    tokens.extend(std::iter::repeat(-1).take(audio_tokens));
    tokens.extend_from_slice(&[151670, 151645, 198, 151644, 77091, 198]);
    let mut rows = Vec::new();
    let text = std::fs::read_to_string(&args[3])?;
    if ctx.config.is_aligner() {
        for word in text.split_whitespace() {
            tokens.extend(tok.encode(word).ok_or("tokenize failed")?);
            rows.push(tokens.len());
            tokens.push(151705);
            rows.push(tokens.len());
            tokens.push(151705);
        }
    } else {
        tokens.extend(
            tok.encode("language English")
                .ok_or("language tokenization failed")?,
        );
        tokens.push(151704);
        // Also exercise actual text positions in a teacher-forced continuation.
        tokens.extend(tok.encode(text.trim()).ok_or("text tokenization failed")?);
        rows.push(tokens.len() - 1);
    }
    if rows.is_empty() {
        return Err("empty output positions".into());
    }
    let dim = ctx.config.dec_hidden;
    let seq = tokens.len();
    let mut input = vec![0f32; seq * dim];
    for (i, &id) in tokens.iter().enumerate() {
        if id < 0 {
            input[i * dim..(i + 1) * dim].copy_from_slice(&encoded[(i - 9) * dim..(i - 8) * dim]);
        } else {
            unsafe {
                decoder::tok_embed_bf16_to_f32(
                    &mut input[i * dim..(i + 1) * dim],
                    ctx.model.decoder.tok_embeddings_bf16,
                    id,
                    dim,
                );
            }
        }
    }
    let start = Instant::now();
    decoder::decoder_prefill(
        &ctx.model.decoder,
        &ctx.config,
        &mut ctx.kv_cache,
        &mut ctx.rope_cache,
        &mut ctx.dec_bufs,
        &input,
        seq,
    );
    let cpu_prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let hidden = &ctx.dec_bufs.pref_x[..seq * dim];
    let mut selected = Vec::new();
    for &r in &rows {
        selected.extend_from_slice(&hidden[r * dim..(r + 1) * dim]);
    }
    let mut normalized = vec![0.0; selected.len()];
    kernels::rms_norm(
        &mut normalized,
        &selected,
        &ctx.model.decoder.norm,
        rows.len(),
        dim,
        ctx.config.dec_rms_norm_eps,
    );
    let out_dim = ctx.config.lm_head_dim();
    let mut logits = vec![0.0; rows.len() * out_dim];
    let mut scratch = vec![0.0; out_dim * dim];
    let weight = ctx
        .model
        .decoder
        .lm_head_bf16
        .unwrap_or(ctx.model.decoder.tok_embeddings_bf16);
    unsafe {
        kernels::linear_nobias_bf16_scratch(
            &mut logits,
            &normalized,
            weight,
            rows.len(),
            dim,
            out_dim,
            &mut scratch,
        );
    }
    write_f32(&format!("{}.input.f32", args[4]), &input)?;
    write_f32(&format!("{}.hidden.f32", args[4]), hidden)?;
    write_f32(&format!("{}.logits.f32", args[4]), &logits)?;
    let top: Vec<usize> = logits
        .chunks(out_dim)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap()
                .0
        })
        .collect();
    let metadata = json!({"seq":seq,"tokens":tokens,"rows":rows,"output_dim":out_dim,"cpu_prefill_ms":cpu_prefill_ms,"argmax":top,"scope":"cpu_decoder_intermediate_oracle"});
    std::fs::write(format!("{}.json", args[4]), metadata.to_string())?;
    println!("{metadata}");
    Ok(())
}
