//! Development-only intermediate oracle. Never a GPU production fallback.
use qwen_asr::{audio, context::QwenCtx, kernels};
use serde_json::json;
use std::{fs::File, io::Write, time::Instant};

fn write_f32(path: &str, data: &[f32]) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(File::create(path)?);
    for value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: encoder-reference <model> <wav> <output-prefix>".into());
    }
    kernels::set_verbose(0);
    kernels::set_threads(kernels::get_num_cpus());
    let mut ctx = QwenCtx::load(&args[1]).ok_or("model load failed")?;
    let samples = audio::load_wav(&args[2]).ok_or("audio load failed")?;
    let (mel, frames) = audio::mel_spectrogram(&samples).ok_or("mel failed")?;
    let start = Instant::now();
    let (encoded, tokens) = ctx
        .model
        .encoder
        .forward(&ctx.config, &mel, frames, Some(&mut ctx.enc_bufs))
        .ok_or("encoder failed")?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    write_f32(&format!("{}.mel.f32", args[3]), &mel)?;
    write_f32(&format!("{}.pcm.f32", args[3]), &samples)?;
    write_f32(&format!("{}.encoded.f32", args[3]), &encoded)?;
    let metadata = json!({"frames":frames,"tokens":tokens,"hidden":ctx.config.enc_output_dim,"cpu_encoder_ms":ms,"model":args[1],"audio":args[2]});
    std::fs::write(format!("{}.json", args[3]), metadata.to_string())?;
    println!("{metadata}");
    Ok(())
}
