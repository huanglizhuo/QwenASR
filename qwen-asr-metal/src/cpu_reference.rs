//! Persistent JSONL adapter for the existing, unmodified CPU engine.
//! Timing wraps the complete public PCM -> result call, not its internal timers.
use qwen_asr::{align, audio, context::QwenCtx, kernels, transcribe};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn emit(value: Value) {
    println!("{value}");
    io::stdout().flush().expect("flush benchmark response");
}

fn required<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing nonempty string: {key}"))
}

fn run(ctx: &mut QwenCtx, task: &str, v: &Value) -> Result<Value, String> {
    if v.get("schema").and_then(Value::as_u64) != Some(1) {
        return Err("schema=1 is required".into());
    }
    required(v, "id")?;
    if required(v, "task")? != task {
        return Err("request task does not match loaded model".into());
    }
    if v.get("op").and_then(Value::as_str) != Some("infer") {
        return Err("only op=infer is supported".into());
    }
    let path = required(v, "audio")?;
    let language = required(v, "language")?;
    if language != "English" {
        return Err("v1 benchmark protocol fixes language to English".into());
    }
    // Input file decoding is separately measured; benchmark materialization
    // ensures every engine receives the same 16 kHz mono PCM16 WAV bytes.
    let io_start = Instant::now();
    let samples = audio::load_wav(path).ok_or("cannot read WAV")?;
    let input_ms = io_start.elapsed().as_secs_f64() * 1000.0;
    if samples.is_empty() {
        return Err("empty audio".into());
    }
    ctx.kv_cache.len = 0;
    let start = Instant::now();
    let (text, words) = if task == "asr" {
        let text = transcribe::transcribe_audio(ctx, &samples).ok_or("ASR failed")?;
        (text, vec![])
    } else {
        let text = required(v, "text")?;
        let aligned =
            align::forced_align(ctx, &samples, text, language).ok_or("forced alignment failed")?;
        let words: Vec<Value> = aligned
            .iter()
            .map(|w| {
                json!({
                    "text": w.text, "start_ms": w.start_ms, "end_ms": w.end_ms,
                })
            })
            .collect();
        (text.to_string(), words)
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(json!({
        "schema": 1, "ok": true, "id": v["id"], "task": task,
        "backend": "cpu", "text": text, "words": words,
        "elapsed_ms": elapsed_ms, "input_ms": input_ms,
        "audio_samples": samples.len(), "sample_rate": 16000,
        "stages": {"encode_ms": ctx.perf_encode_ms, "decode_ms": ctx.perf_decode_ms},
        "gpu_completed": false, "cpu_tensor_fallbacks": null,
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "asr" | "align") {
        return Err("usage: qwen-asr-cpu-reference <asr|align> <model-dir> <threads>".into());
    }
    let task = &args[1];
    let threads: usize = args[3].parse()?;
    let threads = if threads == 0 {
        kernels::get_num_cpus()
    } else {
        threads
    };
    kernels::set_verbose(0);
    kernels::set_threads(threads);
    let load_start = Instant::now();
    let mut ctx = QwenCtx::load(&args[2]).ok_or("model load failed")?;
    if ctx.config.dec_hidden != 1024
        || ctx.config.dec_layers != 28
        || ctx.config.is_aligner() != (task == "align")
    {
        return Err("expected the requested 0.6B model".into());
    }
    // Match the public offline path, with no segmentation, silence removal,
    // past-text conditioning or speculative decode requested by this adapter.
    ctx.segment_sec = 0.0;
    ctx.skip_silence = false;
    ctx.set_force_language("English")
        .map_err(|_| "invalid language")?;
    emit(json!({
        "schema": 1, "ready": true, "backend": "cpu", "task": task,
        "load_ms": load_start.elapsed().as_secs_f64() * 1000.0,
        "threads": threads, "flags": qwen_asr::optimization_flags(),
        "contract": "qwen-asr-metal-bench-v1",
    }));
    for line in io::stdin().lock().lines() {
        let line = line?;
        let parsed = serde_json::from_str::<Value>(&line);
        match parsed {
            Ok(v) => match run(&mut ctx, task, &v) {
                Ok(output) => emit(output),
                Err(error) => {
                    emit(json!({"schema": 1, "ok": false, "id": v["id"], "error": error}))
                }
            },
            Err(error) => emit(json!({"schema": 1, "ok": false, "error": error.to_string()})),
        }
    }
    Ok(())
}
