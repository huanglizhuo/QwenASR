use qasr::{audio, config, context, kernels, transcribe, align};
use config::*;
use context::QwenCtx;

fn stream_token(piece: &str) {
    use std::io::Write;
    print!("{}", piece);
    std::io::stdout().flush().ok();
}

fn usage(prog: &str) {
    eprintln!("q-asr — Qwen3-ASR speech-to-text (pure Rust)\n");
    eprintln!("Usage: {} -d <model_dir> (-i <input.wav> | --stdin) [options]\n", prog);
    eprintln!("Required:");
    eprintln!("  -d <dir>      Model directory (with *.safetensors, vocab.json)");
    eprintln!("  -i <file>     Input WAV file (16-bit PCM, any sample rate)");
    eprintln!("  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)");
    eprintln!("\nOptions:");
    eprintln!("  -t <n>        Number of threads (default: all CPUs)");
    eprintln!("  -S <secs>     Segment target seconds (default: 0 = full-audio decode)");
    eprintln!("  -W <secs>     Segment-cutting silence search window ± seconds (default: 3.0)");
    eprintln!("  --stream      Streaming mode: process in chunks with prefix rollback");
    eprintln!("  --stream-max-new-tokens <n>  Max generated tokens per stream step (default: 32)");
    eprintln!("  --enc-window-sec <secs>    Encoder attention window in seconds (1..8, default 8)");
    eprintln!("  --past-text <yes|no|auto>  Reuse previously decoded text as context");
    eprintln!("  --skip-silence              Drop long silent spans before inference");
    eprintln!("  --prompt <text>            System prompt for biasing");
    eprintln!("  --language <lang>          Force output language");
    eprintln!("\nAlignment mode (requires ForcedAligner model):");
    eprintln!("  --align <text>             Align transcript to audio (word-level timestamps)");
    eprintln!("  --align-language <lang>    Language for word splitting (default: English)");
    eprintln!("  --profile     Print per-operation timing breakdown");
    eprintln!("  --debug       Debug output (per-layer details)");
    eprintln!("  --silent      No status output (only transcription on stdout)");
    eprintln!("  -h            Show this help");
}

fn parse_past_text_mode(s: &str) -> Option<i32> {
    match s.to_lowercase().as_str() {
        "yes" => Some(1),
        "no" => Some(0),
        "auto" => Some(-1),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir: Option<String> = None;
    let mut input_wav: Option<String> = None;
    let mut verbosity = 1i32;
    let mut use_stdin = false;
    let mut n_threads = 0i32;
    let mut segment_sec: f32 = -1.0;
    let mut search_sec: f32 = -1.0;
    let mut stream_mode = false;
    let mut stream_max_new_tokens: i32 = -1;
    let mut enc_window_sec: f32 = -1.0;
    let mut prompt_text: Option<String> = None;
    let mut force_language: Option<String> = None;
    let mut past_text_mode: i32 = -1; // -1 auto, 0 off, 1 on
    let mut skip_silence = false;
    let mut profile = false;
    let mut align_text: Option<String> = None;
    let mut align_language: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-d" => {
                i += 1;
                model_dir = args.get(i).cloned();
            }
            "-i" => {
                i += 1;
                input_wav = args.get(i).cloned();
            }
            "-t" => {
                i += 1;
                n_threads = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "-S" => {
                i += 1;
                segment_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "-W" => {
                i += 1;
                search_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--stream" => {
                stream_mode = true;
            }
            "--stream-max-new-tokens" => {
                i += 1;
                stream_max_new_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "--enc-window-sec" => {
                i += 1;
                enc_window_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "--past-text" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    match parse_past_text_mode(s) {
                        Some(m) => past_text_mode = m,
                        None => {
                            eprintln!("Error: --past-text must be one of yes|no|auto, got '{}'", s);
                            std::process::exit(1);
                        }
                    }
                }
            }
            "--skip-silence" => {
                skip_silence = true;
            }
            "--prompt" => {
                i += 1;
                prompt_text = args.get(i).cloned();
            }
            "--language" => {
                i += 1;
                force_language = args.get(i).cloned();
            }
            "--align" => {
                i += 1;
                align_text = args.get(i).cloned();
            }
            "--align-language" => {
                i += 1;
                align_language = args.get(i).cloned();
            }
            "--stdin" => {
                use_stdin = true;
            }
            "--profile" => {
                profile = true;
            }
            "--debug" => {
                verbosity = 2;
            }
            "--silent" => {
                verbosity = 0;
            }
            "-h" | "--help" => {
                usage(&args[0]);
                return;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                usage(&args[0]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_dir = match model_dir {
        Some(d) => d,
        None => {
            usage(&args[0]);
            std::process::exit(1);
        }
    };

    if input_wav.is_none() && !use_stdin {
        usage(&args[0]);
        std::process::exit(1);
    }

    if input_wav.is_some() && use_stdin {
        eprintln!("Error: -i and --stdin are mutually exclusive");
        std::process::exit(1);
    }

    kernels::set_verbose(verbosity);
    if profile {
        kernels::set_profile(true);
        kernels::profile_reset();
    }
    let emit_tokens = verbosity > 0;

    // Initialize thread pool
    if n_threads <= 0 {
        n_threads = kernels::get_num_cpus() as i32;
    }
    kernels::set_threads(n_threads as usize);

    // Load model
    let mut ctx = match QwenCtx::load(&model_dir) {
        Some(c) => c,
        None => {
            eprintln!("Failed to load model from {}", model_dir);
            std::process::exit(1);
        }
    };

    // Apply settings
    if segment_sec >= 0.0 {
        ctx.segment_sec = segment_sec;
    }
    if search_sec >= 0.0 {
        ctx.search_sec = search_sec;
    }
    if enc_window_sec >= 0.0 {
        let window_frames = (enc_window_sec * 100.0 + 0.5) as usize;
        ctx.config.enc_n_window_infer = window_frames.clamp(100, 800);
    }
    if stream_max_new_tokens > 0 {
        ctx.stream_max_new_tokens = stream_max_new_tokens;
    }
    if past_text_mode >= 0 {
        ctx.past_text_conditioning = past_text_mode == 1;
    } else if stream_mode {
        ctx.past_text_conditioning = true;
    }
    if skip_silence {
        ctx.skip_silence = true;
    }
    if let Some(ref prompt) = prompt_text {
        if ctx.set_prompt(prompt).is_err() {
            eprintln!("Failed to set --prompt text");
            std::process::exit(1);
        }
    }
    if let Some(ref lang) = force_language {
        if ctx.set_force_language(lang).is_err() {
            eprintln!("Unsupported language for --language: {}", lang);
            eprintln!(
                "Supported languages: {}",
                SUPPORTED_LANGUAGES.join(",")
            );
            std::process::exit(1);
        }
    }

    // Alignment mode
    if let Some(ref atext) = align_text {
        let lang = align_language.as_deref().unwrap_or("English");
        let lang_normalized = match normalize_language(lang) {
            Some(l) => l,
            None => {
                eprintln!("Unsupported --align-language: {}", lang);
                eprintln!("Supported languages: {}", SUPPORTED_LANGUAGES.join(","));
                std::process::exit(1);
            }
        };

        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            audio::load_wav(input_wav.as_ref().unwrap())
        };
        let samples = match samples {
            Some(s) => s,
            None => {
                eprintln!("Failed to load audio");
                std::process::exit(1);
            }
        };

        match align::forced_align(&mut ctx, &samples, atext, &lang_normalized) {
            Some(results) => {
                // Output JSON array
                println!("[");
                for (i, r) in results.iter().enumerate() {
                    let comma = if i + 1 < results.len() { "," } else { "" };
                    // Escape the text for JSON
                    let escaped = r.text.replace('\\', "\\\\").replace('"', "\\\"");
                    println!(
                        "  {{\"text\": \"{}\", \"start\": {:.0}, \"end\": {:.0}}}{}",
                        escaped, r.start_ms, r.end_ms, comma
                    );
                }
                println!("]");
            }
            None => {
                eprintln!("Alignment failed");
                std::process::exit(1);
            }
        }

        if verbosity >= 1 {
            eprintln!(
                "Alignment: {:.0} ms (encoding: {:.0}ms, decoding: {:.0}ms)",
                ctx.perf_total_ms, ctx.perf_encode_ms, ctx.perf_decode_ms
            );
        }

        if profile {
            kernels::profile_report();
        }
        return;
    }

    // Set token callback
    if emit_tokens {
        ctx.token_cb = Some(Box::new(stream_token));
    }

    // Transcribe
    let text = if stream_mode {
        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            audio::load_wav(input_wav.as_ref().unwrap())
        };
        match samples {
            Some(s) => transcribe::transcribe_stream(&mut ctx, &s),
            None => None,
        }
    } else if use_stdin {
        transcribe::transcribe_stdin(&mut ctx)
    } else {
        transcribe::transcribe(&mut ctx, input_wav.as_ref().unwrap())
    };

    match text {
        Some(t) => {
            if emit_tokens {
                println!();
            } else {
                println!("{}", t);
            }
        }
        None => {
            eprintln!("Transcription failed");
            std::process::exit(1);
        }
    }

    if verbosity >= 1 {
        let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
            1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
        } else {
            0.0
        };
        eprintln!(
            "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
            ctx.perf_total_ms, ctx.perf_text_tokens, tokens_per_sec,
            ctx.perf_encode_ms, ctx.perf_decode_ms
        );
        if ctx.perf_audio_ms > 0.0 && ctx.perf_total_ms > 0.0 {
            let audio_s = ctx.perf_audio_ms / 1000.0;
            let infer_s = ctx.perf_total_ms / 1000.0;
            eprintln!(
                "Audio: {:.1} s processed in {:.1} s ({:.2}x realtime)",
                audio_s, infer_s, audio_s / infer_s
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}
