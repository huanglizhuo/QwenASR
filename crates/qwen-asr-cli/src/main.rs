mod download;
#[cfg(target_os = "macos")]
mod live_capture;

use config::*;
use context::QwenCtx;
use qwen_asr::{align, audio, config, context, kernels, subtitle, transcribe};

use std::io::Write;
// Only the macOS live-capture loop uses these.
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "macos")]
use std::sync::Arc;

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "mov", "avi", "webm", "m4v", "flv", "ts", "mpg", "mpeg", "wmv",
];

fn is_video_file(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Extract audio from a video file using ffmpeg, returning 16 kHz mono f32 samples.
fn extract_audio_from_video(path: &str) -> Option<Vec<f32>> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-i",
            path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                eprintln!("Error: ffmpeg not found — install it to process video files");
                eprintln!("  macOS:  brew install ffmpeg");
                eprintln!("  Linux:  sudo apt install ffmpeg");
            } else {
                eprintln!("Error: failed to run ffmpeg: {}", e);
            }
        })
        .ok()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Error: ffmpeg failed:\n{}", stderr);
        return None;
    }

    let raw = &output.stdout;
    if raw.len() % 2 != 0 {
        eprintln!("Error: ffmpeg returned odd number of bytes");
        return None;
    }

    let samples: Vec<f32> = raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
        .collect();
    Some(samples)
}

/// Load audio from either a video (via ffmpeg) or a WAV file.
fn load_audio(path: &str) -> Option<Vec<f32>> {
    if is_video_file(path) {
        extract_audio_from_video(path)
    } else {
        audio::load_wav(path)
    }
}

fn default_output_path(input: &str, extension: &str) -> String {
    let stem = std::path::Path::new(input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(input);
    let dir = std::path::Path::new(input)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");
    format!("{}/{}.{}", dir, stem, extension)
}

fn stream_token(piece: &str) {
    use std::io::Write;
    print!("{}", piece);
    std::io::stdout().flush().ok();
}

fn usage(prog: &str) {
    eprintln!("qwen-asr — Qwen3-ASR speech-to-text (pure Rust)\n");
    eprintln!(
        "Usage: {} -d <model_dir> (-i <input> | --stdin | --live) [options]\n",
        prog
    );
    eprintln!("Required:");
    eprintln!("  -d <dir>      Model directory (with *.safetensors, vocab.json)");
    eprintln!(
        "  -i <file>     Input file: WAV (16-bit PCM) or video (mp4/mkv/mov/…, requires ffmpeg)"
    );
    eprintln!("  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)");
    eprintln!("\nLive capture (macOS only):");
    eprintln!("  --live                      Capture from audio input device in real time");
    eprintln!("  --device <name>             Input device name (default: system default)");
    eprintln!("  --list-devices              List available audio input devices and exit");
    eprintln!(
        "  --vad                       Live VAD mode: detect speech segments, transcribe each"
    );
    eprintln!("\nOptions:");
    eprintln!("  -t <n>        Number of threads (default: P-cores + min(E-cores, P-cores))");
    eprintln!("  -S <secs>     Segment target seconds (default: 0 = full-audio decode)");
    eprintln!("  -W <secs>     Segment-cutting silence search window ± seconds (default: 3.0)");
    eprintln!("  --stream      Streaming mode: process in chunks with prefix rollback");
    eprintln!("  --stream-max-new-tokens <n>  Max generated tokens per stream step (default: 32)");
    eprintln!("  --stream-chunk-sec <secs>   Chunk size for streaming (default: 2.0, min ~1.0)");
    eprintln!("  --enc-window-sec <secs>    Encoder attention window in seconds (1..8, default 8)");
    eprintln!("  --past-text <yes|no|auto>  Reuse previously decoded text as context");
    eprintln!("  --skip-silence              Drop long silent spans before inference");
    eprintln!("  --prompt <text>            System prompt for biasing");
    eprintln!("  --language <lang>          Force output language");
    eprintln!("\nAlignment mode (requires ForcedAligner model):");
    eprintln!("  --align <text>             Align transcript to audio (word-level timestamps)");
    eprintln!("  --align-language <lang>    Language for word splitting (default: English)");
    eprintln!("\nSubtitle output:");
    eprintln!(
        "  --srt [path]              Write SRT subtitle file (default: <input>.srt); requires -i"
    );
    eprintln!("  --vtt [path]              Write WebVTT subtitle file (default: <input>.vtt); requires -i");
    eprintln!(
        "  --json [path]             Write structured JSON (stdout when path omitted); requires -i"
    );
    eprintln!("  --aligner-dir <dir>       ForcedAligner model for sentence-level subtitles and word timestamps");
    eprintln!("  --profile     Print per-operation timing breakdown");
    eprintln!("  --debug       Debug output (per-layer details)");
    eprintln!("  --silent      No status output (only transcription on stdout)");
    eprintln!("\nModel management:");
    eprintln!("  {} download [--list] [<model>] [--output <dir>]", prog);
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

    // Handle `download` subcommand: qwen-asr download [args...]
    if args.len() >= 2 && args[1] == "download" {
        download::handle_download_command(&args[2..]);
        return;
    }

    // Handle --list-devices (no model needed)
    if args.iter().any(|a| a == "--list-devices") {
        #[cfg(target_os = "macos")]
        {
            live_capture::print_devices();
        }
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("--list-devices is only supported on macOS.");
            eprintln!("On Linux, use: arecord -l");
        }
        return;
    }

    let mut model_dir: Option<String> = None;
    let mut input_wav: Option<String> = None;
    let mut verbosity = 1i32;
    let mut use_stdin = false;
    let mut live_mode = false;
    // Only consumed by the macOS live-capture path; the flags are still
    // parsed on other platforms (--live errors out below), so gate the
    // bindings and their assignments to match.
    #[cfg(target_os = "macos")]
    let mut device_name: Option<String> = None;
    let mut n_threads = 0i32;
    let mut segment_sec: f32 = -1.0;
    let mut search_sec: f32 = -1.0;
    let mut stream_mode = false;
    #[cfg(target_os = "macos")]
    let mut vad_mode = false;
    let mut stream_max_new_tokens: i32 = -1;
    let mut stream_chunk_sec: f32 = -1.0;
    let mut enc_window_sec: f32 = -1.0;
    let mut prompt_text: Option<String> = None;
    let mut force_language: Option<String> = None;
    let mut past_text_mode: i32 = -1; // -1 auto, 0 off, 1 on
    let mut skip_silence = false;
    let mut profile = false;
    let mut align_text: Option<String> = None;
    let mut align_language: Option<String> = None;
    // None = no SRT, Some(path) = write SRT to path
    let mut srt_path: Option<String> = None;
    let mut srt_requested = false;
    let mut vtt_path: Option<String> = None;
    let mut vtt_requested = false;
    let mut json_path: Option<String> = None;
    let mut json_requested = false;
    let mut aligner_dir: Option<String> = None;

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
            "--vad" => {
                #[cfg(target_os = "macos")]
                {
                    vad_mode = true;
                }
            }
            "--stream-max-new-tokens" => {
                i += 1;
                stream_max_new_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "--stream-chunk-sec" => {
                i += 1;
                stream_chunk_sec = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(-1.0);
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
            "--srt" => {
                srt_requested = true;
                // Optional next arg: path (if present and doesn't start with '-')
                if let Some(next) = args.get(i + 1) {
                    if !next.starts_with('-') {
                        srt_path = Some(next.clone());
                        i += 1;
                    }
                }
            }
            "--vtt" => {
                vtt_requested = true;
                if let Some(next) = args.get(i + 1) {
                    if !next.starts_with('-') {
                        vtt_path = Some(next.clone());
                        i += 1;
                    }
                }
            }
            "--json" => {
                json_requested = true;
                if let Some(next) = args.get(i + 1) {
                    if !next.starts_with('-') {
                        json_path = Some(next.clone());
                        i += 1;
                    }
                }
            }
            "--aligner-dir" => {
                i += 1;
                aligner_dir = match args.get(i) {
                    Some(path) => Some(path.clone()),
                    None => {
                        eprintln!("Error: --aligner-dir requires <dir>");
                        std::process::exit(1);
                    }
                };
            }
            "--stdin" => {
                use_stdin = true;
            }
            "--live" => {
                live_mode = true;
            }
            "--device" => {
                i += 1;
                #[cfg(target_os = "macos")]
                {
                    device_name = args.get(i).cloned();
                }
            }
            "--list-devices" => {
                // Already handled above, but don't error
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

    // Auto-prompt to download if model directory doesn't exist
    if !std::path::Path::new(&model_dir).exists() {
        if let Some(model) = download::find_model(&model_dir) {
            if download::prompt_download(&model_dir) {
                if let Err(e) = download::download_model(model, &model_dir) {
                    eprintln!("Download failed: {}", e);
                    std::process::exit(1);
                }
                eprintln!(); // blank line before model loading
            } else {
                eprintln!("Aborted.");
                std::process::exit(1);
            }
        } else {
            eprintln!("Error: Model directory '{}' not found.", model_dir);
            eprintln!();
            download::list_models();
            std::process::exit(1);
        }
    }

    if input_wav.is_none() && !use_stdin && !live_mode {
        usage(&args[0]);
        std::process::exit(1);
    }

    // Check mutual exclusivity of input modes
    let input_count = [input_wav.is_some(), use_stdin, live_mode]
        .iter()
        .filter(|&&x| x)
        .count();
    if input_count > 1 {
        eprintln!("Error: -i, --stdin, and --live are mutually exclusive");
        std::process::exit(1);
    }

    let output_requested = srt_requested || vtt_requested || json_requested;

    // Resolve file output paths (structured outputs require -i)
    if output_requested && input_wav.is_none() {
        eprintln!("Error: --srt/--vtt/--json requires -i <file>");
        std::process::exit(1);
    }
    if srt_requested && srt_path.is_none() {
        let input = input_wav.as_ref().unwrap();
        srt_path = Some(default_output_path(input, "srt"));
    }
    if vtt_requested && vtt_path.is_none() {
        let input = input_wav.as_ref().unwrap();
        vtt_path = Some(default_output_path(input, "vtt"));
    }

    kernels::set_verbose(verbosity);
    if profile {
        kernels::set_profile(true);
        kernels::profile_reset();
    }
    let emit_tokens = verbosity > 0 && !(json_requested && json_path.is_none());

    // Initialize thread pool
    if n_threads <= 0 {
        n_threads = kernels::get_default_threads() as i32;
    }
    kernels::set_threads(n_threads as usize);

    // Print optimization info
    if verbosity >= 1 {
        let opts = qwen_asr::optimization_flags();
        eprintln!(
            "Optimizations: {} | {} threads | {}",
            opts.join(", "),
            n_threads,
            std::env::consts::ARCH,
        );
    }

    // Load audio front-end concurrently with model weights when an input file is
    // supplied. WAV decode, resample, silence compaction, and mel extraction all
    // need no weights, so wall time becomes roughly max(load, mel) instead of
    // load + mel.
    let audio_handle: Option<std::thread::JoinHandle<Option<Vec<f32>>>> =
        if !use_stdin && !live_mode && input_wav.is_some() && align_text.is_none() {
            let path = input_wav.clone().unwrap();
            Some(std::thread::spawn(move || {
                let _pg = qwen_asr::kernels::ProfileGuard::new(&qwen_asr::kernels::PROF.audio_load);
                load_audio(&path)
            }))
        } else {
            None
        };

    // Load model
    let mut ctx = match QwenCtx::load(&model_dir) {
        Some(c) => c,
        None => {
            eprintln!("Failed to load model from {}", model_dir);
            std::process::exit(1);
        }
    };

    // Wait for the concurrent audio front-end to finish.
    let preloaded_samples: Option<Vec<f32>> = if let Some(h) = audio_handle {
        match h.join() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Error: audio loading thread panicked");
                std::process::exit(1);
            }
        }
    } else {
        None
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
    if stream_chunk_sec > 0.0 {
        ctx.stream_chunk_sec = stream_chunk_sec;
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
            eprintln!("Supported languages: {}", SUPPORTED_LANGUAGES.join(","));
            std::process::exit(1);
        }
    }
    // Structured output without a forced language needs the model to emit its own
    // `language X` header for detection; this skips the default prompt preamble.
    // Plain-text runs keep the byte-identical default decode path.
    if output_requested && force_language.is_none() {
        ctx.want_language_detection = true;
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

    // Live capture mode
    if live_mode {
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("Error: --live is only supported on macOS.");
            eprintln!("On Linux, pipe audio via: arecord -f S16_LE -r 16000 -c 1 | qwen-asr -d <model> --stdin");
            std::process::exit(1);
        }

        #[cfg(target_os = "macos")]
        {
            run_live_capture(
                &mut ctx,
                device_name.as_deref(),
                stream_mode,
                vad_mode,
                verbosity,
                profile,
            );
            return;
        }
    }

    // Structured output mode: load audio once, transcribe once, optionally align once.
    if output_requested {
        let input = input_wav.as_ref().unwrap();
        if verbosity >= 1 && is_video_file(input) {
            eprintln!("Extracting audio from video: {}", input);
        }
        let samples = match preloaded_samples {
            Some(s) => s,
            None => match load_audio(input) {
                Some(s) => s,
                None => {
                    eprintln!("Failed to load audio from {}", input);
                    std::process::exit(1);
                }
            },
        };

        if aligner_dir.is_none() && (srt_requested || vtt_requested) {
            eprintln!("hint: provide --aligner-dir for sentence-level subtitles");
        }

        let mut aligner_ctx = match aligner_dir.as_deref() {
            Some(dir) => match QwenCtx::load(dir) {
                Some(c) => Some(c),
                None => {
                    eprintln!("Failed to load aligner model from {}", dir);
                    std::process::exit(1);
                }
            },
            None => None,
        };

        let mut srt_file = match srt_path.as_deref() {
            Some(path) => match std::fs::File::create(path) {
                Ok(file) => Some(file),
                Err(e) => {
                    eprintln!("Error: failed to create {}: {}", path, e);
                    std::process::exit(1);
                }
            },
            None => None,
        };
        let mut vtt_file = match vtt_path.as_deref() {
            Some(path) => match std::fs::File::create(path) {
                Ok(mut file) => {
                    if let Err(e) = file.write_all(b"WEBVTT\n\n") {
                        eprintln!("Error: failed to write {}: {}", path, e);
                        std::process::exit(1);
                    }
                    Some(file)
                }
                Err(e) => {
                    eprintln!("Error: failed to create {}: {}", path, e);
                    std::process::exit(1);
                }
            },
            None => None,
        };

        let mut srt_index = 1u32;
        let mut vtt_index = 1u32;
        let mut write_error: Option<String> = None;

        // Cues are computed once per segment inside transcribe_full and shared
        // with this callback, so the streamed files match the JSON `vtt` field.
        let mut on_segment = |_segment: &qwen_asr::output::SegmentResult,
                              cues: &[subtitle::Cue]| {
            if let Some(file) = srt_file.as_mut() {
                let chunk = subtitle::format_srt_from_index(cues, srt_index);
                srt_index += subtitle::cue_count(cues);
                if write_error.is_none() {
                    if let Err(e) = file.write_all(chunk.as_bytes()).and_then(|_| file.flush()) {
                        write_error = Some(format!("failed to write SRT: {}", e));
                    }
                }
            }
            if let Some(file) = vtt_file.as_mut() {
                let chunk = subtitle::format_vtt_from_index(cues, vtt_index, false);
                vtt_index += subtitle::cue_count(cues);
                if write_error.is_none() {
                    if let Err(e) = file.write_all(chunk.as_bytes()).and_then(|_| file.flush()) {
                        write_error = Some(format!("failed to write VTT: {}", e));
                    }
                }
            }
        };

        let result = match transcribe::transcribe_full(
            &mut ctx,
            aligner_ctx.as_mut(),
            &samples,
            Some(&mut on_segment),
        ) {
            Some(result) => result,
            None => {
                eprintln!("Transcription failed");
                std::process::exit(1);
            }
        };

        if let Some(error) = write_error {
            eprintln!("Error: {}", error);
            std::process::exit(1);
        }

        if emit_tokens {
            println!();
        }

        if let Some(path) = json_path.as_deref() {
            if let Err(e) = std::fs::write(path, result.to_json()) {
                eprintln!("Error: failed to write {}: {}", path, e);
                std::process::exit(1);
            }
            if verbosity >= 1 {
                eprintln!("JSON written to {}", path);
            }
        } else if json_requested {
            print!("{}", result.to_json());
        }

        if verbosity >= 1 {
            if let Some(path) = srt_path.as_deref() {
                eprintln!("SRT written to {}", path);
            }
            if let Some(path) = vtt_path.as_deref() {
                eprintln!("VTT written to {}", path);
            }
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
            // Total audio duration from the full sample count: transcribe_full
            // overwrites ctx.perf_audio_ms per segment, so it can't be used here.
            let audio_s = samples.len() as f64 / SAMPLE_RATE as f64;
            if audio_s > 0.0 && ctx.perf_total_ms > 0.0 {
                let infer_s = ctx.perf_total_ms / 1000.0;
                eprintln!(
                    "Audio: {:.1} s processed in {:.1} s ({:.2}x realtime)",
                    audio_s,
                    infer_s,
                    audio_s / infer_s
                );
            }
            if let Some(ref aligner) = aligner_ctx {
                eprintln!(
                    "Alignment: {:.0} ms (encoding: {:.0}ms, decoding: {:.0}ms)",
                    aligner.perf_total_ms, aligner.perf_encode_ms, aligner.perf_decode_ms
                );
            }
        }

        if profile {
            kernels::profile_report();
        }
        return;
    }

    // Live incremental transcription from a stdin pipe. Reading the whole
    // stream up front (read_to_end) blocks forever on an open pipe such as
    // `arecord ... | qwen-asr --stdin --stream`, so consume it chunk by chunk
    // and emit text as audio arrives.
    if stream_mode && use_stdin {
        run_stdin_stream(&mut ctx, verbosity, profile, stream_chunk_sec > 0.0);
        return;
    }

    // Transcribe
    let text = if stream_mode {
        let samples = if use_stdin {
            audio::read_pcm_stdin()
        } else {
            preloaded_samples.or_else(|| load_audio(input_wav.as_ref().unwrap()))
        };
        match samples {
            Some(s) => transcribe::transcribe_stream(&mut ctx, &s),
            None => None,
        }
    } else if use_stdin {
        transcribe::transcribe_stdin(&mut ctx)
    } else {
        let samples = preloaded_samples.or_else(|| load_audio(input_wav.as_ref().unwrap()));
        match samples {
            Some(s) => transcribe::transcribe_audio(&mut ctx, &s),
            None => None,
        }
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
                audio_s,
                infer_s,
                audio_s / infer_s
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}

// ========================================================================
// Live Capture Loop (macOS only)
// ========================================================================

#[cfg(target_os = "macos")]
fn run_live_capture(
    ctx: &mut QwenCtx,
    device_name: Option<&str>,
    stream_mode: bool,
    vad_mode: bool,
    verbosity: i32,
    profile: bool,
) {
    use std::time::Duration;

    // Resolve device
    let device_id = if let Some(name) = device_name {
        match live_capture::find_device_by_name(name) {
            Some(dev) => {
                if verbosity >= 1 {
                    eprintln!(
                        "Using input device: {} ({} ch)",
                        dev.name, dev.input_channels
                    );
                }
                dev.id
            }
            None => {
                eprintln!("Error: No input device matching '{}'", name);
                if name.to_lowercase().contains("blackhole") {
                    eprintln!();
                    eprintln!("BlackHole does not appear to be installed.");
                    eprintln!("Install it with: brew install blackhole-2ch");
                    eprintln!("Then set it up as a Multi-Output Device in Audio MIDI Setup.");
                    eprintln!("See: https://github.com/ExistentialAudio/BlackHole");
                }
                eprintln!();
                live_capture::print_devices();
                std::process::exit(1);
            }
        }
    } else {
        match live_capture::default_input_device() {
            Some(id) => {
                if verbosity >= 1 {
                    let devices = live_capture::list_input_devices();
                    if let Some(dev) = devices.iter().find(|d| d.id == id) {
                        eprintln!("Using default input device: {}", dev.name);
                    }
                }
                id
            }
            None => {
                eprintln!("Error: No default input device found");
                std::process::exit(1);
            }
        }
    };

    // Start capture
    let (rx, _handle, device_rate) = match live_capture::start_capture(device_id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: Failed to start audio capture: {}", e);
            std::process::exit(1);
        }
    };

    let mode_label = if stream_mode {
        "streaming"
    } else if vad_mode {
        "VAD"
    } else {
        "segmented"
    };
    if verbosity >= 1 {
        if vad_mode {
            eprintln!("Listening (VAD segmented)... press Ctrl+C to stop\n");
        } else {
            eprintln!(
                "Listening ({}, {:.1}s chunks)... press Ctrl+C to stop\n",
                mode_label,
                if stream_mode {
                    ctx.stream_chunk_sec
                } else {
                    ctx.segment_sec
                }
            );
        }
    }

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl+C handler");

    // Configure context
    ctx.past_text_conditioning = true;
    ctx.reset_perf();

    // Audio accumulation
    let target_rate = 16000;
    let mut raw_buf: Vec<f32> = Vec::new();
    let mut resampled_buf: Vec<f32> = Vec::new();
    let needs_resample = (device_rate - target_rate as f64).abs() > 1.0;
    let wall_start = std::time::Instant::now();

    if stream_mode {
        // ---- Streaming mode: incremental stream_push_audio ----
        //
        // We accumulate audio and call stream_push_audio() which only
        // processes NEW audio incrementally (persistent encoder cache,
        // LCP-reused decoder prefill, monotonic token commit).
        //
        // Buffer reset after ~120s to bound memory.
        let max_window_samples: usize = 120 * target_rate as usize;
        let mut stream_state = transcribe::StreamState::new();

        // Set token callback for direct printing
        ctx.token_cb = None; // stream_push_audio returns delta text, we print it

        // Text-emission timeout: flush rollback tokens after no new text for 5s
        let mut last_text_time: Option<std::time::Instant> = None;
        let text_flush_secs = 5.0_f32;
        let mut flushed = false;

        while running.load(Ordering::SeqCst) {
            // Receive audio
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            // Resample
            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled =
                        qwen_asr::audio::resample(&raw_buf, device_rate as i32, target_rate);
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            // Reset window if buffer exceeds max
            if resampled_buf.len() > max_window_samples {
                // Flush rollback tokens before reset
                if let Some(delta) =
                    transcribe::stream_push_audio(ctx, &resampled_buf, &mut stream_state, true)
                {
                    if !delta.is_empty() {
                        print!("{}", delta);
                    }
                }
                println!();
                resampled_buf.clear();
                stream_state.reset();
                last_text_time = None;
                flushed = false;
                continue;
            }

            // Determine if we should finalize: flush rollback tokens
            // when no new text has been emitted for 5 seconds
            let finalize = !flushed
                && last_text_time.is_some_and(|t| t.elapsed().as_secs_f32() >= text_flush_secs);

            // Process all available full chunks
            if resampled_buf.len() > stream_state.audio_cursor() {
                if let Some(delta) =
                    transcribe::stream_push_audio(ctx, &resampled_buf, &mut stream_state, finalize)
                {
                    if !delta.is_empty() {
                        print!("{}", delta);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                        last_text_time = Some(std::time::Instant::now());
                        flushed = false;
                    } else if finalize {
                        flushed = true; // Don't keep calling finalize
                    }
                }
            }
        }

        // Final flush
        if !raw_buf.is_empty() && needs_resample {
            let resampled = qwen_asr::audio::resample(&raw_buf, device_rate as i32, target_rate);
            resampled_buf.extend_from_slice(&resampled);
        } else {
            resampled_buf.append(&mut raw_buf);
        }

        if resampled_buf.len() > stream_state.audio_cursor() {
            if let Some(delta) = transcribe::stream_push_audio(
                ctx,
                &resampled_buf,
                &mut stream_state,
                true, // finalize: flush rollback
            ) {
                if !delta.is_empty() {
                    print!("{}", delta);
                }
            }
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        println!();
    } else if vad_mode {
        // ---- VAD mode: energy-based speech detection + segment transcription ----
        //
        // Detect speech using RMS energy. When speech ends (silence > 1.5s),
        // transcribe the accumulated speech segment using transcribe_audio().
        // This gives better accuracy than streaming (full segment context)
        // with automatic speech boundary detection.
        let speech_threshold: f32 = 0.001;
        let silence_hangover_secs = 1.5_f32;
        let min_segment_secs = 0.5_f32;
        let max_segment_secs = 30.0_f32;
        let min_segment_samples = (min_segment_secs * target_rate as f32) as usize;
        let max_segment_samples = (max_segment_secs * target_rate as f32) as usize;
        let check_samples = (target_rate as usize) * 30 / 1000; // 30ms window for RMS

        let mut speech_active = false;
        let mut silence_start: Option<std::time::Instant> = None;
        let mut speech_start_idx: usize = 0;

        // Keep a small pre-speech buffer to avoid clipping word beginnings
        let pre_speech_samples = (target_rate as usize) / 4; // 250ms lookback

        // Disable token callback — we print the full result after each segment
        ctx.token_cb = None;

        // Cross-segment context: accumulate text to use as prompt for next segment
        let mut accumulated_text = String::new();

        while running.load(Ordering::SeqCst) {
            // Receive audio
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            // Resample
            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled =
                        qwen_asr::audio::resample(&raw_buf, device_rate as i32, target_rate);
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            // Compute RMS energy of latest 30ms
            let buf_len = resampled_buf.len();
            let rms = if buf_len >= check_samples {
                let tail = &resampled_buf[buf_len - check_samples..];
                let sum_sq: f32 = tail.iter().map(|&s| s * s).sum();
                (sum_sq / check_samples as f32).sqrt()
            } else {
                0.0
            };
            let is_speech = rms >= speech_threshold;

            // Periodic RMS debug output
            if verbosity >= 2 && buf_len % (target_rate as usize * 2) < check_samples {
                eprintln!(
                    "  [VAD] rms={:.6} threshold={:.4} speech={}",
                    rms,
                    speech_threshold,
                    if speech_active { "active" } else { "inactive" }
                );
            }

            if !speech_active {
                if is_speech {
                    // Speech started — mark the start with lookback
                    speech_active = true;
                    silence_start = None;
                    speech_start_idx = buf_len.saturating_sub(pre_speech_samples);
                    if verbosity >= 2 {
                        eprintln!(
                            "  [VAD] speech start at {:.1}s",
                            buf_len as f32 / target_rate as f32
                        );
                    }
                } else {
                    // No speech — bound buffer to avoid unlimited growth
                    // Keep only last 0.5s for lookback context
                    let keep = (target_rate as usize) / 2;
                    if resampled_buf.len() > keep * 4 {
                        let drain = resampled_buf.len() - keep;
                        resampled_buf.drain(..drain);
                    }
                }
            } else {
                // Speech is active
                let segment_len = buf_len - speech_start_idx;

                if is_speech {
                    // Still speaking — reset silence timer
                    silence_start = None;

                    // Force-flush if segment exceeds max duration
                    if segment_len >= max_segment_samples {
                        if verbosity >= 2 {
                            eprintln!(
                                "  [VAD] max segment reached ({:.1}s), flushing",
                                segment_len as f32 / target_rate as f32
                            );
                        }
                        let segment = &resampled_buf[speech_start_idx..];
                        // Set previous text as context
                        if !accumulated_text.is_empty() {
                            ctx.prompt = Some(accumulated_text.clone());
                            ctx.prompt_tokens_ready = false;
                        }
                        ctx.reset_perf();
                        if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                            if !text.is_empty() {
                                println!("{}", text);
                                accumulated_text.push_str(&text);
                            }
                        }
                        resampled_buf.clear();
                        speech_active = false;
                        silence_start = None;
                    }
                } else {
                    // Silence during speech — track duration
                    if silence_start.is_none() {
                        silence_start = Some(std::time::Instant::now());
                    }
                    if let Some(start) = silence_start {
                        if start.elapsed().as_secs_f32() >= silence_hangover_secs {
                            // End of utterance — transcribe the segment
                            if segment_len >= min_segment_samples {
                                // Trim trailing silence (keep only 200ms of it)
                                let trail_keep = (target_rate as usize) / 5;
                                let seg_end = (buf_len - check_samples + trail_keep).min(buf_len);
                                let segment = &resampled_buf[speech_start_idx..seg_end];

                                if verbosity >= 2 {
                                    eprintln!(
                                        "  [VAD] speech end, segment {:.1}s",
                                        segment.len() as f32 / target_rate as f32
                                    );
                                }

                                ctx.reset_perf();
                                // Set previous text as context
                                if !accumulated_text.is_empty() {
                                    ctx.prompt = Some(accumulated_text.clone());
                                    ctx.prompt_tokens_ready = false;
                                }
                                let t0 = std::time::Instant::now();
                                if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                                    if !text.is_empty() {
                                        println!("{}", text);
                                        accumulated_text.push_str(&text);
                                        if verbosity >= 1 {
                                            let audio_secs =
                                                segment.len() as f32 / target_rate as f32;
                                            let compute_secs = t0.elapsed().as_secs_f32();
                                            eprintln!(
                                                "  ({:.1}s audio in {:.1}s, {:.1}x realtime)",
                                                audio_secs,
                                                compute_secs,
                                                audio_secs / compute_secs.max(0.001)
                                            );
                                        }
                                    }
                                }
                            } else if verbosity >= 2 {
                                eprintln!(
                                    "  [VAD] segment too short ({:.2}s), discarding",
                                    segment_len as f32 / target_rate as f32
                                );
                            }

                            resampled_buf.clear();
                            speech_active = false;
                            silence_start = None;
                        }
                    }
                }
            }
        }

        // Flush remaining speech on Ctrl+C
        if speech_active && resampled_buf.len() > speech_start_idx + min_segment_samples {
            let segment = &resampled_buf[speech_start_idx..];
            ctx.reset_perf();
            if let Some(text) = transcribe::transcribe_audio(ctx, segment) {
                if !text.is_empty() {
                    println!("{}", text);
                }
            }
        }
    } else {
        // ---- Segmented mode: independent segments ----
        if ctx.segment_sec <= 0.0 {
            ctx.segment_sec = 5.0;
        }
        let segment_samples_16k = (ctx.segment_sec * target_rate as f32) as usize;

        while running.load(Ordering::SeqCst) {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => raw_buf.extend_from_slice(&chunk),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            while let Ok(chunk) = rx.try_recv() {
                raw_buf.extend_from_slice(&chunk);
            }

            if needs_resample {
                if !raw_buf.is_empty() {
                    let resampled =
                        qwen_asr::audio::resample(&raw_buf, device_rate as i32, target_rate);
                    resampled_buf.extend_from_slice(&resampled);
                    raw_buf.clear();
                }
            } else {
                resampled_buf.append(&mut raw_buf);
            }

            if resampled_buf.len() >= segment_samples_16k {
                ctx.reset_perf();
                let _text = transcribe::transcribe_audio(ctx, &resampled_buf);
                resampled_buf.clear();
                if verbosity > 0 {
                    println!();
                }
            }
        }

        // Flush remaining
        if !raw_buf.is_empty() && needs_resample {
            let resampled = qwen_asr::audio::resample(&raw_buf, device_rate as i32, target_rate);
            resampled_buf.extend_from_slice(&resampled);
        } else {
            resampled_buf.append(&mut raw_buf);
        }
        if !resampled_buf.is_empty() {
            ctx.reset_perf();
            let _text = transcribe::transcribe_audio(ctx, &resampled_buf);
            if verbosity > 0 {
                println!();
            }
        }
    }

    // ---- Benchmark summary ----
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let audio_s = resampled_buf.len() as f64 / target_rate as f64;

    if verbosity >= 1 {
        eprintln!("\nStopped.");
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
        if audio_s > 0.0 && ctx.perf_total_ms > 0.0 {
            let infer_s = ctx.perf_total_ms / 1000.0;
            eprintln!(
                "Audio: {:.1} s processed in {:.1} s compute ({:.2}x realtime), {:.1} s wall clock",
                audio_s,
                infer_s,
                audio_s / infer_s,
                wall_ms / 1000.0
            );
        }
    }

    if profile {
        kernels::profile_report();
    }
}

/// Locate the PCM payload inside a (possibly partial) WAV header.
///
/// Returns `(data_offset, sample_rate, channels)` once the `data` chunk header
/// has been seen, or `None` if more bytes are needed to find it.
fn wav_stream_offset(prefix: &[u8]) -> Option<(usize, i32, i32)> {
    if prefix.len() < 12 || &prefix[0..4] != b"RIFF" || &prefix[8..12] != b"WAVE" {
        return None;
    }
    let read_u16 = |d: &[u8]| u16::from_le_bytes([d[0], d[1]]) as i32;
    let read_u32 = |d: &[u8]| u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize;

    let mut sample_rate = 16000i32;
    let mut channels = 1i32;
    let mut p = 12usize;
    while p + 8 <= prefix.len() {
        let chunk_id = &prefix[p..p + 4];
        let chunk_size = read_u32(&prefix[p + 4..]);
        if chunk_id == b"fmt " && p + 8 + 16 <= prefix.len() {
            channels = read_u16(&prefix[p + 10..]);
            sample_rate = read_u32(&prefix[p + 12..]) as i32;
        } else if chunk_id == b"data" {
            // PCM payload begins right after this 8-byte chunk header.
            return Some((p + 8, sample_rate, channels));
        }
        p += 8 + chunk_size + (chunk_size & 1);
    }
    None
}

/// Spawn a thread that reads s16le PCM from stdin and forwards it as 16 kHz
/// mono f32 chunks.
///
/// Auto-detects a leading WAV header (skipped, with its sample rate / channel
/// count honored) versus raw s16le 16 kHz mono. Non-16 kHz input is resampled
/// and multi-channel input downmixed per batch, mirroring the macOS `--live`
/// loop. The channel closes at stdin EOF, which the consumer treats as
/// end-of-stream.
fn spawn_stdin_pcm_reader(verbosity: i32) -> std::sync::mpsc::Receiver<Vec<f32>> {
    use std::io::Read;
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        let mut pending: Vec<u8> = Vec::new();
        let mut header_done = false;
        let mut src_rate = 16000i32;
        let mut channels = 1i32;
        let mut buf = [0u8; 8192];
        loop {
            let n = match stdin.read(&mut buf) {
                Ok(0) => break, // EOF: pipe closed
                Ok(n) => n,
                Err(_) => break,
            };
            pending.extend_from_slice(&buf[..n]);

            // Decide raw-vs-WAV once we have enough bytes to tell.
            if !header_done {
                if pending.len() < 4 {
                    continue;
                }
                if &pending[0..4] == b"RIFF" {
                    match wav_stream_offset(&pending) {
                        Some((off, rate, ch)) => {
                            src_rate = rate.max(1);
                            channels = ch.max(1);
                            if verbosity >= 2 {
                                eprintln!(
                                    "Detected WAV on stdin: {} Hz, {} ch",
                                    src_rate, channels
                                );
                            }
                            pending.drain(0..off.min(pending.len()));
                            header_done = true;
                        }
                        None => {
                            // Keep buffering until the data chunk appears, but
                            // bail to a sane default if the header is oversized.
                            if pending.len() < 65536 {
                                continue;
                            }
                            pending.drain(0..44.min(pending.len()));
                            header_done = true;
                        }
                    }
                } else {
                    if verbosity >= 2 {
                        eprintln!("Treating stdin as raw s16le 16kHz mono");
                    }
                    header_done = true;
                }
            }

            // Convert whole frames; keep any partial trailing frame for later.
            let frame = 2 * channels as usize;
            let n_frames = pending.len() / frame;
            if n_frames == 0 {
                continue;
            }
            let mut batch = vec![0f32; n_frames];
            for (i, s) in batch.iter_mut().enumerate() {
                if channels == 1 {
                    let v = i16::from_le_bytes([pending[i * 2], pending[i * 2 + 1]]);
                    *s = v as f32 / 32768.0;
                } else {
                    let mut sum = 0.0f32;
                    for c in 0..channels as usize {
                        let off = (i * channels as usize + c) * 2;
                        sum += i16::from_le_bytes([pending[off], pending[off + 1]]) as f32;
                    }
                    *s = (sum / channels as f32) / 32768.0;
                }
            }
            pending.drain(0..n_frames * frame);

            let out = if src_rate != 16000 {
                qwen_asr::audio::resample(&batch, src_rate, 16000)
            } else {
                batch
            };
            if !out.is_empty() && tx.send(out).is_err() {
                break; // consumer gone
            }
        }
    });
    rx
}

/// Live incremental transcription from a stdin pipe (`--stdin --stream`).
///
/// Reads audio as it arrives and emits text deltas, mirroring the macOS
/// `--live` streaming loop but sourced from stdin instead of CoreAudio.
/// Rough terminal display width of a string (CJK / full-width glyphs count as
/// two columns). Good enough to keep the redrawn provisional preview from
/// wrapping; exact grapheme widths are not needed.
fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| {
            let u = c as u32;
            let wide = matches!(u,
                0x1100..=0x115F | 0x2E80..=0x303E | 0x3041..=0x33FF | 0x3400..=0x4DBF |
                0x4E00..=0x9FFF | 0xA000..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF |
                0xFE30..=0xFE4F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6 | 0x20000..=0x3FFFD);
            if wide {
                2
            } else {
                1
            }
        })
        .sum()
}

/// Live transcript renderer for the stdin streaming path.
///
/// On a TTY it prints committed text permanently (it flows and wraps like
/// normal output) and shows the current provisional tail as a dim preview right
/// after it, erasing that preview in place before the next update — so words
/// appear ~1 chunk after they are spoken instead of waiting for a commit. When
/// stdout is a pipe/file it degrades to append-only committed text (no ANSI, no
/// provisional churn) so redirected output stays clean.
struct LiveOut {
    tty: bool,
    /// Display width of the provisional preview currently drawn after the
    /// cursor, so we can erase exactly it (and nothing committed) next time.
    prov_width: usize,
}

impl LiveOut {
    fn new() -> Self {
        use std::io::IsTerminal;
        LiveOut {
            tty: std::io::stdout().is_terminal(),
            prov_width: 0,
        }
    }

    /// Emit newly committed `delta` (permanent) and refresh the `prov` preview.
    fn update(&mut self, delta: &str, prov: &str) {
        let mut out = std::io::stdout().lock();
        if self.tty {
            if self.prov_width > 0 {
                // Move back over the old preview and clear to end of line.
                let _ = write!(out, "\x1b[{}D\x1b[K", self.prov_width);
                self.prov_width = 0;
            }
            if !delta.is_empty() {
                let _ = write!(out, "{}", delta);
            }
            if !prov.is_empty() {
                let _ = write!(out, "\x1b[90m{}\x1b[0m", prov);
                self.prov_width = display_width(prov);
            }
            let _ = out.flush();
        } else if !delta.is_empty() {
            let _ = write!(out, "{}", delta);
            let _ = out.flush();
        }
    }

    /// Break to a fresh line at a speech pause / re-anchor so each utterance
    /// lands on its own line. TTY-only: piped output stays a single flowing
    /// stream (committed tokens already carry their own spacing), matching the
    /// non-streaming transcript format.
    fn newline(&mut self) {
        if !self.tty {
            return;
        }
        let mut out = std::io::stdout().lock();
        if self.prov_width > 0 {
            let _ = write!(out, "\x1b[{}D\x1b[K", self.prov_width);
            self.prov_width = 0;
        }
        let _ = writeln!(out);
        let _ = out.flush();
    }

    /// Terminate output with a trailing newline (both TTY and pipe).
    fn finish(&mut self) {
        let mut out = std::io::stdout().lock();
        if self.tty && self.prov_width > 0 {
            let _ = write!(out, "\x1b[{}D\x1b[K", self.prov_width);
            self.prov_width = 0;
        }
        let _ = writeln!(out);
        let _ = out.flush();
    }
}

fn run_stdin_stream(ctx: &mut QwenCtx, verbosity: i32, profile: bool, chunk_sec_set: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::RecvTimeoutError;
    use std::sync::Arc;
    use std::time::Duration;

    let target_rate = 16000usize;
    let max_window_samples = 120 * target_rate;

    ctx.past_text_conditioning = true;
    ctx.reset_perf();

    // Live streaming wants progressive output. The library default holds every
    // token "unfixed" until chunk 99 (≈13 min) or EOF so a finite `--stdin`
    // file transcribes at maximum accuracy — but on a live mic pipe that means
    // nothing ever prints. Commit after a few chunks (still guarded by the
    // rollback window) so words surface as they stabilize.
    ctx.stream_unfixed_chunks = 3;
    // The 8s library default is tuned for a finite file; on a live mic that is
    // an 8s "speak, then wait" latency floor since a full chunk must accumulate
    // before the encoder runs at all. Default the live path to a snappier 2s
    // unless the user picked a value with --stream-chunk-sec.
    if !chunk_sec_set {
        ctx.stream_chunk_sec = 2.0;
    }
    let chunk_samples = (ctx.stream_chunk_sec * target_rate as f32) as usize;

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let _ = ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    });

    if verbosity >= 1 {
        eprintln!(
            "Listening (streaming, {:.1}s chunks) from stdin... (EOF or Ctrl+C to stop)\n",
            ctx.stream_chunk_sec
        );
    }

    let rx = spawn_stdin_pcm_reader(verbosity);
    let mut state = transcribe::StreamState::new();
    let mut audio_buf: Vec<f32> = Vec::new();
    let mut total_samples: usize = 0;
    ctx.token_cb = None; // stream_push_audio returns delta text directly

    // On a continuous stream, `finalize` only happens at EOF, so stable text
    // held in the rollback window would never surface. Commit it once the
    // hypothesis stops changing for a short gap (a speech pause) — that is
    // exactly when early-commit is safe, and it keeps the correction window
    // open while speech is actively flowing.
    let text_flush_secs = 1.5_f32;
    let mut last_provisional = String::new();
    let mut last_change = std::time::Instant::now();
    let mut out = LiveOut::new();

    // Run one incremental step and hand the committed delta + current
    // provisional tail to the renderer. Returns the delta so callers can tell
    // whether anything committed.
    fn push(
        ctx: &mut QwenCtx,
        audio_buf: &[f32],
        state: &mut transcribe::StreamState,
        finalize: bool,
        out: &mut LiveOut,
    ) -> String {
        if audio_buf.len() > state.audio_cursor() {
            if let Some(delta) = transcribe::stream_push_audio(ctx, audio_buf, state, finalize) {
                out.update(&delta, &state.provisional_text());
                return delta;
            }
        }
        String::new()
    }

    let mut eof = false;
    while running.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                total_samples += chunk.len();
                audio_buf.extend_from_slice(&chunk);
                while let Ok(chunk) = rx.try_recv() {
                    total_samples += chunk.len();
                    audio_buf.extend_from_slice(&chunk);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => eof = true, // stdin closed
        }

        // Bound memory on long-running streams: flush and re-anchor at ~120s.
        if audio_buf.len() > max_window_samples {
            push(ctx, &audio_buf, &mut state, true, &mut out);
            out.newline();
            audio_buf.clear();
            state.reset();
            last_provisional.clear();
            last_change = std::time::Instant::now();
            continue;
        }

        // Process buffered audio one chunk at a time, re-checking `running`
        // between chunks. On a machine slower than realtime the reader thread
        // fills `audio_buf` faster than we consume it; feeding a growing prefix
        // slice keeps each `stream_push_audio` call bounded to a single chunk so
        // Ctrl+C interrupts within one chunk instead of grinding the whole
        // backlog (and re-grinding it again in a `finalize` flush).
        while running.load(Ordering::SeqCst)
            && audio_buf.len().saturating_sub(state.audio_cursor()) >= chunk_samples
        {
            let end = state.audio_cursor() + chunk_samples;
            push(ctx, &audio_buf[..end], &mut state, false, &mut out);
        }

        if eof {
            break;
        }

        // Commit held text if the hypothesis has been stable through a pause,
        // then break the line so each utterance stands on its own.
        let quiet =
            !last_provisional.is_empty() && last_change.elapsed().as_secs_f32() >= text_flush_secs;
        if quiet {
            push(ctx, &audio_buf, &mut state, true, &mut out);
            out.newline();
            last_provisional.clear();
            last_change = std::time::Instant::now();
            continue;
        }

        let prov = state.provisional_text();
        if prov != last_provisional {
            last_provisional = prov;
            last_change = std::time::Instant::now();
        }
    }

    if running.load(Ordering::SeqCst) {
        // Genuine EOF: the input ended, so finalize the remaining tail. The
        // per-chunk loop above already consumed all full chunks, leaving at most
        // a sub-chunk remainder here — cheap to flush.
        while let Ok(chunk) = rx.try_recv() {
            total_samples += chunk.len();
            audio_buf.extend_from_slice(&chunk);
        }
        push(ctx, &audio_buf, &mut state, true, &mut out);
    } else {
        // Ctrl+C: stop now. Promote the already-decoded provisional tail to
        // committed text instead of finalizing (which would re-encode the
        // unprocessed backlog and hang).
        let prov = state.provisional_text();
        out.update(&prov, "");
    }
    out.finish();

    ctx.perf_audio_ms = 1000.0 * total_samples as f64 / target_rate as f64;
    if verbosity >= 1 {
        let tokens_per_sec = if ctx.perf_total_ms > 0.0 {
            1000.0 * ctx.perf_text_tokens as f64 / ctx.perf_total_ms
        } else {
            0.0
        };
        eprintln!(
            "Inference: {:.0} ms, {} text tokens ({:.2} tok/s, encoding: {:.0}ms, decoding: {:.0}ms)",
            ctx.perf_total_ms,
            ctx.perf_text_tokens,
            tokens_per_sec,
            ctx.perf_encode_ms,
            ctx.perf_decode_ms
        );
    }
    if profile {
        kernels::profile_report();
    }
}

#[cfg(test)]
mod tests {
    use super::display_width;
    use super::wav_stream_offset;

    #[test]
    fn display_width_counts_cjk_as_two() {
        assert_eq!(display_width("hello"), 5);
        assert_eq!(display_width(""), 0);
        assert_eq!(display_width("你好"), 4); // two full-width CJK glyphs
        assert_eq!(display_width("a你b"), 4); // 1 + 2 + 1
    }

    fn wav_header(sample_rate: u32, channels: u16, data_len: u32) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + data_len).to_le_bytes());
        h.extend_from_slice(b"WAVE");
        h.extend_from_slice(b"fmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes()); // PCM
        h.extend_from_slice(&channels.to_le_bytes());
        h.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * channels as u32 * 2;
        h.extend_from_slice(&byte_rate.to_le_bytes());
        h.extend_from_slice(&(channels * 2).to_le_bytes()); // block align
        h.extend_from_slice(&16u16.to_le_bytes()); // bits
        h.extend_from_slice(b"data");
        h.extend_from_slice(&data_len.to_le_bytes());
        h
    }

    #[test]
    fn parses_standard_wav_header() {
        let h = wav_header(16000, 1, 1000);
        assert_eq!(wav_stream_offset(&h), Some((44, 16000, 1)));
    }

    #[test]
    fn honors_rate_and_channels() {
        let h = wav_header(44100, 2, 500);
        assert_eq!(wav_stream_offset(&h), Some((44, 44100, 2)));
    }

    #[test]
    fn skips_extra_chunk_before_data() {
        // Insert a LIST chunk between fmt and data; data offset must account for it.
        let mut h = Vec::new();
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(b"WAVE");
        h.extend_from_slice(b"fmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&16000u32.to_le_bytes());
        h.extend_from_slice(&32000u32.to_le_bytes());
        h.extend_from_slice(&2u16.to_le_bytes());
        h.extend_from_slice(&16u16.to_le_bytes());
        h.extend_from_slice(b"LIST");
        h.extend_from_slice(&6u32.to_le_bytes());
        h.extend_from_slice(&[0u8; 6]);
        let data_off = h.len() + 8;
        h.extend_from_slice(b"data");
        h.extend_from_slice(&100u32.to_le_bytes());
        assert_eq!(wav_stream_offset(&h), Some((data_off, 16000, 1)));
    }

    #[test]
    fn none_when_data_chunk_not_yet_seen() {
        // fmt present but no data chunk yet (still streaming the header).
        let h = &wav_header(16000, 1, 1000)[..36];
        assert_eq!(wav_stream_offset(h), None);
    }

    #[test]
    fn none_for_non_riff() {
        assert_eq!(wav_stream_offset(b"not a wav file at all........"), None);
    }
}
