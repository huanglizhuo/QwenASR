//! qwen-asr-server: Groq/OpenAI-compatible ASR endpoint over the qwen-asr-metal
//! GPU engine. Pipeline: upload -> ffmpeg 16k mono -> per-chunk ASR -> self
//! forced-alignment for word timestamps -> verbose_json (PodParrot contract).
//!
//! Edge-case policy (see PROGRESS.md / plan): text always a string, arrays
//! never null, numbers sanitized to finite, word ends clamped to duration,
//! empty ASR short-circuits to empty arrays, `model` accepted and ignored.
use axum::{extract::Multipart, http::{HeaderMap, StatusCode}, response::IntoResponse, routing::{get, post}, Json, Router};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, Semaphore};

const MAX_UPLOAD_BYTES: u64 = 200 * 1024 * 1024;
const CHUNK_SECONDS: f64 = 28.0;
const OVERLAP_SECONDS: f64 = 0.4;
const SEGMENT_GAP: f64 = 0.6;
const SEGMENT_SOFT_CAP: f64 = 15.0;

struct Engine {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
}

struct App {
    asr: Arc<Mutex<Engine>>,
    align: Arc<Mutex<Engine>>,
    gpu: Semaphore, // GPU is single-tenant: one inference at a time.
    admission: Arc<Semaphore>, // 2 permits: 1 running + 1 queued; beyond -> 429.
}

fn die(msg: &str) -> ! {
    eprintln!("qwen-asr-server: {msg}");
    std::process::exit(1);
}

async fn spawn_engine(binary: &Path, task: &str, model: &Path, kernels: &Path) -> Engine {
    let mut child = Command::new(binary)
        .arg(task).arg(model).arg(kernels)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::fs::File::create("/tmp/qwen-engine-stderr.log").map(std::process::Stdio::from).unwrap_or_else(|_| std::process::Stdio::null()))
        .spawn()
        .unwrap_or_else(|e| die(&format!("cannot spawn engine ({e})")));
    let stdin = child.stdin.take().expect("engine stdin");
    let mut raw = child.stdout.take().expect("engine stdout");
    let mut handshake = String::new();
    {
        use tokio::io::AsyncReadExt;
        // read the ready line byte-wise before wrapping in BufReader
        let mut byte = [0u8; 1];
        loop {
            match raw.read(&mut byte).await {
                Ok(0) => break,
                Ok(_) if byte[0] == b'\n' => break,
                Ok(_) => handshake.push(byte[0] as char),
                Err(_) => break,
            }
        }
    }
    if !handshake.contains("\"ready\":true") { die(&format!("engine handshake failed: {handshake}")); }
    let stdout = BufReader::new(raw);
    Engine { child, stdin, stdout }
}

impl App {
    fn asr_arc(&self) -> Arc<Mutex<Engine>> { self.asr.clone() }
    fn align_arc(&self) -> Arc<Mutex<Engine>> { self.align.clone() }
}

impl Engine {
    async fn infer(&mut self, request: Value) -> Result<Value, String> {
        let line = (serde_json::to_string(&request).unwrap() + "\n").into_bytes();
        self.stdin.write_all(&line).await.map_err(|e| format!("engine pipe: {e}"))?;
        self.stdin.flush().await.map_err(|e| format!("engine pipe: {e}"))?;
        let mut buf = String::new();
        self.stdout.read_line(&mut buf).await.map_err(|e| format!("engine read: {e}"))?;
        let v: Value = serde_json::from_str(buf.trim()).map_err(|e| format!("engine response: {e}"))?;
        if v.get("ok") != Some(&json!(true)) { return Err(v["error"].as_str().unwrap_or("engine error").into()); }
        Ok(v)
    }
}

#[tokio::main]
async fn main() {
    let root = std::env::var("QWEN_METAL_ROOT").unwrap_or_else(|_| ".".into());
    let root = PathBuf::from(root);
    let binary = root.join("target/release/qwen-asr-gpu");
    let kernels = root.join("native");
    let asr_model = std::env::var("QWEN_ASR_MODEL").map(PathBuf::from)
        .unwrap_or_else(|_| root.join("models/hybrid-profile/asr"));
    let align_model = std::env::var("QWEN_ALIGN_MODEL").map(PathBuf::from)
        .unwrap_or_else(|_| root.join("models/hybrid-profile/align"));
    let token = std::env::var("QWEN_ASR_TOKEN").unwrap_or_else(|_| die("QWEN_ASR_TOKEN is required"));
    let port: u16 = std::env::var("QWEN_ASR_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);

    let app = Arc::new(App {
        asr: Arc::new(Mutex::new(spawn_engine(&binary, "asr", &asr_model, &kernels).await)),
        align: Arc::new(Mutex::new(spawn_engine(&binary, "align", &align_model, &kernels).await)),
        gpu: Semaphore::new(1),
        admission: Arc::new(Semaphore::new(2)),
    });

    // Startup warmup: one dummy inference per engine so the first real request
    // does not pay model load + shader compilation.
    {
        let dir = std::env::temp_dir().join(format!("qwen-warm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let warm = dir.join("warm.wav");
        write_wav_s16(&warm, &vec![0i16; 16000]).unwrap();
        let a = app.asr.lock().await.infer(json!({"op":"infer","schema":1,"id":"warm","task":"asr","audio":warm.to_str().expect("path"),"language":"English"})).await;
        let b = app.align.lock().await.infer(json!({"op":"infer","schema":1,"id":"warm","task":"align","audio":warm.to_str().expect("path"),"text":"hello","language":"English"})).await;
        if let Err(e) = &a { die(&format!("warmup asr failed: {e}")); }
        if let Err(e) = &b { die(&format!("warmup align failed: {e}")); }
        eprintln!("warmup complete");
    }

    let state = app.clone();
    let auth_token = token.clone();
    let router = Router::new()
        .route("/openai/v1/audio/transcriptions", post(move |h, m| transcriptions(state.clone(), auth_token.clone(), h, m)))
        .route("/health", get(|| async { Json(json!({"ok": true})) }))
        .layer(axum::extract::DefaultBodyLimit::max((MAX_UPLOAD_BYTES + (1 << 20)) as usize));

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap_or_else(|e| die(&format!("bind: {e}")));
    eprintln!("listening on 0.0.0.0:{port}");
    axum::serve(listener, router).await.unwrap();
}

fn authorized(token: &str, headers: &HeaderMap) -> bool {
    headers.get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.strip_prefix("Bearer ") == Some(token))
        .unwrap_or(false)
}

async fn transcriptions(app: Arc<App>, token: String, headers: HeaderMap, mut multipart: Multipart) -> axum::response::Response {
    if !authorized(&token, &headers) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error":{"message":"invalid bearer token","type":"invalid_request_error"}}))).into_response();
    }
    let mut audio: Option<Vec<u8>> = None;
    let mut audio_url: Option<String> = None;
    let mut format = "verbose_json".to_string();
    let mut heartbeat = false;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => { eprintln!("multipart error: {e}"); break; }
        };
        let fname = field.name().unwrap_or("").to_string();
        match fname.to_ascii_lowercase().as_str() {
            "file" => {
                let data = match field.bytes().await {
                    Ok(d) => d,
                    Err(e) => { eprintln!("file field read error: {e}"); return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":format!("upload read error: {e}"),"type":"invalid_request_error"}}))).into_response(); }
                };
                eprintln!("file field: {} bytes", data.len());
                if data.len() as u64 > MAX_UPLOAD_BYTES {
                    return (StatusCode::PAYLOAD_TOO_LARGE, Json(json!({"error":{"message":"file too large","type":"invalid_request_error"}}))).into_response();
                }
                if !data.is_empty() { audio = Some(data.to_vec()); }
            }
            "response_format" => if let Some(v) = field.text().await.ok() { format = v.trim().to_ascii_lowercase(); },
            "url" => if let Some(v) = field.text().await.ok() { if !v.trim().is_empty() { audio_url = Some(v.trim().to_string()); } },
            "heartbeat" => if field.text().await.unwrap_or_default().trim() == "1" { heartbeat = true; },
            // `model`, `language`, `timestamp_granularities[]`, `prompt`,
            // `temperature` are accepted; only model/language are echoed.
            _ => {}
        }
    }
    let audio = match audio {
        Some(a) => a,
        None => match audio_url {
            Some(u) => match download_checked(&u).await {
                Ok(a) => a,
                Err(msg) => return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":msg,"type":"invalid_request_error"}}))).into_response(),
            },
            None => return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"file or url is required","type":"invalid_request_error"}}))).into_response(),
        },
    };
    // Admission control: one request may be queued behind a running one.
    let _admit = match tokio::time::timeout(std::time::Duration::from_secs(120), app.admission.clone().acquire_owned()).await {
        Ok(Ok(p)) => p,
        _ => return (StatusCode::TOO_MANY_REQUESTS, Json(json!({"error":{"message":"server busy, retry later","type":"rate_limit_error"}}))).into_response(),
    };
    if heartbeat {
        // SSE-shaped response: ':' comment heartbeats every 10s keep the
        // origin connection alive under Cloudflare's ~100s first-byte limit;
        // the payload arrives as one `data: {json}` line at the end.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<u8>, std::io::Error>>(8);
        let result_tx = tx.clone();
        let (done_tx, mut done_rx) = tokio::sync::watch::channel(false);
        let app2 = app.clone();
        let fmt = format.clone();
        tokio::spawn(async move {
            let out = tokio::time::timeout(std::time::Duration::from_secs(30 * 60), transcribe(&app2, &audio)).await;
            let payload = match out {
                Ok(Ok(v)) => render(v, &fmt),
                Ok(Err(msg)) => ("application/json".into(), json!({"error":{"message":msg,"type":"server_error"}}).to_string()),
                Err(_) => ("application/json".into(), json!({"error":{"message":"request timed out","type":"server_error"}}).to_string()),
            };
            let single = payload.1.replace('\n', " ");
            let _ = result_tx.send(Ok(format!("event: done\ndata: {single}\n\n").into_bytes())).await;
            let _ = done_tx.send(true);
        });
        let beat_tx = tx;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                        if beat_tx.send(Ok(b": keepalive\n\n".to_vec())).await.is_err() { break; }
                    }
                    _ = done_rx.changed() => break,
                }
            }
        });
        use tokio_stream::wrappers::ReceiverStream;
        let stream = ReceiverStream::new(rx);
        return (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "text/event-stream; charset=utf-8")],
                axum::body::Body::from_stream(stream)).into_response();
    }
    let result = tokio::time::timeout(std::time::Duration::from_secs(30 * 60), transcribe(&app, &audio)).await;
    match format.as_str() {
        "verbose_json" | "json" | "text" | "vtt" => {}
        _ => return (StatusCode::BAD_REQUEST, Json(json!({"error":{"message":"unsupported response_format","type":"invalid_request_error"}}))).into_response(),
    }
    match result {
        Ok(Ok(out)) => respond(out, &format),
        Err(_) => (StatusCode::GATEWAY_TIMEOUT, Json(json!({"error":{"message":"request timed out","type":"server_error"}}))).into_response(),
        Ok(Err(msg)) => {
            let status = if msg.starts_with("decode:") { StatusCode::UNSUPPORTED_MEDIA_TYPE } else { StatusCode::INTERNAL_SERVER_ERROR };
            (status, Json(json!({"error":{"message":msg,"type": if status==StatusCode::UNSUPPORTED_MEDIA_TYPE {"invalid_request_error"} else {"server_error"}}}))).into_response()
        }
    }
}

// SSRF-guarded download: https only, resolve+validate all IPs (no loopback/
// private/link-local), pin the connection to a validated IP via --resolve,
// no redirect auto-follow (manual, <=3 hops, revalidated).
async fn download_checked(url: &str) -> Result<Vec<u8>, String> {
    use std::net::ToSocketAddrs;
    let mut current = url.to_string();
    for _ in 0..3 {
        let parsed = current.split_once("://").ok_or("url must include scheme")?;
        if parsed.0 != "https" { return Err("only https urls are supported".into()); }
        let rest = parsed.1;
        let (hostport, _path) = rest.split_once('/').unwrap_or((rest, ""));
        let (host, port) = match hostport.rsplit_once(':') {
            Some((h, p)) => (h.trim_matches('[').trim_matches(']').to_string(), p.parse::<u16>().unwrap_or(443)),
            None => (hostport.to_string(), 443u16),
        };
        if host.is_empty() { return Err("empty url host".into()); }
        let addrs: Vec<_> = (host.as_str(), port).to_socket_addrs().map_err(|e| format!("dns: {e}"))?.collect();
        if addrs.is_empty() { return Err("dns: no addresses".into()); }
        for a in &addrs {
            let ip = a.ip();
            let bad = ip.is_loopback() || ip.is_unspecified() || ip.is_multicast()
                    || match ip { std::net::IpAddr::V4(v) => v.is_link_local(), std::net::IpAddr::V6(v) => v.segments()[0] & 0xffc0 == 0xfe80 }
                || match ip {
                    std::net::IpAddr::V4(v) => v.is_private() || v.octets()[0] == 169 && v.octets()[1] == 254,
                    std::net::IpAddr::V6(_) => false,
                };
            if bad { return Err("url host resolves to a private address".into()); }
        }
        let ip = addrs[0].ip().to_string();
        let out = tokio::process::Command::new("curl")
            .args(["-s","--max-time","180","--max-filesize",&MAX_UPLOAD_BYTES.to_string(),"--resolve"])
            .arg(format!("{host}:{port}:{ip}"))
            .args(["-D","-","--output","-"]).arg(&current)
            .output().await.map_err(|e| format!("download: {e}"))?;
        let headers = String::from_utf8_lossy(&out.stdout);
        if let Some(code) = headers.lines().next().and_then(|l| l.split_whitespace().nth(1)).and_then(|c| c.parse::<u16>().ok()) {
            if (300..400).contains(&code) {
                if let Some(loc) = headers.lines().find(|l| l.to_ascii_lowercase().starts_with("location:")) {
                    let next = loc[9..].trim().to_string();
                    let base = current.clone();
                    current = if next.starts_with("http") { next } else { format!("{base}/{}", next.trim_start_matches('/')) };
                    continue;
                }
            }
            if code != 200 { return Err(format!("download: http {code}")); }
        }
        // body follows the header block
        if let Some(pos) = out.stdout.windows(4).position(|w| w == b"\r\n\r\n") {
            let body = &out.stdout[pos + 4..];
            if body.len() as u64 > MAX_UPLOAD_BYTES { return Err("file too large".into()); }
            if body.is_empty() { return Err("download: empty body".into()); }
            return Ok(body.to_vec());
        }
        return Err("download: malformed response".into());
    }
    Err("download: too many redirects".into())
}

fn render(out: Value, format: &str) -> (String, String) {
    match format {
        "text" => ("text/plain; charset=utf-8".into(), out["text"].as_str().unwrap_or("").to_string()),
        "vtt" => ("text/vtt; charset=utf-8".into(), out["vtt"].as_str().unwrap_or("WEBVTT\n").to_string()),
        "json" => ("application/json".into(), json!({"text": out["text"]}).to_string()),
        _ => ("application/json".into(), out.to_string()),
    }
}

fn respond(out: Value, format: &str) -> axum::response::Response {
    let (ct, body) = render(out, format);
    (StatusCode::OK, [(axum::http::header::CONTENT_TYPE, ct.as_str())], body).into_response()
}

struct Word { word: String, start: f64, end: f64 }

async fn transcribe(app: &App, audio: &[u8]) -> Result<Value, String> {
    let dir = std::env::temp_dir().join(format!("qwen-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("tmp: {e}"))?;
    let raw = dir.join("in.bin");
    std::fs::write(&raw, audio).map_err(|e| format!("tmp: {e}"))?;
    let wav = dir.join("full.wav");
    let ok = tokio::process::Command::new("ffmpeg").args(["-v","error","-nostdin","-y","-i"]).arg(&raw)
        .args(["-ac","1","-ar","16000","-c:a","pcm_s16le"]).arg(&wav).output().await
        .map_err(|e| format!("ffmpeg unavailable: {e}"))?;
    if !ok.status.success() { return Err(format!("decode: not decodable audio")); }
    let pcm = std::fs::read(&wav).map_err(|e| format!("decode: {e}"))?;
    let samples: Vec<i16> = pcm.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let _ = std::fs::remove_file(&raw);
    let duration = samples.len() as f64 / 16000.0;
    if duration < 0.01 { return Err("decode: audio shorter than 10ms".into()); }

    let mut all_words: Vec<Word> = Vec::new();
    let app_asr = app.asr_arc();
    let app_align = app.align_arc();
    let _ = &app_align;
    let chunk = (CHUNK_SECONDS * 16000.0) as usize;
    let overlap = (OVERLAP_SECONDS * 16000.0) as usize;
    let regions = speech_regions_silero(&samples).unwrap_or_else(|| speech_regions_energy(&samples));
    let mut cuts: Vec<(usize, usize)> = Vec::new();
    {
        // Build chunk ranges over speech regions: cuts land in silence,
        // long non-speech gaps are skipped entirely (no ASR, no hallucination).
        let pad = (0.35 * 16000.0) as usize;
        let mut span: Option<(usize, usize)> = None;
        let mut flush = |span: &mut Option<(usize, usize)>, cuts: &mut Vec<(usize, usize)>| {
            if let Some((a, b)) = *span {
                let a = a.saturating_sub(pad);
                let b = (b + pad).min(samples.len());
                let mut start = a;
                while start < b {
                    let end = (start + chunk).min(b);
                    cuts.push((start, end));
                    if end >= b { break; }
                    start = end.saturating_sub(overlap).max(start + 1);
                }
            }
            *span = None;
        };
        for (rs, re) in regions {
            match span {
                Some((a, b)) if rs.saturating_sub(b) <= (1.5 * 16000.0) as usize => span = Some((a, re)),
                Some(_) => { flush(&mut span, &mut cuts); span = Some((rs, re)); }
                None => span = Some((rs, re)),
            }
        }
        flush(&mut span, &mut cuts);
        // no detected speech -> no chunks -> empty transcription (true silence)
    }
    // Pipelined chunks: align(chunk i-1) runs concurrently with asr(chunk i)
    // (separate engine processes; Metal orders per-process work correctly).
    let app_asr = app_asr.clone();
    let app_align = app_align.clone();
    let mut pending: Option<tokio::task::JoinHandle<Result<Vec<Word>, String>>> = None;
    for (pos, end) in cuts {
        let seg_bytes: Vec<u8> = samples[pos..end].iter().flat_map(|x| x.to_le_bytes()).collect();
        use base64::Engine;
        let seg = base64::engine::general_purpose::STANDARD.encode(&seg_bytes);
        let offset = pos as f64 / 16000.0;
        let text = {
            let mut engine = app_asr.lock().await;
            engine.infer(json!({"op":"infer","schema":1,"id":"c","task":"asr","audio_b64":seg,"language":"English"})).await?
                .get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string()
        };
        if let Some(handle) = pending.take() {
            all_words.extend(handle.await.map_err(|e| format!("align task: {e}"))??);
        }
        let align_text: String = text.split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>())
            .filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
        pending = if !align_text.is_empty() {
            let app = app_align.clone();
            let b64 = seg.clone();
            Some(tokio::task::spawn(async move {
                let mut engine = app.lock().await;
                let resp = engine.infer(json!({"op":"infer","schema":1,"id":"c","task":"align","audio_b64":b64,"text":align_text,"language":"English"})).await?;
                Ok(resp.get("words").and_then(|w| w.as_array()).map(|ws| ws.iter().filter_map(|w| {
                    let start = w["start_ms"].as_f64().unwrap_or(0.0) / 1000.0 + offset;
                    let end = w["end_ms"].as_f64().unwrap_or(0.0) / 1000.0 + offset;
                    w["text"].as_str().map(|word| Word { word: word.to_string(), start: start.max(0.0), end: end.max(start) })
                }).collect::<Vec<_>>()).unwrap_or_default())
            }))
        } else { None };
    }
    if let Some(handle) = pending {
        all_words.extend(handle.await.map_err(|e| format!("align task: {e}"))??);
    }
    let _ = std::fs::remove_dir_all(&dir);

    // Overlap dedup: drop the first word(s) of the next chunk that repeat the
    // previous chunk's tail and overlap it in time (hard-cut safety only).
    // Hard-cut overlap dedup: within the overlap window a repeated identical
    // word that time-overlaps the previous chunk's tail is a duplicate.
    let mut words: Vec<Word> = Vec::new();
    for w in all_words {
        if let Some(prev) = words.last() {
            if w.word == prev.word && w.start < prev.end + 0.02 { continue; }
        }
        words.push(w);
    }
    for w in words.iter_mut() { w.end = w.end.min(duration); }

    let text = words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
    let segments = build_segments(&words);
    let vtt = build_vtt(&segments);
    Ok(json!({
        "text": text,
        "word_count": words.len(),
        "words": words.iter().map(|w| json!({"word": w.word, "start": round3(w.start), "end": round3(w.end)})).collect::<Vec<_>>(),
        "segments": segments.iter().map(|s| json!({
            "start": round3(s.0), "end": round3(s.1), "text": s.2,
            "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0,
            "words": s.3.iter().map(|w| json!({"word": w.word, "start": round3(w.start), "end": round3(w.end)})).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "vtt": vtt,
        "transcription_info": {"language": "en", "language_probability": 1.0, "duration": round3(duration), "duration_after_vad": round3(duration)},
    }))
}


// ---- Silero VAD v6.2.2 (ONNX via ort; CPU) -----------------------------------
// Interface per upstream examples/cpp: input = [1, 576] (512-sample chunk with
// 64-sample carried context), state = [2,1,128], sr = 16000; outputs
// "output" (speech prob) and "stateN". Replaces the energy segmenter; energy
// remains the fallback when the model or runtime is unavailable.
mod vad {
    pub struct Silero {
        session: ort::session::Session,
        state: Vec<f32>,
        context: Vec<f32>,
    }
    const WINDOW: usize = 512;
    const CONTEXT: usize = 64;
    impl Silero {
        pub fn new() -> Result<Self, String> {
            let path = std::env::var("QWEN_SILENO_PATH").unwrap_or_else(|_| "server/assets/silero-vad.onnx".into());
            let session = ort::session::Session::builder().map_err(|e| e.to_string())?
                .commit_from_file(&path).map_err(|e| e.to_string())?;
            Ok(Silero { session, state: vec![0.0; 2 * 128], context: vec![0.0; CONTEXT] })
        }
        pub fn reset(&mut self) {
            self.state.iter_mut().for_each(|v| *v = 0.0);
            self.context.iter_mut().for_each(|v| *v = 0.0);
        }
        pub fn predict(&mut self, chunk: &[f32]) -> Option<f32> {
            let mut input = self.context.clone();
            input.extend_from_slice(chunk);
            let input = ort::value::Tensor::from_array(([1usize, WINDOW + CONTEXT], input)).ok()?;
            let state = ort::value::Tensor::from_array(([2usize, 1usize, 128usize], self.state.clone())).ok()?;
            let sr = ort::value::Tensor::from_array(([1i64], vec![16000i64])).ok()?;
            let outputs = self.session.run(ort::inputs!["input" => input, "state" => state, "sr" => sr]).ok()?;
            let (_shape, data) = outputs["output"].try_extract_tensor::<f32>().ok()?;
            let prob = data.first().copied()?;
            if let Ok((_s, data)) = outputs["stateN"].try_extract_tensor::<f32>() {
                if data.len() == self.state.len() { self.state = data.to_vec(); }
            }
            self.context.copy_from_slice(&chunk[chunk.len() - CONTEXT..]);
            Some(prob)
        }
    }
}
static SILENO: std::sync::OnceLock<Option<vad::Silero>> = std::sync::OnceLock::new();
static SILENO_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn speech_regions_silero(samples: &[i16]) -> Option<Vec<(usize, usize)>> {
    if std::env::var("QWEN_SILENO").ok().as_deref() == Some("0") { return None; }
    let holder = SILENO.get_or_init(|| {
        let r = vad::Silero::new();
        eprintln!("silero-vad init: {}", if r.is_ok() { "ok" } else { "FAILED (energy fallback)" });
        r.ok()
    });
    let vad = holder.as_ref()?;
    let mut vad = vad as *const vad::Silero as *mut vad::Silero; // interior singleton
    let _guard = SILENO_LOCK.lock().unwrap();
    unsafe { (*vad).reset(); }
    let window = 512usize;
    let enter = 0.5f32; let exit = 0.35f32;
    let mut regions: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    let mut low = 0usize;
    for (i, w) in samples.chunks(window).enumerate() {
        let floats: Vec<f32> = w.iter().map(|&x| x as f32 / 32768.0).collect();
        if floats.len() < window { break; }
        let prob = unsafe { (*vad).predict(&floats) }?;
        let at = i * window;
        match start {
            None => if prob > enter { start = Some(at); low = 0; },
            Some(a) => {
                if prob < exit { low += window; if low >= 25 * window { let _ = a; regions.push((a, at + window - low)); start = None; } }
                else { low = 0; }
            }
        }
    }
    if let Some(a) = start { regions.push((a, samples.len())); }
    // merge gaps below 1.5s (downstream re-applies this too, keep for clarity)
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for r in regions {
        match merged.last_mut() {
            Some(last) if r.0.saturating_sub(last.1) <= 24000 => last.1 = last.1.max(r.1),
            _ => merged.push(r),
        }
    }
    if merged.is_empty() { Some(vec![]) } else { Some(merged) }
}

// Speech regions via RMS energy with an adaptive noise floor (dependency-free
// fallback segmentation; Silero v6.2.2 will replace this exact interface).
fn speech_regions_energy(samples: &[i16]) -> Vec<(usize, usize)> {
    let win = 320usize; // 20ms
    let frames: Vec<f64> = samples.chunks(win).map(|c| {
        (c.iter().map(|&x| (x as f64 / 32768.0) * (x as f64 / 32768.0)).sum::<f64>() / c.len() as f64).sqrt()
    }).collect();
    let mut sorted = frames.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let floor = sorted[sorted.len() / 10].max(1e-5); // 10th percentile noise floor
    let thresh = (floor * 3.0).max(0.004);
    let min_speech = 12usize; // 240ms minimum speech region
    let mut regions = Vec::new();
    let mut start: Option<usize> = None;
    let mut quiet = 0usize;
    for (i, e) in frames.iter().enumerate() {
        if *e > thresh {
            if start.is_none() { start = Some(i); }
            quiet = 0;
        } else if let Some(a) = start {
            quiet += 1;
            if quiet >= 25 { // 500ms trailing silence ends the region
                let last = i - quiet + 1;
                if last > a + min_speech { regions.push((a * win, last * win)); }
                start = None;
            }
        }
    }
    if let Some(a) = start {
        if frames.len() > a + min_speech { regions.push((a * win, frames.len() * win)); }
    }
    regions
}

fn build_segments(words: &[Word]) -> Vec<(f64, f64, String, Vec<&Word>)> {
    let mut segments: Vec<(f64, f64, String, Vec<&Word>)> = Vec::new();
    let mut current: Vec<&Word> = Vec::new();
    fn flush<'a>(cur: Vec<&'a Word>, out: &mut Vec<(f64,f64,String,Vec<&'a Word>)>) {
        if cur.is_empty() { return; }
        let text = cur.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
        let (start, end) = (cur.first().unwrap().start, cur.last().unwrap().end);
        out.push((start, end, text, cur));
    }
    for (i, w) in words.iter().enumerate() {
        current.push(w);
        let sentence_end = w.word.ends_with('.') || w.word.ends_with('?') || w.word.ends_with('!') || w.word.ends_with(',') || w.word.ends_with(':') || w.word.ends_with(';');
        let big_gap = words.get(i+1).map(|n| n.start - w.end).unwrap_or(f64::INFINITY) > SEGMENT_GAP;
        let too_long = current.first().map(|f| w.end - f.start).unwrap_or(0.0) > SEGMENT_SOFT_CAP;
        if sentence_end || big_gap || too_long { flush(std::mem::take(&mut current), &mut segments); }
    }
    flush(std::mem::take(&mut current), &mut segments);
    // Soft-cap pass: any remaining overlong segment split at its largest gap.
    let mut result = Vec::new();
    for (start, end, text, ws) in segments {
        if end - start <= SEGMENT_SOFT_CAP || ws.len() < 3 { result.push((start, end, text, ws)); continue; }
        let mut best = 1; let mut best_gap = -1.0;
        for i in 1..ws.len()-1 { let g = ws[i+1].start - ws[i].end; if g > best_gap { best_gap = g; best = i; } }
        let (a, b) = ws.split_at(best + 1);
        for part in [a, b] {
            if !part.is_empty() {
                let t = part.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
                result.push((part.first().unwrap().start, part.last().unwrap().end, t, part.to_vec()));
            }
        }
    }
    result
}

fn build_vtt(segments: &[(f64, f64, String, Vec<&Word>)]) -> String {
    if segments.is_empty() { return String::new(); }
    let mut cues = Vec::new();
    for (i, (s, e, text, _)) in segments.iter().enumerate() {
        cues.push(format!("{}\n{} --> {}\n{}", i + 1, vtt_ts(*s), vtt_ts(*e), text.replace('\n', " ")));
    }
    format!("WEBVTT\n\n{}", cues.join("\n\n"))
}

fn vtt_ts(t: f64) -> String {
    let t = if t.is_finite() && t > 0.0 { t } else { 0.0 };
    let ms = (t * 1000.0).floor() as u64;
    format!("{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, (ms % 3_600_000) / 60_000, (ms % 60_000) / 1000, ms % 1000)
}

fn round3(v: f64) -> f64 { if v.is_finite() { (v * 1000.0).round() / 1000.0 } else { 0.0 } }

// Minimal 16kHz mono f32 "WAV": engine's load_wav expects RIFF PCM; we emit
// f32 sample data as a raw container the engine reads via the same ffmpeg path
// upstream, so here we only need bytes for OUR pipeline (ffmpeg wrote full.wav,
// chunks go through this helper as standard f32 WAVs).
fn write_wav_s16(path: &Path, pcm: &[i16]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let data_len = (pcm.len() * 2) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&16000u32.to_le_bytes())?;
    f.write_all(&32000u32.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for s in pcm { f.write_all(&s.to_le_bytes())?; }
    Ok(())
}

