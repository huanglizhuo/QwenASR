//! Persistent benchmark adapter. CPU work is limited to WAV I/O, tokenizer,
//! scheduling and final text/timestamp formatting. No QwenCtx/CPU inference.
use qwen_asr::{audio, tokenizer::QwenTokenizer};
use serde_json::{json, Value};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    io::{self, BufRead, Write},
    time::Instant,
};
mod timestamp;
#[repr(C)]
#[derive(Default)]
struct Stats {
    gpu_ms: f64,
    audio_tokens: u32,
    decoder_positions: u32,
    generated: u32,
}
#[link(name = "qwen_metal", kind = "static")]
#[link(name = "Foundation", kind = "framework")]
#[link(name = "Metal", kind = "framework")]
#[link(name = "c++")]
#[link(name = "objc")]
extern "C" {
    fn qmetal_create(
        package: *const c_char,
        kernels: *const c_char,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn qmetal_destroy(handle: *mut c_void);
    fn qmetal_infer(
        handle: *mut c_void,
        pcm: *const f32,
        samples: usize,
        prefix: *const i32,
        prefix_count: usize,
        suffix: *const i32,
        suffix_count: usize,
        rows: *const i32,
        row_count: usize,
        max_new: u32,
        output: *mut i32,
        output_capacity: usize,
        stats: *mut Stats,
        error: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}
struct Engine(*mut c_void);
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { qmetal_destroy(self.0) }
    }
}
fn emit(v: Value) {
    println!("{v}");
    io::stdout().flush().expect("flush response");
}
fn required<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing nonempty string: {key}"))
}
fn run(engine: &Engine, tok: &QwenTokenizer, task: &str, v: &Value) -> Result<Value, String> {
    if v["schema"].as_u64() != Some(1)
        || required(v, "op")? != "infer"
        || required(v, "task")? != task
    {
        return Err("invalid request contract".into());
    }
    required(v, "id")?;
    if required(v, "language")? != "English" {
        return Err("v1 benchmark fixes language to English".into());
    }
    let start = Instant::now();
    let pcm = if let Some(raw) = v.get("audio_b64").and_then(Value::as_str) {
        // In-memory s16le mono 16k PCM (long-audio server path; no temp WAV).
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(raw).map_err(|e| format!("audio_b64 decode: {e}"))?;
        if bytes.is_empty() || bytes.len() % 2 != 0 {
            return Err("audio_b64 must be nonempty even-length s16le".into());
        }
        bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect::<Vec<f32>>()
    } else {
        audio::load_wav(required(v, "audio")?).ok_or("cannot read WAV")?
    };
    let input_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let prefix = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669];
    let mut suffix = vec![151670, 151645, 198, 151644, 77091, 198];
    let mut positions = Vec::new();
    let mut words = Vec::new();
    if task == "asr" {
        suffix.extend(
            tok.encode("language English")
                .ok_or("language tokenization failed")?,
        );
        suffix.push(151704);
    } else {
        for word in required(v, "text")?.split_whitespace() {
            words.push(word.to_string());
            suffix.extend(tok.encode(word).ok_or("text tokenization failed")?);
            positions.push(suffix.len() as i32);
            suffix.push(151705);
            positions.push(suffix.len() as i32);
            suffix.push(151705);
        }
        if words.is_empty() {
            return Err("empty alignment text".into());
        }
    }
    let mut output = vec![0i32; if task == "asr" { 2048 } else { positions.len() }];
    let mut stats = Stats::default();
    let mut error = [0 as c_char; 4096];
    let count = unsafe {
        qmetal_infer(
            engine.0,
            pcm.as_ptr(),
            pcm.len(),
            prefix.as_ptr(),
            prefix.len(),
            suffix.as_ptr(),
            suffix.len(),
            positions.as_ptr(),
            positions.len(),
            2048,
            output.as_mut_ptr(),
            output.len(),
            &mut stats,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if count < 0 {
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned());
    }
    output.truncate(count as usize);
    let (text, aligned) = if task == "asr" {
        let mut bytes = Vec::new();
        for &id in &output {
            if id == 151643 || id == 151645 {
                break;
            }
            bytes.extend(tok.decode_bytes(id));
        }
        (
            String::from_utf8_lossy(&bytes).trim().to_string(),
            Vec::new(),
        )
    } else {
        if output.len() != words.len() * 2 {
            return Err("timestamp count mismatch".into());
        }
        let mut times: Vec<f32> = output.iter().map(|&id| id as f32 * 80.0).collect();
        timestamp::fix_timestamps(&mut times);
        let aligned: Vec<Value> = words
            .iter()
            .enumerate()
            .map(|(i, w)| json!({"text":w,"start_ms":times[i*2],"end_ms":times[i*2+1]}))
            .collect();
        (required(v, "text")?.to_string(), aligned)
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(
        json!({"schema":1,"ok":true,"id":v["id"],"task":task,"backend":"metal","text":text,"words":aligned,
              "elapsed_ms":elapsed_ms,"input_ms":input_ms,"audio_samples":pcm.len(),"sample_rate":16000,
              "gpu_completed":true,"cpu_tensor_fallbacks":0,
              "stages":{"gpu_ms":stats.gpu_ms,"audio_tokens":stats.audio_tokens,"decoder_positions":stats.decoder_positions,"generated":stats.generated}}),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "asr" | "align") {
        return Err("usage: qwen-asr-gpu <asr|align> <package> <kernels>".into());
    }
    let start = Instant::now();
    let index: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/index.json", args[2]))?)?;
    if index["task"].as_str() != Some(&args[1]) {
        return Err("package task mismatch".into());
    }
    let tok =
        QwenTokenizer::load(&format!("{}/vocab.json", args[2])).ok_or("cannot load tokenizer")?;
    let package = CString::new(args[2].as_str())?;
    let kernels = CString::new(args[3].as_str())?;
    let mut error = [0 as c_char; 4096];
    let ptr = unsafe {
        qmetal_create(
            package.as_ptr(),
            kernels.as_ptr(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if ptr.is_null() {
        return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned()
            .into());
    }
    let engine = Engine(ptr);
    emit(
        json!({"schema":1,"ready":true,"backend":"metal","task":args[1],"contract":"qwen-asr-metal-bench-v1","load_ms":start.elapsed().as_secs_f64()*1000.0,
                "precision":index["precision"],"weights_sha256":index["weights_sha256"],
                "package_path":std::fs::canonicalize(&args[2])?,"kernel_directory":std::fs::canonicalize(&args[3])?}),
    );
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line?)?;
        match run(&engine, &tok, &args[1], &v) {
            Ok(value) => emit(value),
            Err(error) => {
                emit(json!({"schema":1,"ok":false,"id":v["id"],"error":error}));
                return Err("GPU request failed; engine closed".into());
            }
        }
    }
    Ok(())
}
