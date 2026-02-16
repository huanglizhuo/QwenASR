/// GPT-2 byte-level BPE tokenizer for Qwen.

use std::collections::HashMap;

// GPT-2 bytes-to-unicode mapping
fn init_gpt2_mapping() -> ([i32; 256], [i32; 512]) {
    let mut byte_to_unicode = [0i32; 256];
    let mut unicode_to_byte = [-1i32; 512];

    let mut n = 0i32;
    for b in 0..256i32 {
        let is_normal = (b >= 33 && b <= 126)
            || (b >= 161 && b <= 172)
            || (b >= 174 && b <= 255);

        if is_normal {
            byte_to_unicode[b as usize] = b;
        } else {
            byte_to_unicode[b as usize] = 256 + n;
            n += 1;
        }
    }

    for b in 0..256 {
        let cp = byte_to_unicode[b] as usize;
        if cp < 512 {
            unicode_to_byte[cp] = b as i32;
        }
    }

    (byte_to_unicode, unicode_to_byte)
}

fn utf8_encode_cp(cp: u32) -> Vec<u8> {
    if cp < 0x80 {
        vec![cp as u8]
    } else if cp < 0x800 {
        vec![
            (0xC0 | (cp >> 6)) as u8,
            (0x80 | (cp & 0x3F)) as u8,
        ]
    } else {
        vec![
            (0xE0 | (cp >> 12)) as u8,
            (0x80 | ((cp >> 6) & 0x3F)) as u8,
            (0x80 | (cp & 0x3F)) as u8,
        ]
    }
}

/// Decode a GPT-2 encoded token string (vocab key) to raw bytes.
fn decode_gpt2_token(token_str: &str, unicode_to_byte: &[i32; 512]) -> String {
    let mut bytes = Vec::new();

    for ch in token_str.chars() {
        let cp = ch as u32;
        if cp < 512 && unicode_to_byte[cp as usize] >= 0 {
            bytes.push(unicode_to_byte[cp as usize] as u8);
        } else {
            bytes.push(b'?');
        }
    }

    String::from_utf8_lossy(&bytes).into_owned()
}

/// Convert UTF-8 bytes to GPT-2 byte-level unicode string.
fn text_to_bpe_unicode(text: &str, byte_to_unicode: &[i32; 256]) -> String {
    let mut out = String::new();
    for &b in text.as_bytes() {
        let cp = byte_to_unicode[b as usize] as u32;
        for byte in utf8_encode_cp(cp) {
            out.push(byte as char);
        }
    }
    // Actually, we need to push the encoded codepoint as a char
    let mut out2 = String::new();
    for &b in text.as_bytes() {
        let cp = byte_to_unicode[b as usize] as u32;
        if let Some(ch) = char::from_u32(cp) {
            out2.push(ch);
        }
    }
    out2
}

fn utf8_char_len(c: u8) -> usize {
    if c & 0x80 == 0 { 1 }
    else if c & 0xE0 == 0xC0 { 2 }
    else if c & 0xF0 == 0xE0 { 3 }
    else if c & 0xF8 == 0xF0 { 4 }
    else { 1 }
}

fn split_utf8_symbols(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut syms = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let len = utf8_char_len(bytes[i]);
        let end = (i + len).min(bytes.len());
        if let Ok(ch) = std::str::from_utf8(&bytes[i..end]) {
            syms.push(ch.to_string());
        }
        i = end;
    }
    syms
}

fn fnv1a_hash(s: &str) -> u64 {
    let mut h = 1469598103934665603u64;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211u64);
    }
    h
}

pub struct QwenTokenizer {
    pub vocab_size: usize,
    id_to_text: Vec<Option<String>>,
    id_to_bpe: Vec<Option<String>>,
    vocab_map: HashMap<String, i32>,
    merge_map: HashMap<String, i32>,
    byte_to_unicode: [i32; 256],
    unicode_to_byte: [i32; 512],
}

impl QwenTokenizer {
    pub fn load(vocab_json_path: &str) -> Option<Self> {
        let (byte_to_unicode, unicode_to_byte) = init_gpt2_mapping();

        // Read vocab.json
        let json = std::fs::read_to_string(vocab_json_path).ok()?;

        // Parse vocab.json: { "token": id, ... }
        let mut max_id = 0i32;
        let mut entries: Vec<(String, i32)> = Vec::new();

        let bytes = json.as_bytes();
        let mut pos = 0;
        skip_ws(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] != b'{' {
            return None;
        }
        pos += 1;

        loop {
            skip_ws(bytes, &mut pos);
            if pos >= bytes.len() || bytes[pos] == b'}' {
                break;
            }
            if bytes[pos] == b',' {
                pos += 1;
                continue;
            }

            let key = parse_json_string_tok(bytes, &mut pos)?;
            skip_ws(bytes, &mut pos);
            if pos >= bytes.len() || bytes[pos] != b':' {
                return None;
            }
            pos += 1;
            let id = parse_json_int_tok(bytes, &mut pos)? as i32;

            if id > max_id {
                max_id = id;
            }
            entries.push((key, id));
        }

        let vocab_size = (max_id + 1) as usize;
        let mut id_to_text = vec![None; vocab_size];
        let mut id_to_bpe = vec![None; vocab_size];
        let mut vocab_map = HashMap::new();

        for (key, id) in entries {
            let idx = id as usize;
            if idx < vocab_size {
                let text = decode_gpt2_token(&key, &unicode_to_byte);
                id_to_text[idx] = Some(text);
                vocab_map.insert(key.clone(), id);
                id_to_bpe[idx] = Some(key);
            }
        }

        // Load merges.txt
        let merge_map = load_merges(vocab_json_path);

        Some(QwenTokenizer {
            vocab_size,
            id_to_text,
            id_to_bpe,
            vocab_map,
            merge_map,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    pub fn decode(&self, token_id: i32) -> &str {
        if token_id < 0 || token_id as usize >= self.vocab_size {
            return "";
        }
        match &self.id_to_text[token_id as usize] {
            Some(s) => s.as_str(),
            None => "",
        }
    }

    pub fn encode(&self, text: &str) -> Option<Vec<i32>> {
        if text.is_empty() {
            return None;
        }

        let mapped = text_to_bpe_unicode(text, &self.byte_to_unicode);
        let ids = self.encode_bpe_word(&mapped)?;
        Some(ids)
    }

    fn encode_bpe_word(&self, mapped: &str) -> Option<Vec<i32>> {
        if mapped.is_empty() {
            return Some(Vec::new());
        }

        let mut syms = split_utf8_symbols(mapped);
        if syms.is_empty() {
            return Some(Vec::new());
        }

        while syms.len() > 1 {
            let mut best_rank = i32::MAX;
            let mut best_i = -1i32;

            for i in 0..syms.len() - 1 {
                let pair = format!("{} {}", syms[i], syms[i + 1]);
                if let Some(&rank) = self.merge_map.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i as i32;
                    }
                }
            }

            if best_i < 0 || best_rank == i32::MAX {
                break;
            }

            let i = best_i as usize;
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }

        let mut ids = Vec::new();
        for sym in &syms {
            let id = self.vocab_map.get(sym.as_str()).copied()?;
            ids.push(id);
        }

        Some(ids)
    }
}

fn load_merges(vocab_path: &str) -> HashMap<String, i32> {
    let mut merge_map = HashMap::new();

    // Derive merges.txt path from vocab.json path
    let merges_path = if let Some(slash) = vocab_path.rfind('/') {
        format!("{}/merges.txt", &vocab_path[..slash])
    } else {
        "merges.txt".to_string()
    };

    let content = match std::fs::read_to_string(&merges_path) {
        Ok(c) => c,
        Err(_) => return merge_map,
    };

    let mut rank = 0i32;
    for line in content.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(space_pos) = line.find(' ') {
            let a = &line[..space_pos];
            let b = line[space_pos + 1..].trim_start();
            if !a.is_empty() && !b.is_empty() {
                let key = format!("{} {}", a, b);
                merge_map.insert(key, rank);
                rank += 1;
            }
        }
    }

    merge_map
}

// Minimal JSON parsing helpers
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() {
        match bytes[*pos] {
            b' ' | b'\n' | b'\r' | b'\t' => *pos += 1,
            _ => break,
        }
    }
}

fn parse_json_string_tok(bytes: &[u8], pos: &mut usize) -> Option<String> {
    skip_ws(bytes, pos);
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;

    let mut result = Vec::new();
    while *pos < bytes.len() && bytes[*pos] != b'"' {
        if bytes[*pos] == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() {
                return None;
            }
            match bytes[*pos] {
                b'n' => result.push(b'\n'),
                b't' => result.push(b'\t'),
                b'"' => result.push(b'"'),
                b'\\' => result.push(b'\\'),
                b'/' => result.push(b'/'),
                b'u' => {
                    *pos += 1;
                    let mut cp = 0u32;
                    for _ in 0..4 {
                        if *pos >= bytes.len() {
                            return None;
                        }
                        cp <<= 4;
                        let c = bytes[*pos];
                        cp |= match c {
                            b'0'..=b'9' => (c - b'0') as u32,
                            b'a'..=b'f' => (c - b'a' + 10) as u32,
                            b'A'..=b'F' => (c - b'A' + 10) as u32,
                            _ => return None,
                        };
                        *pos += 1;
                    }
                    if let Some(ch) = char::from_u32(cp) {
                        let mut buf = [0u8; 4];
                        let s = ch.encode_utf8(&mut buf);
                        result.extend_from_slice(s.as_bytes());
                    }
                    continue;
                }
                other => result.push(other),
            }
        } else {
            result.push(bytes[*pos]);
        }
        *pos += 1;
    }

    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;

    String::from_utf8(result).ok()
}

fn parse_json_int_tok(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    skip_ws(bytes, pos);
    let mut neg = false;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        neg = true;
        *pos += 1;
    }
    let mut val: i64 = 0;
    let mut found = false;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        val = val * 10 + (bytes[*pos] - b'0') as i64;
        *pos += 1;
        found = true;
    }
    if !found {
        return None;
    }
    Some(if neg { -val } else { val })
}
