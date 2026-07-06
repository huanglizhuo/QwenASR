//! Structured transcription output and JSON serialization.

use crate::subtitle::is_cjk_width_char;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordTimestamp {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentResult {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionResult {
    pub language: String,
    pub duration_ms: u64,
    pub text: String,
    pub segments: Vec<SegmentResult>,
    pub vtt: String,
}

impl TranscriptionResult {
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"transcription_info\": {\n");
        push_json_string_field(&mut out, 4, "language", &self.language, true);
        push_json_number_field(&mut out, 4, "duration", &secs(self.duration_ms), false);
        out.push_str("  },\n");
        push_json_string_field(&mut out, 2, "text", &self.text, true);
        push_json_number_field(
            &mut out,
            2,
            "word_count",
            &count_words(&self.text).to_string(),
            true,
        );
        out.push_str("  \"segments\": [\n");
        for (idx, segment) in self.segments.iter().enumerate() {
            out.push_str("    {\n");
            push_json_number_field(&mut out, 6, "start", &secs(segment.start_ms), true);
            push_json_number_field(&mut out, 6, "end", &secs(segment.end_ms), true);
            push_json_string_field(&mut out, 6, "text", &segment.text, true);
            out.push_str("      \"words\": [\n");
            for (word_idx, word) in segment.words.iter().enumerate() {
                out.push_str("        { ");
                out.push_str("\"word\": ");
                push_json_string_value(&mut out, &word.word);
                out.push_str(", \"start\": ");
                out.push_str(&secs(word.start_ms));
                out.push_str(", \"end\": ");
                out.push_str(&secs(word.end_ms));
                out.push_str(" }");
                if word_idx + 1 < segment.words.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str("      ],\n");
            push_json_number_field(
                &mut out,
                6,
                "word_count",
                &count_words(&segment.text).to_string(),
                false,
            );
            out.push_str("    }");
            if idx + 1 < self.segments.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n");
        push_json_string_field(&mut out, 2, "vtt", &self.vtt, false);
        out.push_str("}\n");
        out
    }
}

pub fn count_words(text: &str) -> usize {
    let mut count = 0usize;
    let mut in_non_cjk_word = false;

    for ch in text.chars() {
        if is_cjk_width_char(ch) {
            if in_non_cjk_word {
                count += 1;
                in_non_cjk_word = false;
            }
            count += 1;
        } else if ch.is_whitespace() {
            if in_non_cjk_word {
                count += 1;
                in_non_cjk_word = false;
            }
        } else {
            in_non_cjk_word = true;
        }
    }

    if in_non_cjk_word {
        count += 1;
    }

    count
}

fn secs(ms: u64) -> String {
    format!("{:.3}", ms as f64 / 1000.0)
}

fn push_json_string_field(out: &mut String, indent: usize, key: &str, value: &str, comma: bool) {
    push_indent(out, indent);
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
    push_json_string_value(out, value);
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_json_number_field(out: &mut String, indent: usize, key: &str, value: &str, comma: bool) {
    push_indent(out, indent);
    out.push('"');
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(value);
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_json_string_value(out: &mut String, value: &str) {
    out.push('"');
    escape_json_string_into(out, value);
    out.push('"');
}

fn escape_json_string_into(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                out.push_str("\\u00");
                let code = ch as u8;
                out.push(nibble_to_hex(code >> 4));
                out.push(nibble_to_hex(code & 0x0f));
            }
            _ => out.push(ch),
        }
    }
}

fn nibble_to_hex(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'A' + nibble - 10) as char,
        _ => unreachable!(),
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_english_chinese_and_mixed_text() {
        assert_eq!(count_words("hello world again"), 3);
        assert_eq!(count_words("你好世界"), 4);
        assert_eq!(count_words("hello 世界 again"), 4);
    }

    #[test]
    fn serializes_pretty_json_with_escaping() {
        let result = TranscriptionResult {
            language: "zh".to_string(),
            duration_ms: 1234,
            text: "hello \"世界\"\npath\\x\u{001f}".to_string(),
            segments: vec![SegmentResult {
                start_ms: 0,
                end_ms: 1234,
                text: "hello 世界".to_string(),
                words: vec![WordTimestamp {
                    word: "hello".to_string(),
                    start_ms: 120,
                    end_ms: 450,
                }],
            }],
            vtt: "WEBVTT\n\n1\n00:00:00.120 --> 00:00:00.450\nhello\n\n".to_string(),
        };

        assert_eq!(
            result.to_json(),
            "{\n  \"transcription_info\": {\n    \"language\": \"zh\",\n    \"duration\": 1.234\n  },\n  \"text\": \"hello \\\"世界\\\"\\npath\\\\x\\u001F\",\n  \"word_count\": 6,\n  \"segments\": [\n    {\n      \"start\": 0.000,\n      \"end\": 1.234,\n      \"text\": \"hello 世界\",\n      \"words\": [\n        { \"word\": \"hello\", \"start\": 0.120, \"end\": 0.450 }\n      ],\n      \"word_count\": 3\n    }\n  ],\n  \"vtt\": \"WEBVTT\\n\\n1\\n00:00:00.120 --> 00:00:00.450\\nhello\\n\\n\"\n}\n"
        );
    }
}
