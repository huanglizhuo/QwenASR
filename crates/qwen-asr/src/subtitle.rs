//! Subtitle cue grouping and SRT/WebVTT formatting.

use crate::align::AlignResult;
use crate::transcribe::TranscriptSegment;

const MAX_WIDTH: usize = 84;
const MAX_DUR_MS: f32 = 6000.0;
const MIN_DUR_MS: u64 = 1000;
const TAIL_MS: u64 = 300;
const TS_GRID_MS: f32 = 80.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone)]
struct Word {
    text: String,
    start_ms: f32,
    end_ms: f32,
}

pub fn group_words_to_cues(words: &[AlignResult], audio_end_ms: u64) -> Vec<Cue> {
    let words = normalize_words(words);
    if words.is_empty() {
        return Vec::new();
    }

    let mut groups: Vec<Vec<Word>> = Vec::new();
    let mut cur: Vec<Word> = Vec::new();
    let mut soft_break: Option<usize> = None;

    for (idx, word) in words.iter().enumerate() {
        if !cur.is_empty() && would_overflow(&cur, word, soft_break) {
            if let Some(split_at) = soft_break {
                let tail = cur.split_off(split_at);
                groups.push(cur);
                cur = tail;
            } else {
                groups.push(cur);
                cur = Vec::new();
            }
            soft_break = None;
        }

        cur.push(word.clone());

        if is_sentence_end(&word.text) || idx + 1 == words.len() {
            groups.push(cur);
            cur = Vec::new();
            soft_break = None;
        } else if is_clause_end(&word.text) {
            soft_break = Some(cur.len());
        }
    }

    if !cur.is_empty() {
        groups.push(cur);
    }

    groups_to_cues(groups, audio_end_ms)
}

pub fn segment_to_cue(seg: &TranscriptSegment) -> Cue {
    Cue {
        start_ms: seg.start_ms,
        end_ms: seg.end_ms,
        text: seg.text.trim().to_string(),
    }
}

pub fn format_srt(cues: &[Cue]) -> String {
    format_cues(cues, false, 1)
}

pub fn format_vtt(cues: &[Cue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    out.push_str(&format_cues(cues, true, 1));
    out
}

pub fn format_srt_from_index(cues: &[Cue], start_index: u32) -> String {
    format_cues(cues, false, start_index)
}

pub fn format_vtt_from_index(cues: &[Cue], start_index: u32, include_header: bool) -> String {
    let mut out = String::new();
    if include_header {
        out.push_str("WEBVTT\n\n");
    }
    out.push_str(&format_cues(cues, true, start_index));
    out
}

pub fn cue_count(cues: &[Cue]) -> u32 {
    cues.iter()
        .filter(|cue| !cue.text.trim().is_empty())
        .count() as u32
}

fn normalize_words(words: &[AlignResult]) -> Vec<Word> {
    words
        .iter()
        .map(|word| {
            let start_ms = word.start_ms;
            let mut end_ms = word.end_ms.max(start_ms);
            if end_ms == start_ms {
                end_ms = start_ms + TS_GRID_MS;
            }
            Word {
                text: word.text.clone(),
                start_ms,
                end_ms,
            }
        })
        .collect()
}

fn would_overflow(cur: &[Word], next: &Word, soft_break: Option<usize>) -> bool {
    let width_overflow = cue_width_with(cur, next) > MAX_WIDTH;
    let duration_overflow =
        soft_break.is_some_and(|idx| idx > 1) && next.end_ms - cur[0].start_ms > MAX_DUR_MS;
    width_overflow || duration_overflow
}

fn groups_to_cues(groups: Vec<Vec<Word>>, audio_end_ms: u64) -> Vec<Cue> {
    let mut cues: Vec<Cue> = groups
        .into_iter()
        .filter(|group| !group.is_empty())
        .map(|group| {
            let start_ms = group[0].start_ms as u64;
            let last = group.last().unwrap();
            let mut end_ms = last.end_ms as u64 + TAIL_MS;
            if end_ms.saturating_sub(start_ms) < MIN_DUR_MS {
                end_ms = start_ms + MIN_DUR_MS;
            }
            Cue {
                start_ms,
                end_ms,
                text: join_words(&group),
            }
        })
        .collect();

    for idx in 0..cues.len() {
        let clamp_to = if idx + 1 < cues.len() {
            cues[idx + 1].start_ms
        } else {
            audio_end_ms
        };
        cues[idx].end_ms = cues[idx].end_ms.min(clamp_to);
        if cues[idx].end_ms <= cues[idx].start_ms {
            cues[idx].end_ms = cues[idx].start_ms + TS_GRID_MS as u64;
        }
    }

    cues
}

fn cue_width_with(cur: &[Word], next: &Word) -> usize {
    let mut width = 0usize;
    for (idx, word) in cur.iter().chain(std::iter::once(next)).enumerate() {
        if idx > 0 {
            let prev = if idx == cur.len() {
                &cur[cur.len() - 1]
            } else {
                &cur[idx - 1]
            };
            if !both_cjk(&prev.text, &word.text) {
                width += 1;
            }
        }
        width += text_width(&word.text);
    }
    width
}

fn join_words(words: &[Word]) -> String {
    let mut out = String::new();
    for (idx, word) in words.iter().enumerate() {
        if idx > 0 && !both_cjk(&words[idx - 1].text, &word.text) {
            out.push(' ');
        }
        out.push_str(&word.text);
    }
    out
}

fn both_cjk(left: &str, right: &str) -> bool {
    left.chars().all(is_cjk_width_char) && right.chars().all(is_cjk_width_char)
}

pub fn is_cjk_width_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF
            | 0x3040..=0x30FF
            | 0xAC00..=0xD7AF
            | 0x3000..=0x303F
            | 0xFF00..=0xFF60
    )
}

fn text_width(text: &str) -> usize {
    text.chars()
        .map(|ch| if is_cjk_width_char(ch) { 2 } else { 1 })
        .sum()
}

fn sentence_tail(text: &str) -> Option<char> {
    text.trim_end_matches(|ch| matches!(ch, '"' | '\'' | '」' | '』' | '）' | ')' | ']' | '》'))
        .chars()
        .next_back()
}

fn is_sentence_end(text: &str) -> bool {
    matches!(
        sentence_tail(text),
        Some('.' | '!' | '?' | '…' | '。' | '！' | '？')
    )
}

fn is_clause_end(text: &str) -> bool {
    matches!(
        sentence_tail(text),
        Some(',' | ';' | ':' | '，' | '；' | '：' | '、')
    )
}

fn format_cues(cues: &[Cue], vtt: bool, start_index: u32) -> String {
    let mut out = String::new();
    let mut idx = start_index;
    for cue in cues {
        if cue.text.trim().is_empty() {
            continue;
        }
        out.push_str(&idx.to_string());
        out.push('\n');
        out.push_str(&format_time(cue.start_ms, vtt));
        out.push_str(" --> ");
        out.push_str(&format_time(cue.end_ms, vtt));
        out.push('\n');
        out.push_str(cue.text.trim());
        out.push_str("\n\n");
        idx += 1;
    }
    out
}

fn format_time(ms: u64, vtt: bool) -> String {
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    let sep = if vtt { '.' } else { ',' };
    format!("{:02}:{:02}:{:02}{}{:03}", h, m, s, sep, millis)
}
