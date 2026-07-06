use qwen_asr::align::AlignResult;
use qwen_asr::subtitle::{format_srt, format_vtt, group_words_to_cues, Cue};

fn word(text: &str, start_ms: f32, end_ms: f32) -> AlignResult {
    AlignResult {
        text: text.to_string(),
        start_ms,
        end_ms,
    }
}

fn golden_words() -> Vec<AlignResult> {
    vec![
        word("Shenyang,", 1120.0, 1120.0),
        word("a", 1120.0, 1440.0),
        word("city", 4080.0, 4960.0),
        word("with", 5680.0, 5920.0),
        word("its", 5920.0, 6160.0),
        word("own", 6320.0, 6720.0),
        word("small", 6720.0, 7280.0),
        word("secrets.", 9760.0, 9760.0),
        word("Since", 9760.0, 10240.0),
        word("you", 10240.0, 10400.0),
        word("are", 10400.0, 10720.0),
        word("going", 10720.0, 11200.0),
        word("there,", 11360.0, 11440.0),
        word("I", 12720.0, 12800.0),
        word("expect", 12800.0, 13760.0),
        word("you", 13760.0, 13840.0),
        word("to", 13840.0, 14080.0),
        word("keep", 14080.0, 14400.0),
        word("your", 14400.0, 14720.0),
        word("eyes", 14720.0, 15200.0),
        word("open.", 15600.0, 15680.0),
        word("Some", 17360.0, 17760.0),
        word("things", 17760.0, 18160.0),
        word("are", 18160.0, 18560.0),
        word("worth", 19280.0, 19680.0),
        word("bringing", 19680.0, 20240.0),
        word("back,", 22480.0, 22480.0),
        word("and", 22480.0, 22720.0),
        word("you", 22720.0, 22880.0),
        word("know", 22880.0, 23360.0),
        word("disappointing", 23600.0, 24640.0),
        word("me", 24640.0, 25520.0),
        word("is", 26000.0, 26160.0),
        word("rarely", 26160.0, 26880.0),
        word("a", 27040.0, 27040.0),
        word("wise", 27040.0, 27600.0),
        word("decision.", 28160.0, 28160.0),
    ]
}

#[test]
fn groups_issue_39_words_into_sentence_cues() {
    let cues = group_words_to_cues(&golden_words(), 28560);
    assert_eq!(
        cues,
        vec![
            Cue {
                start_ms: 1120,
                end_ms: 9760,
                text: "Shenyang, a city with its own small secrets.".to_string(),
            },
            Cue {
                start_ms: 9760,
                end_ms: 15980,
                text: "Since you are going there, I expect you to keep your eyes open.".to_string(),
            },
            Cue {
                start_ms: 17360,
                end_ms: 22480,
                text: "Some things are worth bringing back,".to_string(),
            },
            Cue {
                start_ms: 22480,
                end_ms: 28540,
                text: "and you know disappointing me is rarely a wise decision.".to_string(),
            },
        ]
    );
}

#[test]
fn repairs_zero_duration_words() {
    let cues = group_words_to_cues(&[word("Hi.", 500.0, 500.0)], 2000);
    assert_eq!(cues[0].start_ms, 500);
    assert_eq!(cues[0].end_ms, 1500);
}

#[test]
fn respects_cjk_width_budget_without_spaces() {
    let words: Vec<_> = (0..40)
        .map(|i| word("汉字", (i * 100) as f32, (i * 100 + 80) as f32))
        .collect();
    let cues = group_words_to_cues(&words, 5000);
    assert_eq!(cues.len(), 2);
    assert!(!cues[0].text.contains(' '));
}

#[test]
fn falls_back_to_soft_break_on_width_overflow() {
    let cues = group_words_to_cues(
        &[
            word("alpha,", 0.0, 100.0),
            word("bravo", 100.0, 200.0),
            word("charlie", 200.0, 300.0),
            word(
                "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                300.0,
                400.0,
            ),
            word("done.", 400.0, 500.0),
        ],
        3000,
    );
    assert_eq!(cues[0].text, "alpha,");
    assert_eq!(
        cues[1].text,
        "bravo charlie xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    );
    assert_eq!(cues[2].text, "done.");
}

#[test]
fn splits_on_duration_overflow() {
    let cues = group_words_to_cues(
        &[
            word("one", 0.0, 1000.0),
            word("two,", 1000.0, 1100.0),
            word("two", 3000.0, 4000.0),
            word("three", 6200.0, 6300.0),
            word("done.", 6300.0, 6400.0),
        ],
        8000,
    );
    assert_eq!(cues[0].text, "one two,");
    assert_eq!(cues[1].text, "two three done.");
}

#[test]
fn formats_srt_and_vtt() {
    let cues = vec![
        Cue {
            start_ms: 1120,
            end_ms: 9760,
            text: "First cue.".to_string(),
        },
        Cue {
            start_ms: 9760,
            end_ms: 15980,
            text: "Second cue.".to_string(),
        },
    ];
    assert_eq!(
        format_srt(&cues),
        "1\n00:00:01,120 --> 00:00:09,760\nFirst cue.\n\n2\n00:00:09,760 --> 00:00:15,980\nSecond cue.\n\n"
    );
    assert_eq!(
        format_vtt(&cues),
        "WEBVTT\n\n1\n00:00:01.120 --> 00:00:09.760\nFirst cue.\n\n2\n00:00:09.760 --> 00:00:15.980\nSecond cue.\n\n"
    );
}
