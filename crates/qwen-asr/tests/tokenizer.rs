use qwen_asr::tokenizer::QwenTokenizer;

fn get_tokenizer() -> Option<QwenTokenizer> {
    let path = "qwen3-asr-0.6b/vocab.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping tokenizer test: model not downloaded");
        return None;
    }
    QwenTokenizer::load(path)
}

#[test]
fn test_tokenizer_load() {
    if let Some(tok) = get_tokenizer() {
        // Check a normal token decodes correctly (e.g., "the" is a common BPE token)
        let hello = tok.decode(9707); // "Hello" in Qwen tokenizer
        assert!(!hello.is_empty(), "Should decode common token");
    }
}

#[test]
fn test_encode_decode_roundtrip() {
    let tok = match get_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let texts = vec![
        "Hello world",
        "And so, my fellow Americans",
        "The quick brown fox jumps over the lazy dog.",
        "speech-to-text",
        "12345",
    ];

    for text in texts {
        let tokens = tok.encode(text);
        assert!(tokens.is_some(), "Should encode: '{}'", text);
        let tokens = tokens.unwrap();
        assert!(!tokens.is_empty(), "Encoding should produce tokens for: '{}'", text);

        let mut decoded = String::new();
        for &t in &tokens {
            decoded.push_str(tok.decode(t));
        }
        assert_eq!(decoded, text,
            "Round-trip failed for '{}': got '{}'", text, decoded);
    }
}

#[test]
fn test_special_tokens_not_in_vocab() {
    let tok = match get_tokenizer() {
        Some(t) => t,
        None => return,
    };

    // Special tokens (151643+) are NOT in vocab.json, decode returns ""
    let eos = tok.decode(151643);
    assert!(eos.is_empty(), "EOS token should decode to empty (not in vocab.json)");

    let im_start = tok.decode(151644);
    assert!(im_start.is_empty(), "im_start should decode to empty (not in vocab.json)");
}

#[test]
fn test_encode_empty() {
    let tok = match get_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let tokens = tok.encode("");
    // Empty string may return Some(vec![]) or None depending on implementation
    if let Some(toks) = tokens {
        assert!(toks.is_empty(), "Empty string should produce no tokens");
    }
}

#[test]
fn test_encode_known_phrase() {
    let tok = match get_tokenizer() {
        Some(t) => t,
        None => return,
    };

    // "language en" is used as force_language prompt - must encode successfully
    let tokens = tok.encode("language en");
    assert!(tokens.is_some(), "Should encode 'language en'");
    let tokens = tokens.unwrap();
    assert!(tokens.len() >= 2, "'language en' should produce at least 2 tokens");

    // Verify roundtrip
    let mut decoded = String::new();
    for &t in &tokens {
        decoded.push_str(tok.decode(t));
    }
    assert_eq!(decoded, "language en");
}
