//! R15-8: `decoder_prefill_kv_only` must leave a KV cache bit-identical to the
//! full `decoder_prefill` (it only skips the last layer's Q/attention/O/FFN,
//! whose output is the unused final hidden state).

use qwen_asr::context::QwenCtx;
use qwen_asr::decoder::{
    decoder_prefill, decoder_prefill_kv_only, tok_embed_bf16_to_f32, DecoderBuffers, KvCache,
    RopeCache,
};

mod common;

#[test]
fn kv_only_prefill_matches_full_prefill_cache() {
    let model_dir = match common::model_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping prefill_kv_only test: model not downloaded");
            return;
        }
    };
    let ctx = QwenCtx::load(&model_dir).expect("load ctx");
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;

    // A prompt-like token run (system/user/assistant scaffolding + text ids).
    let tokens: Vec<i32> = vec![
        151644, 8948, 198, 151645, 198, 151644, 872, 198, 9707, 11, 1879, 0, 151645,
    ];
    let seq = tokens.len();
    let mut embeds = vec![0f32; seq * dim];
    for (i, &t) in tokens.iter().enumerate() {
        unsafe {
            tok_embed_bf16_to_f32(
                &mut embeds[i * dim..(i + 1) * dim],
                decoder.tok_embeddings_bf16,
                t,
                dim,
            )
        };
    }

    let run = |kv_only: bool| {
        let mut kv = KvCache::new(cfg.dec_layers, 256, cfg.dec_kv_heads, cfg.dec_head_dim);
        let mut rope = RopeCache::new();
        let mut bufs = DecoderBuffers::new(cfg);
        if kv_only {
            decoder_prefill_kv_only(decoder, cfg, &mut kv, &mut rope, &mut bufs, &embeds, seq);
        } else {
            decoder_prefill(decoder, cfg, &mut kv, &mut rope, &mut bufs, &embeds, seq);
        }
        assert_eq!(kv.len, seq);
        kv
    };
    let full = run(false);
    let kv_only = run(true);

    let hs = full.head_stride();
    let hd = cfg.dec_head_dim;
    for l in 0..cfg.dec_layers {
        for h in 0..cfg.dec_kv_heads {
            let off = (l * cfg.dec_kv_heads + h) * hs;
            for (name, a, b) in [("K", &full.k, &kv_only.k), ("V", &full.v, &kv_only.v)] {
                let (a, b) = (&a[off..off + seq * hd], &b[off..off + seq * hd]);
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{name} cache differs at layer {l} head {h}"
                );
            }
        }
    }
}
