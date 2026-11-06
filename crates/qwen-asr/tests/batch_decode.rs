//! Batched decode must preserve every session's token, hidden state and KV bits.
//! Own process: keep the global kernel pool at one fixed width throughout.
#![cfg(target_arch = "aarch64")]

use qwen_asr::context::QwenCtx;
use qwen_asr::decoder::{
    decoder_forward, decoder_forward_batched, decoder_prefill, tok_embed_bf16_to_f32,
    BatchedSession, DecoderBuffers, KvCache, RopeCache,
};
use qwen_asr::kernels;

mod common;

#[test]
fn batch_decode_matches_sequential_state() {
    let model_dir = match common::model_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping batch decode test: model not downloaded");
            return;
        }
    };
    kernels::set_verbose(0);
    kernels::set_threads(12);
    let ctx = QwenCtx::load(&model_dir).expect("load ctx");
    let cfg = &ctx.config;
    let decoder = &ctx.model.decoder;
    let dim = cfg.dec_hidden;
    let mut reference_bufs = DecoderBuffers::new(cfg);
    const STEPS: usize = 4;

    for b in [1, 2, 5, 8] {
        let mut reference = Vec::with_capacity(b);
        let mut batched = Vec::with_capacity(b);
        for bi in 0..b {
            // Different positions and strides catch accidental cross-session addressing.
            let seq = 1 + bi * 3;
            let capacity = seq + STEPS + 1;
            let mut kv = KvCache::new(cfg.dec_layers, capacity, cfg.dec_kv_heads, cfg.dec_head_dim);
            let mut rope = RopeCache::new();
            let mut prefill = vec![0.0; seq * dim];
            for (i, embedding) in prefill.chunks_exact_mut(dim).enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(
                        embedding,
                        decoder.tok_embeddings_bf16,
                        (10 + bi * 31 + i * 7) as i32,
                        dim,
                    );
                }
            }
            reference_bufs.ensure_prefill(seq, cfg);
            decoder_prefill(
                decoder,
                cfg,
                &mut kv,
                &mut rope,
                &mut reference_bufs,
                &prefill,
                seq,
            );
            assert_eq!(kv.len, seq);

            // Copy the actual prefill so only decode scheduling differs.
            let mut batch_kv =
                KvCache::new(cfg.dec_layers, capacity, cfg.dec_kv_heads, cfg.dec_head_dim);
            batch_kv.k.copy_from_slice(&kv.k);
            batch_kv.v.copy_from_slice(&kv.v);
            batch_kv.len = kv.len;
            reference.push((kv, rope));
            batched.push((batch_kv, RopeCache::new(), DecoderBuffers::new(cfg)));
        }

        let mut embeds = vec![vec![0.0; dim]; b];
        let mut got_tokens = vec![0; b];
        for step in 0..STEPS {
            for (bi, embedding) in embeds.iter_mut().enumerate() {
                unsafe {
                    tok_embed_bf16_to_f32(
                        embedding,
                        decoder.tok_embeddings_bf16,
                        (7 + step * 43 + bi * 11) as i32,
                        dim,
                    );
                }
            }
            {
                let mut sessions: Vec<_> = batched
                    .iter_mut()
                    .zip(&embeds)
                    .map(|((kv_cache, rope, bufs), input_embed)| BatchedSession {
                        kv_cache,
                        rope,
                        bufs,
                        input_embed,
                    })
                    .collect();
                decoder_forward_batched(decoder, cfg, &mut sessions, &mut got_tokens);
            }

            for (bi, ((want_kv, want_rope), (got_kv, _, got_bufs))) in
                reference.iter_mut().zip(&batched).enumerate()
            {
                let want_token = decoder_forward(
                    decoder,
                    cfg,
                    want_kv,
                    want_rope,
                    &mut reference_bufs,
                    &embeds[bi],
                );
                assert_eq!(
                    got_tokens[bi], want_token,
                    "token B={b} step={step} session={bi}"
                );
                assert!(
                    got_bufs.x[..dim]
                        .iter()
                        .zip(&reference_bufs.x[..dim])
                        .all(|(got, want)| got.to_bits() == want.to_bits()),
                    "hidden state B={b} step={step} session={bi}"
                );
                assert_eq!(want_kv.len, 1 + bi * 3 + step + 1);
                assert_eq!(got_kv.len, want_kv.len);
                let active = want_kv.len * cfg.dec_head_dim;
                for layer in 0..cfg.dec_layers {
                    for head in 0..cfg.dec_kv_heads {
                        let index = layer * cfg.dec_kv_heads + head;
                        let want_start = index * want_kv.head_stride();
                        let got_start = index * got_kv.head_stride();
                        for (name, want, got) in
                            [("K", &want_kv.k, &got_kv.k), ("V", &want_kv.v, &got_kv.v)]
                        {
                            assert!(
                                got[got_start..got_start + active]
                                    .iter()
                                    .zip(&want[want_start..want_start + active])
                                    .all(|(got, want)| got.to_bits() == want.to_bits()),
                                "{name} B={b} step={step} session={bi} layer={layer} head={head}"
                            );
                        }
                    }
                }
            }
        }
    }
}
