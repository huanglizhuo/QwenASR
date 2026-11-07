use qwen_asr::decoder::KvCache;

#[test]
fn growing_cache_preserves_every_head_and_bounds_spare_capacity() {
    let mut small = KvCache::new(1, 16, 1, 1);
    for (required, capacity) in [(17, 32), (513, 1024), (1025, 2048)] {
        small.grow(required);
        assert_eq!(small.max_seq, capacity);
    }

    let mut cache = KvCache::new(2, 2048, 3, 4);
    for required in [2048, 2049, 3073, 9613] {
        let old_len = cache.len;
        cache.grow(required);
        assert!(cache.max_seq >= required);
        assert!(
            cache.max_seq - required < required / 2,
            "capacity {} for {required}",
            cache.max_seq
        );
        assert_eq!(cache.len, old_len);

        for layer in 0..cache.n_layers {
            for head in 0..cache.n_kv_heads {
                let base = (layer * cache.n_kv_heads + head) * cache.head_stride();
                for pos in 0..cache.max_seq {
                    for dim in 0..cache.head_dim {
                        let offset = base + pos * cache.head_dim + dim;
                        let value = ((layer * 3 + head) * 10_000 + pos) * 4 + dim;
                        let expected = if pos < old_len { value as f32 } else { 0.0 };
                        assert_eq!(cache.k[offset].to_bits(), expected.to_bits());
                        let expected_v = if pos < old_len { -expected } else { 0.0 };
                        assert_eq!(cache.v[offset].to_bits(), expected_v.to_bits());
                    }
                }
            }
            for pos in old_len..required {
                let k: Vec<f32> = (0..cache.n_kv_heads * cache.head_dim)
                    .map(|i| (((layer * 3 + i / 4) * 10_000 + pos) * 4 + i % 4) as f32)
                    .collect();
                let v: Vec<f32> = k.iter().map(|x| -x).collect();
                cache.k_write_pos(layer, pos, &k);
                cache.v_write_pos(layer, pos, &v);
            }
        }
        cache.len = required;
    }
    let (k, v) = (cache.k.as_ptr(), cache.v.as_ptr());
    cache.grow(cache.len);
    assert_eq!(cache.k.as_ptr(), k);
    assert_eq!(cache.v.as_ptr(), v);
}
