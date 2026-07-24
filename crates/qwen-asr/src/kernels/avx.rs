// x86 AVX2+FMA implementations of hot kernels.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    let n = src.len();
    let mut i = 0usize;

    while i + 8 <= n {
        let raw = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        let wide = _mm256_cvtepu16_epi32(raw);
        let shifted = _mm256_slli_epi32(wide, 16);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_castsi256_ps(shifted));
        i += 8;
    }

    while i < n {
        dst[i] = f32::from_bits((src[i] as u32) << 16);
        i += 1;
    }
}

/// Convert 8 BF16 values (in a __m128i) to 8 f32 values (in a __m256).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bf16x8_to_f32(raw: __m128i) -> __m256 {
    // Zero-extend u16 -> u32, shift left 16 to put BF16 bits in f32 position
    let wide = _mm256_cvtepu16_epi32(raw);
    let shifted = _mm256_slli_epi32(wide, 16);
    _mm256_castsi256_ps(shifted)
}

/// Horizontal sum of __m256 -> f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_ps(v: __m256) -> f32 {
    // Add high 128 to low 128
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    // Horizontal add twice to reduce 4 -> 2 -> 1
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sum64 = _mm_add_ps(sum128, shuf); // [0+1, _, 2+3, _]
    let hi64 = _mm_movehl_ps(sum64, sum64); // [2+3, _, _, _]
    let sum32 = _mm_add_ss(sum64, hi64);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn bf16_matvec_fused(
    y: &mut [f32],
    x: &[f32],
    w_bf16: *const u16,
    bias: Option<&[f32]>,
    in_dim: usize,
    out_dim: usize,
) {
    let mut o = 0usize;

    // Process 2 output rows at a time
    while o + 1 < out_dim {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut s0 = bias.map_or(0.0f32, |b| b[o]);
        let mut s1 = bias.map_or(0.0f32, |b| b[o + 1]);

        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut k = 0usize;

        // Main loop: 16 elements per iteration
        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));

            // Row 0
            let raw0lo = _mm_loadu_si128(w0.add(k) as *const __m128i);
            let raw0hi = _mm_loadu_si128(w0.add(k + 8) as *const __m128i);
            let w0lo = bf16x8_to_f32(raw0lo);
            let w0hi = bf16x8_to_f32(raw0hi);
            a0 = _mm256_fmadd_ps(w0lo, xlo, a0);
            a1 = _mm256_fmadd_ps(w0hi, xhi, a1);

            // Row 1
            let raw1lo = _mm_loadu_si128(w1.add(k) as *const __m128i);
            let raw1hi = _mm_loadu_si128(w1.add(k + 8) as *const __m128i);
            let w1lo = bf16x8_to_f32(raw1lo);
            let w1hi = bf16x8_to_f32(raw1hi);
            b0 = _mm256_fmadd_ps(w1lo, xlo, b0);
            b1 = _mm256_fmadd_ps(w1hi, xhi, b1);

            k += 16;
        }

        // 8-element cleanup
        while k + 8 <= in_dim {
            let xv = _mm256_loadu_ps(x.as_ptr().add(k));
            let r0 = bf16x8_to_f32(_mm_loadu_si128(w0.add(k) as *const __m128i));
            let r1 = bf16x8_to_f32(_mm_loadu_si128(w1.add(k) as *const __m128i));
            a0 = _mm256_fmadd_ps(r0, xv, a0);
            b0 = _mm256_fmadd_ps(r1, xv, b0);
            k += 8;
        }

        s0 += hsum_ps(_mm256_add_ps(a0, a1));
        s1 += hsum_ps(_mm256_add_ps(b0, b1));

        // Scalar tail
        while k < in_dim {
            let wv0 = f32::from_bits((*w0.add(k) as u32) << 16);
            let wv1 = f32::from_bits((*w1.add(k) as u32) << 16);
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
            k += 1;
        }

        y[o] = s0;
        y[o + 1] = s1;
        o += 2;
    }

    // Handle remaining odd row
    while o < out_dim {
        let w_row = w_bf16.add(o * in_dim);
        let mut sum = bias.map_or(0.0f32, |b| b[o]);
        let mut k = 0usize;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));
            let wlo = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            let whi = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k + 8) as *const __m128i));
            acc0 = _mm256_fmadd_ps(wlo, xlo, acc0);
            acc1 = _mm256_fmadd_ps(whi, xhi, acc1);
            k += 16;
        }

        while k + 8 <= in_dim {
            let xv = _mm256_loadu_ps(x.as_ptr().add(k));
            let wv = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            acc0 = _mm256_fmadd_ps(wv, xv, acc0);
            k += 8;
        }

        sum += hsum_ps(_mm256_add_ps(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits((*w_row.add(k) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        y[o] = sum;
        o += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn argmax_bf16_range(
    x: &[f32],
    w_bf16: *const u16,
    in_dim: usize,
    start: usize,
    end: usize,
) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;
    let mut o = start;

    // Process 2 rows at a time
    while o + 1 < end {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut k = 0usize;

        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));

            let r0lo = bf16x8_to_f32(_mm_loadu_si128(w0.add(k) as *const __m128i));
            let r0hi = bf16x8_to_f32(_mm_loadu_si128(w0.add(k + 8) as *const __m128i));
            a0 = _mm256_fmadd_ps(r0lo, xlo, a0);
            a1 = _mm256_fmadd_ps(r0hi, xhi, a1);

            let r1lo = bf16x8_to_f32(_mm_loadu_si128(w1.add(k) as *const __m128i));
            let r1hi = bf16x8_to_f32(_mm_loadu_si128(w1.add(k + 8) as *const __m128i));
            b0 = _mm256_fmadd_ps(r1lo, xlo, b0);
            b1 = _mm256_fmadd_ps(r1hi, xhi, b1);

            k += 16;
        }

        let mut s0 = hsum_ps(_mm256_add_ps(a0, a1));
        let mut s1 = hsum_ps(_mm256_add_ps(b0, b1));

        while k < in_dim {
            let wv0 = f32::from_bits((*w0.add(k) as u32) << 16);
            let wv1 = f32::from_bits((*w1.add(k) as u32) << 16);
            s0 += wv0 * x[k];
            s1 += wv1 * x[k];
            k += 1;
        }

        if s0 > best_val {
            best_val = s0;
            best = o;
        }
        if s1 > best_val {
            best_val = s1;
            best = o + 1;
        }
        o += 2;
    }

    while o < end {
        let w_row = w_bf16.add(o * in_dim);
        let mut sum = 0.0f32;
        let mut k = 0usize;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        while k + 16 <= in_dim {
            let xlo = _mm256_loadu_ps(x.as_ptr().add(k));
            let xhi = _mm256_loadu_ps(x.as_ptr().add(k + 8));
            let wlo = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            let whi = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k + 8) as *const __m128i));
            acc0 = _mm256_fmadd_ps(wlo, xlo, acc0);
            acc1 = _mm256_fmadd_ps(whi, xhi, acc1);
            k += 16;
        }
        sum += hsum_ps(_mm256_add_ps(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits((*w_row.add(k) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        if sum > best_val {
            best_val = sum;
            best = o;
        }
        o += 1;
    }

    (best, best_val)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 16)),
            _mm256_loadu_ps(b.as_ptr().add(i + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 24)),
            _mm256_loadu_ps(b.as_ptr().add(i + 24)),
            acc3,
        );
        i += 32;
    }

    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc0,
        );
        i += 8;
    }

    let mut sum = hsum_ps(_mm256_add_ps(
        _mm256_add_ps(acc0, acc1),
        _mm256_add_ps(acc2, acc3),
    ));

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    let mut i = 0usize;
    let s = _mm256_set1_ps(scale);

    while i + 32 <= n {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), s),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 8),
            _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 8)), s),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 16),
            _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 16)), s),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 24),
            _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 24)), s),
        );
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), s),
        );
        i += 8;
    }

    while i < n {
        dst[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    let mut i = 0usize;
    let a = _mm256_set1_ps(alpha);

    while i + 32 <= n {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(src.as_ptr().add(i)),
                a,
                _mm256_loadu_ps(dst.as_ptr().add(i)),
            ),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 8),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(src.as_ptr().add(i + 8)),
                a,
                _mm256_loadu_ps(dst.as_ptr().add(i + 8)),
            ),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 16),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(src.as_ptr().add(i + 16)),
                a,
                _mm256_loadu_ps(dst.as_ptr().add(i + 16)),
            ),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 24),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(src.as_ptr().add(i + 24)),
                a,
                _mm256_loadu_ps(dst.as_ptr().add(i + 24)),
            ),
        );
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(src.as_ptr().add(i)),
                a,
                _mm256_loadu_ps(dst.as_ptr().add(i)),
            ),
        );
        i += 8;
    }

    while i < n {
        dst[i] += alpha * src[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    let mut i = 0usize;
    let c = _mm256_set1_ps(correction);

    while i + 32 <= n {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(dst.as_ptr().add(i)),
                c,
                _mm256_loadu_ps(src.as_ptr().add(i)),
            ),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 8),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(dst.as_ptr().add(i + 8)),
                c,
                _mm256_loadu_ps(src.as_ptr().add(i + 8)),
            ),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 16),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(dst.as_ptr().add(i + 16)),
                c,
                _mm256_loadu_ps(src.as_ptr().add(i + 16)),
            ),
        );
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i + 24),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(dst.as_ptr().add(i + 24)),
                c,
                _mm256_loadu_ps(src.as_ptr().add(i + 24)),
            ),
        );
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(
            dst.as_mut_ptr().add(i),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(dst.as_ptr().add(i)),
                c,
                _mm256_loadu_ps(src.as_ptr().add(i)),
            ),
        );
        i += 8;
    }

    while i < n {
        dst[i] = dst[i] * correction + src[i];
        i += 1;
    }
}

/// AVX2-accelerated RMS norm for a single row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], hidden: usize, eps: f32) {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= hidden {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(x0, x0, acc0);
        acc1 = _mm256_fmadd_ps(x1, x1, acc1);
        i += 16;
    }
    while i + 8 <= hidden {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(xv, xv, acc0);
        i += 8;
    }

    let mut sum_sq = hsum_ps(_mm256_add_ps(acc0, acc1));
    while i < hidden {
        sum_sq += x[i] * x[i];
        i += 1;
    }

    let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    let rms_v = _mm256_set1_ps(rms_inv);

    i = 0;
    while i + 8 <= hidden {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(xv, rms_v), wv),
        );
        i += 8;
    }
    while i < hidden {
        out[i] = x[i] * rms_inv * weight[i];
        i += 1;
    }
}

/// AVX2-accelerated layer norm for a single row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn layer_norm_row(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    hidden: usize,
    eps: f32,
) {
    // Pass 1: compute mean
    let mut i = 0usize;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    while i + 16 <= hidden {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(x.as_ptr().add(i)));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(x.as_ptr().add(i + 8)));
        i += 16;
    }
    while i + 8 <= hidden {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(x.as_ptr().add(i)));
        i += 8;
    }
    let mut mean = hsum_ps(_mm256_add_ps(sum0, sum1));
    while i < hidden {
        mean += x[i];
        i += 1;
    }
    mean /= hidden as f32;

    // Pass 2: compute variance
    let mean_v = _mm256_set1_ps(mean);
    i = 0;
    let mut var0 = _mm256_setzero_ps();
    let mut var1 = _mm256_setzero_ps();
    while i + 16 <= hidden {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean_v);
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i + 8)), mean_v);
        var0 = _mm256_fmadd_ps(d0, d0, var0);
        var1 = _mm256_fmadd_ps(d1, d1, var1);
        i += 16;
    }
    while i + 8 <= hidden {
        let d = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean_v);
        var0 = _mm256_fmadd_ps(d, d, var0);
        i += 8;
    }
    let mut var = hsum_ps(_mm256_add_ps(var0, var1));
    while i < hidden {
        let d = x[i] - mean;
        var += d * d;
        i += 1;
    }

    let inv_std = 1.0 / (var / hidden as f32 + eps).sqrt();
    let inv_v = _mm256_set1_ps(inv_std);

    // Pass 3: normalize
    i = 0;
    while i + 8 <= hidden {
        let xv = _mm256_sub_ps(_mm256_loadu_ps(x.as_ptr().add(i)), mean_v);
        let wv = _mm256_loadu_ps(weight.as_ptr().add(i));
        let bv = _mm256_loadu_ps(bias.as_ptr().add(i));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_fmadd_ps(_mm256_mul_ps(xv, inv_v), wv, bv),
        );
        i += 8;
    }
    while i < hidden {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        i += 1;
    }
}

/// Fast exp approximation using AVX2+FMA (~1e-4 relative error for |x| < 88).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fast_exp_avx(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(1.442695041);
    let ln2 = _mm256_set1_ps(0.6931471806);

    let val = _mm256_mul_ps(x, log2e);
    let val = _mm256_min_ps(val, _mm256_set1_ps(126.0));
    let val = _mm256_max_ps(val, _mm256_set1_ps(-126.0));

    let ipart = _mm256_cvtps_epi32(val);
    let fpart = _mm256_sub_ps(val, _mm256_cvtepi32_ps(ipart));

    let exp_i = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ipart, _mm256_set1_epi32(127)),
        23,
    ));

    let f = _mm256_mul_ps(fpart, ln2);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);

    let mut p = _mm256_fmadd_ps(c5, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0));

    _mm256_mul_ps(exp_i, p)
}

/// AVX2-accelerated exp() in-place using fast polynomial approximation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn exp_inplace(x: &mut [f32]) {
    let n = x.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), fast_exp_avx(v));
        i += 8;
    }
    while i < n {
        x[i] = x[i].exp();
        i += 1;
    }
}

/// AVX2-accelerated GELU (tanh approximation).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gelu_inplace(x: &mut [f32], n: usize) {
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let coeff = _mm256_set1_ps(0.7978845608028654);
    let c3 = _mm256_set1_ps(0.044715);
    let mut i = 0usize;

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let v2 = _mm256_mul_ps(v, v);
        let v3 = _mm256_mul_ps(v2, v);
        let inner = _mm256_mul_ps(coeff, _mm256_fmadd_ps(c3, v3, v));
        let exp2x = fast_exp_avx(_mm256_mul_ps(two, inner));
        let tanh_v = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(exp2x, one)));
        let result = _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, tanh_v)));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < n {
        let val = x[i];
        let x3 = val * val * val;
        let inner = 0.7978845608028654f32 * (val + 0.044715 * x3);
        x[i] = 0.5 * val * (1.0 + inner.tanh());
        i += 1;
    }
}

// ========================================================================
// INT8 decode kernels (AVX2)
//
// x86_64 mirrors of the NEON SDOT kernels in neon.rs. AVX2 has no i8 dot
// instruction (VNNI is intentionally not required), so each 32-byte block is
// sign-extended to i16 and multiplied pairwise with `_mm256_madd_epi16`,
// whose i32 output is EXACT (no saturation: |2*127*127| = 32258 < 2^31), the
// same exactness SDOT gives the NEON path. Integer accumulation order is
// irrelevant, and every float epilogue below is byte-for-byte the same
// expression as its NEON counterpart, so per-row results match the aarch64
// kernels bit-for-bit whenever the NEON path is internally consistent
// (tail-free `in_dim`, or the matching 2-row/1-row variant).
// ========================================================================

/// Horizontal sum of 8×i32 -> i32 (exact; grouping is irrelevant).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
    _mm_cvtsi128_si32(s)
}

/// Horizontal max of 8×f32 -> f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_ps(v: __m256) -> f32 {
    let s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 0b00_00_00_01));
    _mm_cvtss_f32(s)
}

/// Accumulate one 32-byte block of `x·w` (i8×i8) into the i32 lanes of `acc`.
/// Exact AVX2 analogue of the NEON `sdot_s32` helper: sign-extend both
/// operands to i16 and use `_mm256_madd_epi16` (exact i32 pairwise sums).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sdot32(acc: __m256i, x: *const i8, w: *const i8) -> __m256i {
    let xv = _mm256_loadu_si256(x as *const __m256i);
    let wv = _mm256_loadu_si256(w as *const __m256i);
    let plo = _mm256_madd_epi16(
        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv)),
        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv)),
    );
    let phi = _mm256_madd_epi16(
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1)),
        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv, 1)),
    );
    _mm256_add_epi32(acc, _mm256_add_epi32(plo, phi))
}

/// Quantize BF16 weight matrix to INT8 per-row with absmax scaling.
/// Returns (int8_data, scales) where `scales[row]` is the per-row scale factor.
/// Bit-identical to the NEON `quantize_bf16_to_int8` (including tails):
/// `_mm256_cvttps_epi32` truncates toward zero exactly like `vcvtq_s32_f32`
/// (fcvtzs), the saturating packs match `vqmovn`, and the scalar tail uses the
/// same `.round()` expression as the NEON scalar tail.
///
/// # Safety
/// w_bf16 must point to at least out_dim * in_dim valid bf16 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_bf16_to_int8(
    w_bf16: *const u16,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut int8_data = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

    for row in 0..out_dim {
        let w_row = w_bf16.add(row * in_dim);

        // Find absmax of the row
        let mut k = 0;
        let mut vmax = _mm256_setzero_ps();
        while k + 8 <= in_dim {
            let f = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(f, abs_mask));
            k += 8;
        }
        let mut max_abs = hmax_ps(vmax);
        while k < in_dim {
            let v = f32::from_bits((*w_row.add(k) as u32) << 16).abs();
            if v > max_abs {
                max_abs = v;
            }
            k += 1;
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 127.0 / max_abs.max(1e-10);
        scales[row] = scale;

        // Quantize row
        let dst = int8_data.as_mut_ptr().add(row * in_dim);
        k = 0;
        let inv_s = _mm256_set1_ps(inv_scale);
        while k + 16 <= in_dim {
            let f0 = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            let f1 = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k + 8) as *const __m128i));
            let q0 = _mm256_cvttps_epi32(_mm256_mul_ps(f0, inv_s));
            let q1 = _mm256_cvttps_epi32(_mm256_mul_ps(f1, inv_s));
            // Saturating i32 -> i16 -> i8 narrowing (matches vqmovn). packs
            // works per 128-bit lane, so restore element order between steps.
            let p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0b11_01_10_00);
            let p8 = _mm_packs_epi16(
                _mm256_castsi256_si128(p16),
                _mm256_extracti128_si256(p16, 1),
            );
            _mm_storeu_si128(dst.add(k) as *mut __m128i, p8);
            k += 16;
        }
        while k + 8 <= in_dim {
            let f = bf16x8_to_f32(_mm_loadu_si128(w_row.add(k) as *const __m128i));
            let q = _mm256_cvttps_epi32(_mm256_mul_ps(f, inv_s));
            let p16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
            let p8 = _mm_packs_epi16(p16, p16);
            _mm_storel_epi64(dst.add(k) as *mut __m128i, p8);
            k += 8;
        }
        while k < in_dim {
            let v = f32::from_bits((*w_row.add(k) as u32) << 16);
            *dst.add(k) = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            k += 1;
        }
    }

    (int8_data, scales)
}

/// INT8 matvec: `y = W_int8 @ x_int8 * (x_scale * w_scales[row])`
/// Produces f32 output. Optionally adds bias (for fused residual add).
/// Structure and float epilogues mirror the NEON `matvec_int8` exactly.
///
/// # Safety
/// Uses AVX2 intrinsics; pointers must be valid for the stated ranges.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)] // hot kernel entry point; params mirror the C-style ABI
pub unsafe fn matvec_int8(
    y: &mut [f32],
    x_int8: *const i8,
    x_scale: f32,
    w_int8: *const i8,
    w_scales: &[f32],
    bias: Option<&[f32]>,
    in_dim: usize,
    out_dim: usize,
) {
    let mut o = 0;
    while o + 1 < out_dim {
        let w0 = w_int8.add(o * in_dim);
        let w1 = w_int8.add((o + 1) * in_dim);
        let mut acc0a = _mm256_setzero_si256();
        let mut acc0b = _mm256_setzero_si256();
        let mut acc1a = _mm256_setzero_si256();
        let mut acc1b = _mm256_setzero_si256();
        let mut k = 0;

        while k + 64 <= in_dim {
            acc0a = sdot32(acc0a, x_int8.add(k), w0.add(k));
            acc0b = sdot32(acc0b, x_int8.add(k + 32), w0.add(k + 32));
            acc1a = sdot32(acc1a, x_int8.add(k), w1.add(k));
            acc1b = sdot32(acc1b, x_int8.add(k + 32), w1.add(k + 32));
            k += 64;
        }
        while k + 32 <= in_dim {
            acc0a = sdot32(acc0a, x_int8.add(k), w0.add(k));
            acc1a = sdot32(acc1a, x_int8.add(k), w1.add(k));
            k += 32;
        }

        let sum0 = hsum_epi32(_mm256_add_epi32(acc0a, acc0b));
        let sum1 = hsum_epi32(_mm256_add_epi32(acc1a, acc1b));

        let mut v0 = sum0 as f32 * x_scale * w_scales[o];
        let mut v1 = sum1 as f32 * x_scale * w_scales[o + 1];

        // Scalar tail
        while k < in_dim {
            let xv = *x_int8.add(k) as i32;
            v0 += xv as f32 * (*w0.add(k) as i32) as f32 * x_scale * w_scales[o];
            v1 += xv as f32 * (*w1.add(k) as i32) as f32 * x_scale * w_scales[o + 1];
            k += 1;
        }

        if let Some(b) = bias {
            v0 += b[o];
            v1 += b[o + 1];
        }
        y[o] = v0;
        y[o + 1] = v1;
        o += 2;
    }
    while o < out_dim {
        let w_row = w_int8.add(o * in_dim);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut k = 0;
        while k + 64 <= in_dim {
            acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
            acc1 = sdot32(acc1, x_int8.add(k + 32), w_row.add(k + 32));
            k += 64;
        }
        while k + 32 <= in_dim {
            acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
            k += 32;
        }
        let mut val = hsum_epi32(_mm256_add_epi32(acc0, acc1)) as f32 * x_scale * w_scales[o];
        while k < in_dim {
            val += (*x_int8.add(k) as f32) * (*w_row.add(k) as f32) * x_scale * w_scales[o];
            k += 1;
        }
        if let Some(b) = bias {
            val += b[o];
        }
        y[o] = val;
        o += 1;
    }
}

/// INT8 argmax: find argmax of x @ W.T where W is int8-quantized.
/// x_int8: quantized input `[in_dim]`, x_scale: input quantization scale
/// W_int8: quantized weights `[out_dim * in_dim]`, w_scales: per-row scales `[out_dim]`
/// Structure and float epilogues mirror the NEON `argmax_int8_range` exactly.
///
/// # Safety
/// Uses AVX2 intrinsics. in_dim should be a multiple of 32 for best perf.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn argmax_int8_range(
    x_int8: *const i8,
    x_scale: f32,
    w_int8: *const i8,
    w_scales: &[f32],
    in_dim: usize,
    start: usize,
    end: usize,
) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;
    let mut o = start;

    while o + 1 < end {
        let w0 = w_int8.add(o * in_dim);
        let w1 = w_int8.add((o + 1) * in_dim);
        let mut acc0a = _mm256_setzero_si256();
        let mut acc0b = _mm256_setzero_si256();
        let mut acc0c = _mm256_setzero_si256();
        let mut acc0d = _mm256_setzero_si256();
        let mut acc1a = _mm256_setzero_si256();
        let mut acc1b = _mm256_setzero_si256();
        let mut acc1c = _mm256_setzero_si256();
        let mut acc1d = _mm256_setzero_si256();
        let mut k = 0;

        while k + 128 <= in_dim {
            acc0a = sdot32(acc0a, x_int8.add(k), w0.add(k));
            acc0b = sdot32(acc0b, x_int8.add(k + 32), w0.add(k + 32));
            acc0c = sdot32(acc0c, x_int8.add(k + 64), w0.add(k + 64));
            acc0d = sdot32(acc0d, x_int8.add(k + 96), w0.add(k + 96));
            acc1a = sdot32(acc1a, x_int8.add(k), w1.add(k));
            acc1b = sdot32(acc1b, x_int8.add(k + 32), w1.add(k + 32));
            acc1c = sdot32(acc1c, x_int8.add(k + 64), w1.add(k + 64));
            acc1d = sdot32(acc1d, x_int8.add(k + 96), w1.add(k + 96));
            k += 128;
        }

        while k + 32 <= in_dim {
            acc0a = sdot32(acc0a, x_int8.add(k), w0.add(k));
            acc1a = sdot32(acc1a, x_int8.add(k), w1.add(k));
            k += 32;
        }

        let sum0_i32 = hsum_epi32(_mm256_add_epi32(
            _mm256_add_epi32(acc0a, acc0c),
            _mm256_add_epi32(acc0b, acc0d),
        ));
        let sum1_i32 = hsum_epi32(_mm256_add_epi32(
            _mm256_add_epi32(acc1a, acc1c),
            _mm256_add_epi32(acc1b, acc1d),
        ));

        let mut tail0 = 0i32;
        let mut tail1 = 0i32;
        while k < in_dim {
            let xv = *x_int8.add(k) as i32;
            tail0 += xv * (*w0.add(k) as i32);
            tail1 += xv * (*w1.add(k) as i32);
            k += 1;
        }

        let val0 = (sum0_i32 + tail0) as f32 * x_scale * w_scales[o];
        let val1 = (sum1_i32 + tail1) as f32 * x_scale * w_scales[o + 1];

        if val0 > best_val {
            best_val = val0;
            best = o;
        }
        if val1 > best_val {
            best_val = val1;
            best = o + 1;
        }
        o += 2;
    }

    while o < end {
        let w_row = w_int8.add(o * in_dim);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut k = 0;

        while k + 128 <= in_dim {
            acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
            acc1 = sdot32(acc1, x_int8.add(k + 32), w_row.add(k + 32));
            acc2 = sdot32(acc2, x_int8.add(k + 64), w_row.add(k + 64));
            acc3 = sdot32(acc3, x_int8.add(k + 96), w_row.add(k + 96));
            k += 128;
        }

        while k + 32 <= in_dim {
            acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
            k += 32;
        }

        let sum_i32 = hsum_epi32(_mm256_add_epi32(
            _mm256_add_epi32(acc0, acc2),
            _mm256_add_epi32(acc1, acc3),
        ));
        let val = sum_i32 as f32 * x_scale * w_scales[o];

        // Scalar tail
        let mut tail_sum = 0i32;
        while k < in_dim {
            tail_sum += (*x_int8.add(k) as i32) * (*w_row.add(k) as i32);
            k += 1;
        }
        let val = val + tail_sum as f32 * x_scale * w_scales[o];

        if val > best_val {
            best_val = val;
            best = o;
        }
        o += 1;
    }

    (best, best_val)
}

// ========================================================================
// Batched (lockstep) INT8 decode kernels
//
// Same contract as the NEON batched kernels: rows outer, sessions inner, and
// each session's per-row output byte-identical to the single-session AVX2
// kernels above (exact integer dots + identical float combine expressions).
// ========================================================================

/// One output row's f32 value for `matvec_int8`-style kernels. Byte-identical
/// to the per-row math of [`matvec_int8`] (both its 2-row and 1-row-tail
/// paths); mirrors the NEON `int8_row_dot_f32`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn int8_row_dot_f32(
    w_row: *const i8,
    x_int8: *const i8,
    x_scale: f32,
    w_scale: f32,
    in_dim: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut k = 0;
    while k + 64 <= in_dim {
        acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
        acc1 = sdot32(acc1, x_int8.add(k + 32), w_row.add(k + 32));
        k += 64;
    }
    while k + 32 <= in_dim {
        acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
        k += 32;
    }
    let mut val = hsum_epi32(_mm256_add_epi32(acc0, acc1)) as f32 * x_scale * w_scale;
    while k < in_dim {
        val += (*x_int8.add(k) as f32) * (*w_row.add(k) as f32) * x_scale * w_scale;
        k += 1;
    }
    val
}

/// One output row's f32 value for `argmax_int8_range`-style scoring: integer
/// dot with an integer-accumulated tail combined *before* the float conversion
/// (`(sum + tail) as f32 * x_scale * w_scale`). Byte-identical to
/// [`argmax_int8_range`] for tail-free `in_dim` (the lm_head case, in_dim =
/// dec_hidden = 1024) and to its 2-row path in general. Mirrors the NEON
/// `int8_row_dot_argmax`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn int8_row_dot_argmax(
    w_row: *const i8,
    x_int8: *const i8,
    x_scale: f32,
    w_scale: f32,
    in_dim: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut k = 0;
    while k + 64 <= in_dim {
        acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
        acc1 = sdot32(acc1, x_int8.add(k + 32), w_row.add(k + 32));
        k += 64;
    }
    while k + 32 <= in_dim {
        acc0 = sdot32(acc0, x_int8.add(k), w_row.add(k));
        k += 32;
    }
    let sum = hsum_epi32(_mm256_add_epi32(acc0, acc1));
    let mut tail = 0i32;
    while k < in_dim {
        tail += (*x_int8.add(k) as i32) * (*w_row.add(k) as i32);
        k += 1;
    }
    (sum + tail) as f32 * x_scale * w_scale
}

/// Batched INT8 matvec: for each output row, stream the weight row once and
/// apply it to all `b` sessions (each with its own quantized input + scale +
/// output + optional residual bias). Row-`o`, session-`bi` output equals
/// [`matvec_int8`]'s row-`o` output for that session, exactly. Mirrors the
/// NEON `matvec_int8_batched`.
///
/// # Safety
/// All pointers must be valid for the stated ranges; slices have length `b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matvec_int8_batched(
    b: usize,
    y: &[*mut f32],
    x_int8: &[*const i8],
    x_scale: &[f32],
    w_int8: *const i8,
    w_scales: &[f32],
    bias: Option<&[*const f32]>,
    in_dim: usize,
    out_dim: usize,
) {
    for (o, &ws) in w_scales.iter().enumerate().take(out_dim) {
        let w_row = w_int8.add(o * in_dim);
        for bi in 0..b {
            let mut val = int8_row_dot_f32(w_row, x_int8[bi], x_scale[bi], ws, in_dim);
            if let Some(bs) = bias {
                val += *bs[bi].add(o);
            }
            *y[bi].add(o) = val;
        }
    }
}

/// Batched fused gate_up + SwiGLU. For each intermediate row `j`, stream the
/// gate row `2j` and up row `2j+1` once and apply to all `b` sessions.
/// Byte-identical to the single-session fused SwiGLU per (row, session).
/// Mirrors the NEON `swiglu_int8_batched`.
///
/// # Safety
/// All pointers must be valid for the stated ranges; slices have length `b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn swiglu_int8_batched(
    b: usize,
    ffn: &[*mut f32],
    x_int8: &[*const i8],
    x_scale: &[f32],
    w_int8: *const i8,
    w_scales: &[f32],
    in_dim: usize,
    n_rows: usize,
) {
    for j in 0..n_rows {
        let wg = w_int8.add(2 * j * in_dim);
        let wu = w_int8.add((2 * j + 1) * in_dim);
        let sg = w_scales[2 * j];
        let su = w_scales[2 * j + 1];
        for bi in 0..b {
            let g = int8_row_dot_f32(wg, x_int8[bi], x_scale[bi], sg, in_dim);
            let u = int8_row_dot_f32(wu, x_int8[bi], x_scale[bi], su, in_dim);
            *ffn[bi].add(j) = g / (1.0 + (-g).exp()) * u;
        }
    }
}

/// Batched INT8 argmax (lm_head): stream each weight row of `[start, end)`
/// once and update every session's running `(best, best_val)` with
/// index-stable tie-breaking (strict `>`, so the lowest row index wins ties —
/// identical to [`argmax_int8_range`]). `best`/`best_val` are per-session
/// running state (init to `0` / `-1e30`); call across disjoint row ranges then
/// reduce. Mirrors the NEON `argmax_int8_batched`.
///
/// # Safety
/// All pointers must be valid for the stated ranges; slices have length `b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn argmax_int8_batched(
    b: usize,
    best: &mut [usize],
    best_val: &mut [f32],
    x_int8: &[*const i8],
    x_scale: &[f32],
    w_int8: *const i8,
    w_scales: &[f32],
    in_dim: usize,
    start: usize,
    end: usize,
) {
    for (o, &ws) in w_scales.iter().enumerate().take(end).skip(start) {
        let w_row = w_int8.add(o * in_dim);
        for bi in 0..b {
            let val = int8_row_dot_argmax(w_row, x_int8[bi], x_scale[bi], ws, in_dim);
            if val > best_val[bi] {
                best_val[bi] = val;
                best[bi] = o;
            }
        }
    }
}
