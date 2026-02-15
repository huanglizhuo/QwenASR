// x86 AVX2+FMA implementations of hot kernels.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
    y: &mut [f32], x: &[f32], w_bf16: *const u16,
    bias: Option<&[f32]>, in_dim: usize, out_dim: usize,
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
    x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize,
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

        if s0 > best_val { best_val = s0; best = o; }
        if s1 > best_val { best_val = s1; best = o + 1; }
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
        if sum > best_val { best_val = sum; best = o; }
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
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), s));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 8)), s));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 16)), s));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 24)), s));
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), s));
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
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i)), a, _mm256_loadu_ps(dst.as_ptr().add(i))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i + 8)), a, _mm256_loadu_ps(dst.as_ptr().add(i + 8))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i + 16)), a, _mm256_loadu_ps(dst.as_ptr().add(i + 16))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i + 24)), a, _mm256_loadu_ps(dst.as_ptr().add(i + 24))));
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i)), a, _mm256_loadu_ps(dst.as_ptr().add(i))));
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
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), c, _mm256_loadu_ps(src.as_ptr().add(i))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 8)), c, _mm256_loadu_ps(src.as_ptr().add(i + 8))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 16)), c, _mm256_loadu_ps(src.as_ptr().add(i + 16))));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i + 24)), c, _mm256_loadu_ps(src.as_ptr().add(i + 24))));
        i += 32;
    }

    while i + 8 <= n {
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(_mm256_loadu_ps(dst.as_ptr().add(i)), c, _mm256_loadu_ps(src.as_ptr().add(i))));
        i += 8;
    }

    while i < n {
        dst[i] = dst[i] * correction + src[i];
        i += 1;
    }
}
