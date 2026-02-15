/// ARM NEON implementations of hot kernels.
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
pub unsafe fn bf16_matvec_fused(y: &mut [f32], x: &[f32], w_bf16: *const u16, bias: Option<&[f32]>, in_dim: usize, out_dim: usize) {
    let mut o = 0usize;

    // Process 2 output rows at a time
    while o + 1 < out_dim {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut s0 = bias.map_or(0.0f32, |b| b[o]);
        let mut s1 = bias.map_or(0.0f32, |b| b[o + 1]);

        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut b0 = vdupq_n_f32(0.0);
        let mut b1 = vdupq_n_f32(0.0);
        let mut b2 = vdupq_n_f32(0.0);
        let mut b3 = vdupq_n_f32(0.0);
        let mut k = 0usize;

        while k + 32 <= in_dim {
            let x0 = vld1q_f32(x.as_ptr().add(k));
            let x1 = vld1q_f32(x.as_ptr().add(k + 4));
            let x2 = vld1q_f32(x.as_ptr().add(k + 8));
            let x3 = vld1q_f32(x.as_ptr().add(k + 12));
            let x4 = vld1q_f32(x.as_ptr().add(k + 16));
            let x5 = vld1q_f32(x.as_ptr().add(k + 20));
            let x6 = vld1q_f32(x.as_ptr().add(k + 24));
            let x7 = vld1q_f32(x.as_ptr().add(k + 28));

            let r0a = vld1q_u16(w0.add(k));
            let r0b = vld1q_u16(w0.add(k + 8));
            let r0c = vld1q_u16(w0.add(k + 16));
            let r0d = vld1q_u16(w0.add(k + 24));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            let r1a = vld1q_u16(w1.add(k));
            let r1b = vld1q_u16(w1.add(k + 8));
            let r1c = vld1q_u16(w1.add(k + 16));
            let r1d = vld1q_u16(w1.add(k + 24));
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);

            k += 32;
        }
        while k + 8 <= in_dim {
            let xv0 = vld1q_f32(x.as_ptr().add(k));
            let xv1 = vld1q_f32(x.as_ptr().add(k + 4));
            let r0 = vld1q_u16(w0.add(k));
            let r1 = vld1q_u16(w1.add(k));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0), 16)), xv0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0), 16)), xv1);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1), 16)), xv0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1), 16)), xv1);
            k += 8;
        }
        s0 += vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        s1 += vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        while k < in_dim {
            let wv0 = f32::from_bits(((*w0.add(k)) as u32) << 16);
            let wv1 = f32::from_bits(((*w1.add(k)) as u32) << 16);
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

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while k + 8 <= in_dim {
            let bf = vld1q_u16(w_row.add(k));
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k)));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k + 4)));
            k += 8;
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits(((*w_row.add(k)) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        y[o] = sum;
        o += 1;
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn argmax_bf16_range(x: &[f32], w_bf16: *const u16, in_dim: usize, start: usize, end: usize) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;
    let mut o = start;

    // Process 2 rows at a time
    while o + 1 < end {
        let w0 = w_bf16.add(o * in_dim);
        let w1 = w_bf16.add((o + 1) * in_dim);
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut a2 = vdupq_n_f32(0.0);
        let mut a3 = vdupq_n_f32(0.0);
        let mut b0 = vdupq_n_f32(0.0);
        let mut b1 = vdupq_n_f32(0.0);
        let mut b2 = vdupq_n_f32(0.0);
        let mut b3 = vdupq_n_f32(0.0);
        let mut k = 0usize;

        while k + 32 <= in_dim {
            let x0 = vld1q_f32(x.as_ptr().add(k));
            let x1 = vld1q_f32(x.as_ptr().add(k + 4));
            let x2 = vld1q_f32(x.as_ptr().add(k + 8));
            let x3 = vld1q_f32(x.as_ptr().add(k + 12));
            let x4 = vld1q_f32(x.as_ptr().add(k + 16));
            let x5 = vld1q_f32(x.as_ptr().add(k + 20));
            let x6 = vld1q_f32(x.as_ptr().add(k + 24));
            let x7 = vld1q_f32(x.as_ptr().add(k + 28));

            let r0a = vld1q_u16(w0.add(k));
            let r0b = vld1q_u16(w0.add(k + 8));
            let r0c = vld1q_u16(w0.add(k + 16));
            let r0d = vld1q_u16(w0.add(k + 24));
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0a), 16)), x0);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0a), 16)), x1);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0b), 16)), x2);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0b), 16)), x3);
            a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0c), 16)), x4);
            a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0c), 16)), x5);
            a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r0d), 16)), x6);
            a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r0d), 16)), x7);

            let r1a = vld1q_u16(w1.add(k));
            let r1b = vld1q_u16(w1.add(k + 8));
            let r1c = vld1q_u16(w1.add(k + 16));
            let r1d = vld1q_u16(w1.add(k + 24));
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1a), 16)), x0);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1a), 16)), x1);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1b), 16)), x2);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1b), 16)), x3);
            b0 = vfmaq_f32(b0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1c), 16)), x4);
            b1 = vfmaq_f32(b1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1c), 16)), x5);
            b2 = vfmaq_f32(b2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r1d), 16)), x6);
            b3 = vfmaq_f32(b3, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r1d), 16)), x7);

            k += 32;
        }

        let s0_v = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3)));
        let s1_v = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b2), vaddq_f32(b1, b3)));

        let mut s0 = s0_v;
        let mut s1 = s1_v;
        while k < in_dim {
            let wv0 = f32::from_bits(((*w0.add(k)) as u32) << 16);
            let wv1 = f32::from_bits(((*w1.add(k)) as u32) << 16);
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

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while k + 8 <= in_dim {
            let bf = vld1q_u16(w_row.add(k));
            acc0 = vfmaq_f32(acc0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k)));
            acc1 = vfmaq_f32(acc1, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)),
                             vld1q_f32(x.as_ptr().add(k + 4)));
            k += 8;
        }
        sum += vaddvq_f32(vaddq_f32(acc0, acc1));

        while k < in_dim {
            let w_val = f32::from_bits(((*w_row.add(k)) as u32) << 16);
            sum += w_val * x[k];
            k += 1;
        }
        if sum > best_val { best_val = sum; best = o; }
        o += 1;
    }

    (best, best_val)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    while i + 8 <= n {
        let a0 = vld1q_f32(a.as_ptr().add(i));
        let b0 = vld1q_f32(b.as_ptr().add(i));
        let a1 = vld1q_f32(a.as_ptr().add(i + 4));
        let b1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        i += 8;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    let mut i = 0usize;
    let s = vdupq_n_f32(scale);
    while i + 8 <= n {
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(d0, s));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(d1, s));
        i += 8;
    }
    while i < n {
        dst[i] *= scale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    let mut i = 0usize;
    let a = vdupq_n_f32(alpha);
    while i + 8 <= n {
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(d0, s0, a));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vfmaq_f32(d1, s1, a));
        i += 8;
    }
    while i < n {
        dst[i] += alpha * src[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    let mut i = 0usize;
    let c = vdupq_n_f32(correction);
    while i + 8 <= n {
        let d0 = vld1q_f32(dst.as_ptr().add(i));
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let d1 = vld1q_f32(dst.as_ptr().add(i + 4));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vfmaq_f32(s0, d0, c));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vfmaq_f32(s1, d1, c));
        i += 8;
    }
    while i < n {
        dst[i] = dst[i] * correction + src[i];
        i += 1;
    }
}
