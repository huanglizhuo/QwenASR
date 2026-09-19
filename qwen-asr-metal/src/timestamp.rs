// Scalar output postprocessing mirrored from crates/qwen-asr/src/align.rs
// (baseline a932504). Preserve tie-breaking and interpolation exactly.
fn longest_increasing_subsequence(vals: &[f32]) -> Vec<usize> {
    let n = vals.len();
    if n == 0 {
        return Vec::new();
    }

    // dp[i] = length of LIS ending at i
    let mut dp = vec![1usize; n];
    let mut prev = vec![usize::MAX; n];

    for i in 1..n {
        for j in 0..i {
            if vals[j] <= vals[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                prev[i] = j;
            }
        }
    }

    // Find the end of the longest sequence
    let mut best_len = 0;
    let mut best_end = 0;
    for (i, &dp_val) in dp.iter().enumerate().take(n) {
        if dp_val > best_len {
            best_len = dp_val;
            best_end = i;
        }
    }

    // Trace back
    let mut lis_indices = Vec::with_capacity(best_len);
    let mut idx = best_end;
    loop {
        lis_indices.push(idx);
        if prev[idx] == usize::MAX {
            break;
        }
        idx = prev[idx];
    }
    lis_indices.reverse();
    lis_indices
}

pub fn fix_timestamps(timestamps: &mut [f32]) {
    if timestamps.len() <= 1 {
        return;
    }

    let lis_indices = longest_increasing_subsequence(timestamps);
    if lis_indices.len() == timestamps.len() {
        return; // Already monotonically increasing
    }

    // Mark which indices are in the LIS (normal)
    let n = timestamps.len();
    let mut is_normal = vec![false; n];
    for &idx in &lis_indices {
        is_normal[idx] = true;
    }

    // Fix anomalous regions
    let mut i = 0;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }

        // Find the extent of this anomalous block
        let block_start = i;
        while i < n && !is_normal[i] {
            i += 1;
        }
        let block_end = i; // exclusive
        let block_len = block_end - block_start;

        // Get boundary values
        let left_val = if block_start > 0 {
            timestamps[block_start - 1]
        } else {
            0.0
        };
        let right_val = if block_end < n {
            timestamps[block_end]
        } else {
            left_val + block_len as f32 * 80.0
        };

        if block_len <= 2 {
            // Small block: fill with nearest normal value
            let fill = if block_start > 0 { left_val } else { right_val };
            for ts in timestamps.iter_mut().take(block_end).skip(block_start) {
                *ts = fill;
            }
        } else {
            // Larger block: linearly interpolate
            for j in 0..block_len {
                let t = (j + 1) as f32 / (block_len + 1) as f32;
                timestamps[block_start + j] = left_val + t * (right_val - left_val);
            }
        }
    }
}
