//! Prompt-lookup draft source for offline speculative decode (R14-B1).
//!
//! Single-token decode is DRAM-bandwidth-bound: every generated token streams
//! the full ~575 MB of INT8 decoder weights. Verifying `k` drafted tokens
//! through [`crate::decoder::decoder_forward_verify`] streams those weights
//! once, so every accepted draft is a nearly-free token — but only if a draft
//! source exists that costs ~nothing to consult. Prompt-lookup is that source:
//! it drafts the tokens that followed the most recent earlier occurrence of
//! the current suffix (last committed tokens plus the pending token) *within
//! the same segment* (self-match).
//! No model, no early exit, no extra forward pass — just a suffix scan over at
//! most 2048 committed tokens.
//!
//! Drafts are only predictions: the verifier accepts a draft iff it equals the
//! sequential greedy argmax, so a bad guess costs nothing in correctness
//! (byte-identical output by construction) and only the verify window's unused
//! lanes in time. Acceptance economics are measured via the verbose decode
//! stats (per-draft acceptance `p`, tokens/verify-step).

/// N-gram self-match draft table over one segment's committed-token history.
/// Reused across the whole segment (the scratch `drafts` vec never
/// reallocates in steady state).
pub struct PromptLookup {
    /// Committed tokens of this segment, in order.
    history: Vec<i32>,
    /// Scratch for the current proposal.
    drafts: Vec<i32>,
}

impl PromptLookup {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            drafts: Vec::new(),
        }
    }

    /// Record a committed token (the pending token is pushed only once it is
    /// actually committed by the decode loop).
    pub fn push(&mut self, tok: i32) {
        self.history.push(tok);
    }

    /// Propose up to `k_max` tokens predicted to follow the PENDING token
    /// (the model's latest output, not yet committed). The verify window is
    /// `[t0, d1..d_k]` where lane 0 consumes `t0` and `verify_accepted_len`
    /// checks `d1` against the argmax AFTER `t0`, so drafts must predict what
    /// comes after `pending` — the match pattern therefore ends at `pending`
    /// itself (standard prompt-lookup alignment).
    ///
    /// Tries the longest suffix first (n = 4, then 3, then 2): the pattern is
    /// the last `n-1` committed tokens plus `pending`; for each n, scans
    /// backwards for the most recent earlier occurrence and drafts whatever
    /// followed it. The follow-through slice may overlap the current tail
    /// (fine — drafts are predictions and the verifier decides). Returns a
    /// slice (possibly empty) into internal scratch, invalidated by the next
    /// call.
    pub fn propose(&mut self, pending: i32, k_max: usize) -> &[i32] {
        self.drafts.clear();
        let h = &self.history;
        for n in [4usize, 3, 2] {
            // Pattern: h[pat_start..] (n-1 committed tokens) + pending.
            // An earlier occurrence at j must lie fully inside h: j + n <= len.
            if h.len() < n {
                continue;
            }
            let pat_start = h.len() + 1 - n;
            let pat_committed = &h[pat_start..];
            let mut j = pat_start;
            while j > 0 {
                j -= 1;
                if h[j..j + n - 1] == *pat_committed && h[j + n - 1] == pending {
                    let start = j + n;
                    let end = (start + k_max).min(h.len());
                    self.drafts.extend_from_slice(&h[start..end]);
                    return &self.drafts;
                }
            }
        }
        &self.drafts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(lookup: &mut PromptLookup, toks: &[i32]) {
        for &t in toks {
            lookup.push(t);
        }
    }

    #[test]
    fn empty_history_proposes_nothing() {
        let mut l = PromptLookup::new();
        assert!(l.propose(7, 7).is_empty());
    }

    #[test]
    fn short_history_proposes_nothing() {
        // [10, 20] + pending 7: no earlier occurrence of the pattern.
        let mut l = PromptLookup::new();
        feed(&mut l, &[10, 20]);
        assert!(l.propose(7, 7).is_empty());
    }

    #[test]
    fn n2_match_drafts_follow_through() {
        // Pattern [3, 4] (last committed 3 + pending 4) occurred at index 0;
        // what followed was [5, 6, ...].
        let mut l = PromptLookup::new();
        feed(&mut l, &[3, 4, 5, 6, 7, 3]);
        assert_eq!(l.propose(4, 7), &[5, 6, 7, 3]);
    }

    #[test]
    fn longest_suffix_wins() {
        // n=4 pattern [1, 2, 3, 4] matches at index 0 (followed by 9); the n=2
        // pattern [3, 4] would also match there, but the n=4 match must be
        // preferred.
        let mut l = PromptLookup::new();
        feed(&mut l, &[1, 2, 3, 4, 9, 1, 2, 3]);
        assert_eq!(l.propose(4, 7), &[9, 1, 2, 3]);
    }

    #[test]
    fn falls_back_to_shorter_n() {
        // No 4- or 3-gram repeat, but the bigram [8, 9] repeats.
        let mut l = PromptLookup::new();
        feed(&mut l, &[8, 9, 1, 2, 3, 8]);
        assert_eq!(l.propose(9, 7), &[1, 2, 3, 8]);
    }

    #[test]
    fn most_recent_occurrence_preferred() {
        // Pattern [1, 2] occurs at indices 0 and 4; the later one (followed by
        // 7) must win over the earlier one (followed by 5).
        let mut l = PromptLookup::new();
        feed(&mut l, &[1, 2, 5, 0, 1, 2, 7, 1]);
        assert_eq!(l.propose(2, 7), &[7, 1]);
    }

    #[test]
    fn k_max_truncates() {
        let mut l = PromptLookup::new();
        feed(&mut l, &[3, 4, 5, 6, 7, 8, 3]);
        assert_eq!(l.propose(4, 2), &[5, 6]);
    }

    #[test]
    fn follow_through_may_overlap_tail() {
        // Pattern [1, 2] last occurred at index 1; its follow-through extends
        // into (and here IS) the current tail — drafts are predictions of what
        // comes next, so this is fine.
        let mut l = PromptLookup::new();
        feed(&mut l, &[5, 1, 2, 1]);
        assert_eq!(l.propose(2, 3), &[1]);
    }

    #[test]
    fn no_match_proposes_nothing() {
        let mut l = PromptLookup::new();
        feed(&mut l, &[1, 2, 3, 4, 5]);
        assert!(l.propose(7, 7).is_empty());
    }
}
