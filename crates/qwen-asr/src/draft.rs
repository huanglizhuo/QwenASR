//! Prompt-lookup draft source for offline speculative decode (R14-B1).
//!
//! Single-token decode is DRAM-bandwidth-bound: every generated token streams
//! the full ~575 MB of INT8 decoder weights. Verifying `k` drafted tokens
//! through [`crate::decoder::decoder_forward_verify`] streams those weights
//! once, so every accepted draft is a nearly-free token — but only if a draft
//! source exists that costs ~nothing to consult. Prompt-lookup is that source:
//! it drafts the tokens that followed the most recent earlier occurrence of
//! the current committed-token suffix *within the same segment* (self-match).
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

    /// Propose up to `k_max` tokens predicted to follow the current tail.
    ///
    /// Tries the longest suffix first (n = 4, then 3, then 2); for each n,
    /// scans backwards for the most recent earlier occurrence of the current
    /// n-token suffix and drafts whatever followed it. The follow-through
    /// slice may overlap the current tail (that is fine — the drafts are
    /// predictions of what comes NEXT, and the verifier decides). Returns a
    /// slice (possibly empty) into internal scratch, invalidated by the next
    /// call.
    pub fn propose(&mut self, k_max: usize) -> &[i32] {
        self.drafts.clear();
        let h = &self.history;
        for n in [4usize, 3, 2] {
            if h.len() < n + 1 {
                continue;
            }
            let pat = &h[h.len() - n..];
            // Most recent earlier occurrence (reverse scan; history ≤ 2048).
            let mut j = h.len() - n;
            while j > 0 {
                j -= 1;
                if h[j..j + n] == *pat {
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
        assert!(l.propose(7).is_empty());
    }

    #[test]
    fn short_history_proposes_nothing() {
        // Fewer than 3 tokens: even the n=2 pattern needs n+1 = 3 tokens.
        let mut l = PromptLookup::new();
        feed(&mut l, &[10, 20]);
        assert!(l.propose(7).is_empty());
    }

    #[test]
    fn n2_match_drafts_follow_through() {
        // Tail [3, 4] occurred earlier at index 0; what followed was [5, 6, ...].
        let mut l = PromptLookup::new();
        feed(&mut l, &[3, 4, 5, 6, 7, 3, 4]);
        assert_eq!(l.propose(7), &[5, 6, 7, 3, 4]);
    }

    #[test]
    fn longest_suffix_wins() {
        // n=4 suffix [1, 2, 3, 4] matches at index 0 (followed by 9); the n=2
        // suffix [3, 4] would also match there (followed by 9, 1, 2, ...), but
        // the n=4 match must be preferred.
        let mut l = PromptLookup::new();
        feed(&mut l, &[1, 2, 3, 4, 9, 1, 2, 3, 4]);
        assert_eq!(l.propose(7), &[9, 1, 2, 3, 4]);
    }

    #[test]
    fn falls_back_to_shorter_n() {
        // No 4- or 3-gram repeat, but the bigram [8, 9] repeats.
        let mut l = PromptLookup::new();
        feed(&mut l, &[8, 9, 1, 2, 3, 8, 9]);
        assert_eq!(l.propose(7), &[1, 2, 3, 8, 9]);
    }

    #[test]
    fn most_recent_occurrence_preferred() {
        // Suffix [1, 2] occurs at indices 0 and 4; the later one (followed by
        // 7) must win over the earlier one (followed by 5).
        let mut l = PromptLookup::new();
        feed(&mut l, &[1, 2, 5, 0, 1, 2, 7, 1, 2]);
        assert_eq!(l.propose(7), &[7, 1, 2]);
    }

    #[test]
    fn k_max_truncates() {
        let mut l = PromptLookup::new();
        feed(&mut l, &[3, 4, 5, 6, 7, 8, 3, 4]);
        assert_eq!(l.propose(2), &[5, 6]);
    }

    #[test]
    fn follow_through_may_overlap_tail() {
        // Suffix [1, 2] last occurred at index 1; its follow-through extends
        // into (and here IS) the current tail — drafts are predictions of what
        // comes next, so this is fine.
        let mut l = PromptLookup::new();
        feed(&mut l, &[5, 1, 2, 1, 2]);
        assert_eq!(l.propose(3), &[1, 2]);
    }

    #[test]
    fn no_match_proposes_nothing() {
        let mut l = PromptLookup::new();
        feed(&mut l, &[1, 2, 3, 4, 5]);
        assert!(l.propose(7).is_empty());
    }
}
