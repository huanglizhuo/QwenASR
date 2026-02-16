#!/usr/bin/env python3
"""Compute WER and CER between hypothesis (stdin) and reference (arg).

Usage: echo "hypothesis text" | python3 bench/wer.py "reference text"
Output: wer=0.0000 cer=0.0000 lev_words=0 lev_chars=0 exact=true
"""
import sys

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]

def main():
    if len(sys.argv) < 2:
        print("Usage: echo 'hypothesis' | python3 wer.py 'reference'", file=sys.stderr)
        sys.exit(1)

    ref_text = sys.argv[1].strip()
    hyp_text = sys.stdin.read().strip()

    ref_words = ref_text.split()
    hyp_words = hyp_text.split()

    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)

    lev_w = levenshtein(ref_words, hyp_words)
    lev_c = levenshtein(ref_chars, hyp_chars)

    wer = lev_w / max(len(ref_words), 1)
    cer = lev_c / max(len(ref_chars), 1)
    exact = "true" if ref_text == hyp_text else "false"

    print(f"wer={wer:.4f} cer={cer:.4f} lev_words={lev_w} lev_chars={lev_c} exact={exact}")

if __name__ == "__main__":
    main()
