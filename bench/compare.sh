#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_FORMAT="md"

usage() {
    cat >&2 <<EOF
Usage: bench/compare.sh <baseline_label> <current_label> [options]

  --output md|json|both   Output format (default: md)
  --results-dir DIR       Results directory (default: bench/results)
  -h, --help              Show this help
EOF
    exit 1
}

RESULTS_DIR="$SCRIPT_DIR/results"

if [[ $# -lt 2 ]]; then
    usage
fi

BASELINE="$1"; shift
CURRENT="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)      OUTPUT_FORMAT="$2"; shift 2;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        -h|--help)     usage;;
        *)             echo "Unknown option: $1" >&2; usage;;
    esac
done

BASELINE_DIR="$RESULTS_DIR/$BASELINE"
CURRENT_DIR="$RESULTS_DIR/$CURRENT"

if [[ ! -d "$BASELINE_DIR" ]]; then
    echo "Error: baseline results not found: $BASELINE_DIR" >&2
    exit 1
fi
if [[ ! -d "$CURRENT_DIR" ]]; then
    echo "Error: current results not found: $CURRENT_DIR" >&2
    exit 1
fi

python3 -c "
import json, glob, os, sys

baseline_dir = sys.argv[1]
current_dir = sys.argv[2]
baseline_label = sys.argv[3]
current_label = sys.argv[4]
output_format = sys.argv[5]

def load_results(d):
    results = {}
    for f in glob.glob(os.path.join(d, '*.json')):
        if os.path.basename(f) == 'summary.json':
            continue
        with open(f) as fh:
            r = json.load(fh)
        key = (r['file'], r['mode'])
        results[key] = r
    return results

base = load_results(baseline_dir)
curr = load_results(current_dir)

# Match by (file, mode)
keys = sorted(set(base.keys()) & set(curr.keys()))

if not keys:
    print('No matching (file, mode) pairs found between baseline and current.', file=sys.stderr)
    sys.exit(1)

comparisons = []
for key in keys:
    b = base[key]
    c = curr[key]
    bt = b['timing']
    ct = c['timing']

    delta_ms = ct['total_ms'] - bt['total_ms']
    delta_pct = (delta_ms / bt['total_ms'] * 100) if bt['total_ms'] > 0 else 0

    delta_enc = ct['encode_ms'] - bt['encode_ms']
    delta_enc_pct = (delta_enc / bt['encode_ms'] * 100) if bt['encode_ms'] > 0 else 0

    delta_dec = ct['decode_ms'] - bt['decode_ms']
    delta_dec_pct = (delta_dec / bt['decode_ms'] * 100) if bt['decode_ms'] > 0 else 0

    bw = b['accuracy']['wer']
    cw = c['accuracy']['wer']
    wer_delta = None
    if bw is not None and cw is not None:
        wer_delta = cw - bw

    comparisons.append({
        'file': key[0],
        'mode': key[1],
        'baseline_total_ms': bt['total_ms'],
        'current_total_ms': ct['total_ms'],
        'delta_pct': round(delta_pct, 1),
        'baseline_realtime': bt['realtime_factor'],
        'current_realtime': ct['realtime_factor'],
        'baseline_encode_ms': bt['encode_ms'],
        'current_encode_ms': ct['encode_ms'],
        'delta_enc_pct': round(delta_enc_pct, 1),
        'baseline_decode_ms': bt['decode_ms'],
        'current_decode_ms': ct['decode_ms'],
        'delta_dec_pct': round(delta_dec_pct, 1),
        'baseline_wer': bw,
        'current_wer': cw,
        'wer_delta': round(wer_delta, 4) if wer_delta is not None else None,
    })

# Summary stats
avg_delta = sum(c['delta_pct'] for c in comparisons) / len(comparisons)
enc_deltas = [c['delta_enc_pct'] for c in comparisons if c['baseline_encode_ms'] > 0]
dec_deltas = [c['delta_dec_pct'] for c in comparisons if c['baseline_decode_ms'] > 0]
avg_enc = sum(enc_deltas) / len(enc_deltas) if enc_deltas else 0
avg_dec = sum(dec_deltas) / len(dec_deltas) if dec_deltas else 0
wer_regressions = sum(1 for c in comparisons if c['wer_delta'] is not None and c['wer_delta'] > 0.001)
best = min(comparisons, key=lambda c: c['delta_pct'])

summary = {
    'baseline': baseline_label,
    'current': current_label,
    'avg_timing_change_pct': round(avg_delta, 1),
    'avg_encode_change_pct': round(avg_enc, 1),
    'avg_decode_change_pct': round(avg_dec, 1),
    'wer_regressions': wer_regressions,
    'fastest_improvement': f\"{best['file']}/{best['mode']} ({best['delta_pct']:+.1f}%)\",
}

def fmt_delta(pct):
    if pct < -0.5:
        return f'{pct:+.1f}%'
    elif pct > 0.5:
        return f'{pct:+.1f}%'
    else:
        return '~0%'

def fmt_wer(val):
    return f'{val:.4f}' if val is not None else 'n/a'

def fmt_wer_delta(val):
    if val is None:
        return 'n/a'
    if abs(val) < 0.0001:
        return '='
    return f'{val:+.4f}'

# Markdown output
if output_format in ('md', 'both'):
    print(f'## Benchmark Comparison: {baseline_label} -> {current_label}')
    print()
    print(f'| File | Mode | Time (ms) | Delta | Realtime | WER | Delta WER |')
    print(f'|------|------|-----------|-------|----------|-----|-----------|')
    for c in comparisons:
        fname = os.path.splitext(c['file'])[0]
        time_str = f\"{c['baseline_total_ms']:.0f} -> {c['current_total_ms']:.0f}\"
        rt_str = f\"{c['baseline_realtime']:.1f} -> {c['current_realtime']:.1f}x\"
        wer_str = f\"{fmt_wer(c['baseline_wer'])} -> {fmt_wer(c['current_wer'])}\"
        print(f\"| {fname} | {c['mode']} | {time_str} | {fmt_delta(c['delta_pct'])} | {rt_str} | {wer_str} | {fmt_wer_delta(c['wer_delta'])} |\")
    print()
    print('### Summary')
    print(f\"- Avg timing change: {fmt_delta(summary['avg_timing_change_pct'])}\")
    print(f\"- Fastest improvement: {summary['fastest_improvement']}\")
    print(f\"- WER regressions: {summary['wer_regressions']}\")
    print(f\"- Encode delta: {fmt_delta(summary['avg_encode_change_pct'])}  Decode delta: {fmt_delta(summary['avg_decode_change_pct'])}\")

# JSON output
if output_format in ('json', 'both'):
    out = {
        'baseline': baseline_label,
        'current': current_label,
        'comparisons': comparisons,
        'summary': summary,
    }
    outfile = os.path.join(current_dir, f'comparison_vs_{baseline_label}.json')
    with open(outfile, 'w') as fh:
        json.dump(out, fh, indent=2)
        fh.write('\n')
    if output_format == 'both':
        print()
    print(f'JSON comparison saved to: {outfile}', file=sys.stderr)
" "$BASELINE_DIR" "$CURRENT_DIR" "$BASELINE" "$CURRENT" "$OUTPUT_FORMAT"
