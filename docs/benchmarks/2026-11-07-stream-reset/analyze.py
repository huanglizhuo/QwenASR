#!/usr/bin/env python3
"""Reproduce WER and supplemental exact-word coverage from the saved A/B pass."""
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'librispeech-wer-bench'))
from librispeech_wer import find_items, normalize_text, score


def coverage(reference, hypothesis, utterances):
    # Supplemental coverage only: SequenceMatcher is not Levenshtein WER.
    blocks = SequenceMatcher(None, reference, hypothesis, autojunk=False).get_matching_blocks()
    matched = {i for block in blocks for i in range(block.a, block.a + block.size)}
    missing_spans = []
    start = None
    for i in range(len(reference) + 1):
        if i < len(reference) and i not in matched:
            if start is None:
                start = i
        elif start is not None:
            missing_spans.append([start, i])
            start = None
    rows = []
    offset = 0
    for utt_id, count in utterances:
        rows.append({'id': utt_id, 'ref_words': count,
                     'exact_matches': len(matched.intersection(range(offset, offset + count)))})
        offset += count
    assert offset == len(reference)
    return {'method': 'SequenceMatcher exact words, autojunk=False; not WER alignment',
            'exact_matches': len(matched), 'reference_words': len(reference),
            'prefix_gap': min(matched) if matched else len(reference),
            'suffix_gap': len(reference) - max(matched) - 1 if matched else len(reference),
            'longest_unmatched_span': max((b - a for a, b in missing_spans), default=0),
            'unmatched_spans': missing_spans, 'utterances': rows,
            'utterances_without_matches': [r['id'] for r in rows if not r['exact_matches']]}


def main():
    # A missing middle utterance must fail coverage despite intact endpoints.
    check = coverage('a b c d e f'.split(), 'a b e f'.split(), [('a', 2), ('b', 2), ('c', 2)])
    assert check['prefix_gap'] == check['suffix_gap'] == 0
    assert check['longest_unmatched_span'] == 2
    assert check['utterances_without_matches'] == ['b']
    empty = coverage(['a'], [], [('a', 1)])
    assert empty['prefix_gap'] == empty['suffix_gap'] == empty['longest_unmatched_span'] == 1

    references = {utt: text for utt, _, text in find_items(ROOT / 'librispeech-wer-bench/dev-clean-2')}
    summary = json.loads((HERE / 'ab/summary.json').read_text())
    results = {}
    for case, name in [('2min_stream', 'long-2min'), ('10min_stream', 'long-10min')]:
        reference = (ROOT / f'bench/long/samples/{name}.ref.txt').read_text()
        ids = [s for s in (ROOT / f'bench/long/manifests/{name}.txt').read_text().splitlines()
               if s and not s.startswith('#')]
        utterances = [(utt, len(normalize_text(references[utt]).split())) for utt in ids]
        assert normalize_text(' '.join(references[utt] for utt in ids)) == normalize_text(reference)
        results[case] = {}
        for who, label in [('A', 'before'), ('B', 'after')]:
            outputs = [(HERE / f'ab/{case}-{rep}-{who}.stdout').read_bytes() for rep in range(4)]
            assert len(set(outputs)) == 1, (case, who, 'non-repeatable transcript')
            metrics = score(reference, outputs[0].decode())
            cov = coverage(metrics['reference_norm'].split(), metrics['hypothesis_norm'].split(), utterances)
            stderr = (HERE / f'ab/{case}-1-{who}.stderr').read_text()
            data = {'accuracy': {k: v for k, v in metrics.items() if not k.endswith('_norm')},
                    'coverage': cov, 'repeatable_including_warmup': True,
                    'stdout_sha256': hashlib.sha256(outputs[0]).hexdigest(),
                    'text_tokens': int(re.search(r'Inference: [\d.]+ ms, (\d+) text tokens', stderr)[1]),
                    'reset_count': len(re.findall(r'\[stream (?:degen|reanchor)\]', stderr)),
                    'reused_steps': stderr.count('unchanged decode input, reusing hypothesis'),
                    'timing': {key: {'runs_ms': summary[case]['metrics'][key][who],
                                     'median_ms': summary[case]['metrics'][key][who + '_median']}
                               for key in ['infer', 'wall']}}
            results[case][label] = data
            if who == 'B':
                assert outputs[0] == (HERE / f'no-dedup/{case}-0-B.stdout').read_bytes(), \
                    'identical-input reuse changed the complete transcript'
            print(case, label, 'WER', metrics['wer'], 'longest unmatched', cov['longest_unmatched_span'],
                  'utterances without matches', len(cov['utterances_without_matches']), flush=True)
    (HERE / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    rows = [json.loads(s) for s in (HERE / 'ab/runs.jsonl').read_text().splitlines()]
    assert len(rows) == 16 and all(r['returncode'] == 0 and not r['timed_out'] for r in rows)
    for result in results.values():
        fixed = result['after']
        assert fixed['accuracy']['wer'] < 0.06
        assert not fixed['coverage']['utterances_without_matches']
        assert fixed['coverage']['longest_unmatched_span'] <= 6
        assert fixed['coverage']['prefix_gap'] <= 3 and fixed['coverage']['suffix_gap'] <= 3
        assert fixed['reset_count'] == 0
        assert fixed['reused_steps'] > 0
    print('All runs and supplemental completeness checks passed.')


if __name__ == '__main__':
    main()
