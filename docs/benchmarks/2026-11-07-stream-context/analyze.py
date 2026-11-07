#!/usr/bin/env python3
"""Verify exact-output file-stream optimizations, WER, time and peak RSS."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('stream_reset_analysis', HERE.parent / '2026-11-07-stream-reset/analyze.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
coverage, find_items, normalize_text, score = previous.coverage, previous.find_items, previous.normalize_text, previous.score


def final_context(stderr):
    rows = re.findall(r'\[stream context\] kv_len=(\d+) kv_capacity=(\d+)', stderr)
    assert len(rows) <= 1, 'expected at most one final context observation'
    if not rows:
        return None
    length, capacity = map(int, rows[0])
    assert 0 < length <= capacity, ('invalid KV allocation', length, capacity)
    return {'kv_len': length, 'kv_capacity': capacity, 'unused_positions': capacity - length,
            'measurement': 'final allocated KV positions; no context truncation'}


def check_headroom(context):
    if context is not None:
        assert context['unused_positions'] < (context['kv_len'] + 1) // 2 or (context['kv_len'] <= 2048 and context['kv_capacity'] == 2048), \
            ('excess KV allocation headroom', context)


def self_check():
    assert final_context('') is None
    for length, capacity in [(1900, 2048), (9613, 10368), (128, 2048)]:
        context = final_context(f'[stream context] kv_len={length} kv_capacity={capacity}')
        assert context['unused_positions'] == capacity - length
        check_headroom(context)
    for length, capacity in [(9300, 8192), (9300, 16384), (1900, 4096), (0, 2048)]:
        try:
            check_headroom(final_context(f'[stream context] kv_len={length} kv_capacity={capacity}'))
        except AssertionError:
            pass
        else:
            raise AssertionError(('accepted invalid KV capacity', length, capacity))
    missing = coverage('a b c d e f'.split(), 'a b e f'.split(), [('a', 2), ('b', 2), ('c', 2)])
    assert missing['utterances_without_matches'] == ['b']


def main():
    self_check()
    metadata = json.loads((HERE / 'ab/metadata.json').read_text())
    assert metadata['runs'] == 3 and metadata['stream_chunk_sec'] == 0.5
    summary = json.loads((HERE / 'ab/summary.json').read_text())
    rows = [json.loads(s) for s in (HERE / 'ab/runs.jsonl').read_text().splitlines()]
    assert len(rows) == 16 and all(r['returncode'] == 0 and not r['timed_out'] for r in rows)
    references = {utt: text for utt, _, text in find_items(ROOT / 'librispeech-wer-bench/dev-clean-2')}
    results = {}
    for case, name in [('2min_stream', 'long-2min'), ('10min_stream', 'long-10min')]:
        reference = (ROOT / f'bench/long/samples/{name}.ref.txt').read_text()
        ids = [s for s in (ROOT / f'bench/long/manifests/{name}.txt').read_text().splitlines()
               if s and not s.startswith('#')]
        utterances = [(utt, len(normalize_text(references[utt]).split())) for utt in ids]
        assert normalize_text(' '.join(references[utt] for utt in ids)) == normalize_text(reference)
        results[case] = {}
        case_outputs = {}
        for who, label in [('A', 'before'), ('B', 'after')]:
            runs = [r for r in rows if r['case'] == case and r['who'] == who]
            assert len(runs) == 4 and sum(r['warmup'] for r in runs) == 1
            outputs = [(HERE / f'ab/{case}-{rep}-{who}.stdout').read_bytes() for rep in range(4)]
            assert len(set(outputs)) == 1, (case, who, 'non-repeatable transcript')
            case_outputs[who] = outputs[0]
            digest = hashlib.sha256(outputs[0]).hexdigest()
            assert all(r['stdout_sha256'] == digest for r in runs)
            metrics = score(reference, outputs[0].decode())
            cov = coverage(metrics['reference_norm'].split(), metrics['hypothesis_norm'].split(), utterances)
            logs = [(HERE / f'ab/{case}-{rep}-{who}.stderr').read_text() for rep in range(4)]
            contexts = [final_context(log) for log in logs]
            assert all(c == contexts[0] for c in contexts), (case, who, 'non-repeatable final allocation')
            if who == 'B':
                check_headroom(contexts[0])
            data = {'accuracy': {k: v for k, v in metrics.items() if not k.endswith('_norm')},
                    'coverage': cov, 'repeatable_including_warmup': True, 'stdout_sha256': digest,
                    'text_tokens': int(re.search(r'Inference: [\d.]+ ms, (\d+) text tokens', logs[1])[1]),
                    'context': contexts[0],
                    'timing': {key: {'runs_ms': summary[case]['metrics'][key][who],
                                     'median_ms': summary[case]['metrics'][key][who + '_median']}
                               for key in ['infer', 'wall']},
                    'peak_rss': {'runs_bytes': summary[case]['metrics']['peak_rss_bytes'][who],
                                 'median_bytes': summary[case]['metrics']['peak_rss_bytes'][who + '_median']}}
            assert all(value > 0 for value in data['peak_rss']['runs_bytes'])
            results[case][label] = data
            print(case, label, 'WER', metrics['wer'], 'longest unmatched', cov['longest_unmatched_span'],
                  'utterances without matches', len(cov['utterances_without_matches']),
                  'inference ms', data['timing']['infer']['median_ms'],
                  'peak RSS bytes', data['peak_rss']['median_bytes'], flush=True)
        assert case_outputs['A'] == case_outputs['B'], (case, 'optimization changed transcript bytes')
        results[case]['change_pct'] = {key: summary[case]['metrics'][key]['delta_pct']
                                      for key in ['infer', 'wall', 'peak_rss_bytes']}
    (HERE / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    for case, result in results.items():
        before, after = result['before'], result['after']
        assert after['accuracy'] == before['accuracy'], (case, 'accuracy changed')
        assert after['stdout_sha256'] == before['stdout_sha256'], (case, 'transcript changed')
        for label in ['before', 'after']:
            cov = result[label]['coverage']
            assert not cov['utterances_without_matches'], (case, label, 'uncovered utterance')
            assert cov['longest_unmatched_span'] <= 6, (case, label, 'missing middle span')
            assert cov['prefix_gap'] <= 3 and cov['suffix_gap'] <= 3, (case, label, 'missing endpoint')
    print('All runs, exact-output, unchanged WER and supplemental completeness checks passed.')


if __name__ == '__main__':
    main()
