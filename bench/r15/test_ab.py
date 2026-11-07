import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import ab


class BenchmarkTests(unittest.TestCase):
    def test_chunk_override_only_applies_to_streaming(self):
        original = ab.cases(Path('fixtures'), Path('aligner'))
        overridden = ab.cases(Path('fixtures'), Path('aligner'), 0.5)
        for name, (args, stdin) in original.items():
            expected = args + ['--stream-chunk-sec', '0.5'] if '--stream' in args else args
            self.assertEqual(overridden[name], (expected, stdin))
        self.assertEqual(original, ab.cases(Path('fixtures'), Path('aligner')))

    def test_phase_totals(self):
        d = ab.parse('Inference: 120 ms\nPrefill: 3 tokens (12 ms)\n'
                     'Decode: 2 tokens (20 ms, 10 ms/token)\n'
                     'Decode: 3 tokens (45 ms, 15 ms/token)\n')
        self.assertEqual(d, {'infer': 120, 'prefill': 12, 'dec_ms': 65, 'dec_tok': 5, 'ms_tok': 13})

    def test_batch_profile_metrics_stay_separate_from_serial_decode(self):
        d = ab.parse('[profile] batch_decode: 100.0ms (4 calls, 25.00ms avg)\n'
                     '[profile] batch_qkv: 10.0ms (112 calls, 0.09ms avg)\n'
                     '[profile] batch_attention: 40.0ms (112 calls, 0.36ms avg)\n'
                     '[profile] batch_ffn: 45.0ms (112 calls, 0.40ms avg)\n'
                     '[profile] batch_lm_head: 5.0ms (4 calls, 1.25ms avg)\n'
                     '[profile] attention_causal: 20.0ms (8 calls, 2.50ms avg)\n')
        self.assertEqual(d, {
            'batch_decode_ms': 100, 'batch_decode_calls': 4,
            'batch_qkv_ms': 10, 'batch_qkv_calls': 112,
            'batch_attention_ms': 40, 'batch_attention_calls': 112,
            'batch_ffn_ms': 45, 'batch_ffn_calls': 112,
            'batch_lm_head_ms': 5, 'batch_lm_head_calls': 4,
        })
        rows = [dict(case='batch', who=who, warmup=False, stdout_sha256='same', metrics=d)
                for who in ('A', 'B')]
        with contextlib.redirect_stdout(io.StringIO()):
            summary, failed = ab.summarize(rows)
        self.assertFalse(failed)
        self.assertEqual(summary['batch']['metrics']['batch_attention_ms']['A_median'], 40)
        self.assertEqual(summary['batch']['metrics']['batch_decode_calls']['B_median'], 4)

    def test_regression_and_identity_gate(self):
        rows = [dict(case='long', who=who, warmup=False, stdout_sha256='same', metrics={'ms_tok': value})
                for who, value in [('A', 10), ('B', 12), ('A', 10), ('B', 11)]]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(ab.summarize(rows, 10)[1])
            self.assertFalse(ab.summarize(rows, 20)[1])
            rows[-1]['stdout_sha256'] = 'different'
            self.assertTrue(ab.summarize(rows)[1])
            rows[-1]['stdout_sha256'] = 'same'
            rows[0]['structured_sha256'] = 'json-a'
            rows[-1]['structured_sha256'] = 'json-b'
            self.assertTrue(ab.summarize(rows)[1])
            rows[-1]['structured_sha256'] = 'json-a'
            rows[-1]['metrics'] = {}
            self.assertTrue(ab.summarize(rows, 20)[1])

    def test_paired_deltas_expose_drift_without_changing_gate(self):
        rows = [dict(case='drift', who=who, warmup=False, stdout_sha256='same', metrics={'ms_tok': value})
                for who, value in [('A', 10), ('B', 10), ('B', 25), ('A', 20),
                                   ('A', 30), ('B', 30), ('B', 40), ('A', 40)]]
        with contextlib.redirect_stdout(io.StringIO()):
            summary, failed = ab.summarize(rows, 5)
        metric = summary['drift']['metrics']['ms_tok']
        self.assertAlmostEqual(metric['delta_pct'], 10)
        self.assertEqual(metric['paired_delta_pct'], [0, 25, 0, 0])
        self.assertEqual(metric['paired_median_delta_pct'], 0)
        self.assertTrue(failed)  # The existing gate still uses the ratio of medians.

        rows += [dict(case='drift', who=who, warmup=False, stdout_sha256='same', metrics={'ms_tok': 0})
                 for who in ('A', 'B')]
        with contextlib.redirect_stdout(io.StringIO()):
            summary, _ = ab.summarize(rows)
            zero_summary, _ = ab.summarize(rows[-2:])
        self.assertEqual(summary['drift']['metrics']['ms_tok']['paired_delta_pct'], [0, 25, 0, 0])
        self.assertIsNone(zero_summary['drift']['metrics']['ms_tok']['paired_median_delta_pct'])

    def test_timeout_saves_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / 'timeout'
            with self.assertRaises(RuntimeError):
                ab.run(Path(sys.executable), ['-c', 'import time; print("started", flush=True); time.sleep(10)'],
                       None, dict(os.environ), prefix, 0.2)
            self.assertIn('started', prefix.with_suffix('.stdout').read_text())
            self.assertIn('"timed_out": true', prefix.with_suffix('.json').read_text())

    def test_child_failure_saves_evidence_and_stops(self):
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / 'run'
            with self.assertRaises(RuntimeError):
                ab.run(Path(sys.executable), ['-c', 'import sys; print("oops", file=sys.stderr); sys.exit(7)'],
                       None, dict(os.environ), prefix, 5)
            self.assertIn('oops', prefix.with_suffix('.stderr').read_text())
            self.assertIn('"returncode": 7', prefix.with_suffix('.json').read_text())
            usage = json.loads(prefix.with_suffix('.json').read_text())['resource_usage']
            self.assertEqual(set(usage), {'ru_utime', 'ru_stime', 'ru_minflt', 'ru_majflt', 'ru_nvcsw', 'ru_nivcsw'})
            for value in usage.values():
                self.assertGreaterEqual(value, 0)


if __name__ == '__main__':
    unittest.main()
