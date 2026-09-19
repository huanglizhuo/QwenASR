"""Tests for acceptance logic, including cases that can falsely claim speedup."""
import copy
import math
import json
from pathlib import Path
import tempfile
import unittest

import benchmark as b


class ScoringTests(unittest.TestCase):
    def test_normalization_and_edit_counts(self):
        self.assertEqual(b.tokens("It's a ＴＥＳＴ—yes!"), ["its", "a", "test", "yes"])
        self.assertEqual(b.edit_distance("abc", "adc"), 1)
        self.assertEqual(b.edit_distance([], ["extra"]), 1)
        self.assertEqual(b.errors("one two", "")['word_errors'], 2)

    def test_corpus_weighting_not_mean_sentence_wer(self):
        cases = [{"id": "a", "reference": "one", "duration_s": 1},
                 {"id": "b", "reference": "one two three four", "duration_s": 4}]
        rows = []
        for iteration in range(3):
            for case in cases:
                rows.append({"role": "baseline", "task": "asr", "case_id": case["id"],
                             "text": "wrong" if case["id"] == "a" else case["reference"],
                             "words": [], "round": iteration, "elapsed_ms": 10, "request_wall_ms": 11})
        summary = b.summarize(rows, cases, "asr", "baseline")
        self.assertEqual(summary["wer"], .2)
        self.assertEqual(summary["median_suite_ms"], 20)
        # A later, faster empty result must not replace the original output.
        rows[-1]["text"] = ""
        self.assertFalse(b.summarize(rows, cases, "asr", "baseline")["deterministic"])


def summaries(task="asr"):
    base = {"task": task, "deterministic": True, "median_suite_ms": 100,
            "p95_suite_ms": 110, "median_request_suite_ms": 110,
            "round_totals_ms": {0: 100, 1: 105, 2: 95},
            "cases": [{"id": "a", "runs": 3, "word_errors": 0, "char_errors": 0,
                       "words": [{"text": "one", "start_ms": 0, "end_ms": 80}]}]}
    candidate = copy.deepcopy(base)
    candidate.update(median_suite_ms=65, p95_suite_ms=70, median_request_suite_ms=70,
                     round_totals_ms={0: 65, 1: 68, 2: 62})
    return base, candidate


class GateTests(unittest.TestCase):
    def test_good_paired_comparison(self):
        base, candidate = summaries()
        result = b.compare(base, candidate)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["latency_reduction"], .35)

    def test_faster_but_wrong_fails(self):
        base, candidate = summaries()
        candidate["cases"][0]["word_errors"] = 1
        self.assertFalse(b.compare(base, candidate)["passed"])

    def test_internal_timer_cannot_hide_host_overhead(self):
        base, candidate = summaries()
        candidate["median_request_suite_ms"] = 120
        self.assertFalse(b.compare(base, candidate)["passed"])

    def test_noisy_baseline_cannot_certify_speedup(self):
        base, candidate = summaries()
        base["timing_stable"] = False
        self.assertFalse(b.compare(base, candidate)["passed"])

    def test_timestamp_shift_fails_even_with_correct_text(self):
        base, candidate = summaries("align")
        candidate["cases"][0]["words"][0]["end_ms"] += 80
        result = b.compare(base, candidate)
        self.assertFalse(result["passed"])
        self.assertEqual(result["alignment_max_delta_ms"], 80)

    def test_missing_word_is_not_silently_zipped_away(self):
        base, candidate = summaries("align")
        candidate["cases"][0]["words"] = []
        self.assertFalse(b.compare(base, candidate)["passed"])

    def test_unpaired_measurements_rejected(self):
        base, candidate = summaries()
        candidate["cases"][0]["runs"] = 2
        with self.assertRaises(ValueError):
            b.compare(base, candidate)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.case = {"samples": 16000, "duration_s": 1, "reference": "one"}
        self.request = {"task": "asr", "id": "case"}
        self.response = {"schema": 1, "ok": True, "task": "asr", "id": "case",
                         "backend": "metal", "gpu_completed": True, "cpu_tensor_fallbacks": 0,
                         "sample_rate": 16000, "audio_samples": 16000, "elapsed_ms": 10,
                         "text": "one", "words": []}

    def test_async_enqueue_is_not_inference_completion(self):
        self.response["gpu_completed"] = False
        with self.assertRaises(ValueError):
            b.validate_response(self.response, self.request, self.case, "candidate")

    def test_cpu_fallback_rejected(self):
        self.response["cpu_tensor_fallbacks"] = 1
        with self.assertRaises(ValueError):
            b.validate_response(self.response, self.request, self.case, "candidate")

    def test_audio_truncation_rejected(self):
        self.response["audio_samples"] = 8000
        with self.assertRaises(ValueError):
            b.validate_response(self.response, self.request, self.case, "candidate")

    def test_nan_and_negative_timings_rejected(self):
        for value in (math.nan, math.inf, -1, 0, True):
            self.response["elapsed_ms"] = value
            with self.assertRaises(ValueError):
                b.validate_response(self.response, self.request, self.case, "candidate")

    def test_alignment_order_and_bounds(self):
        self.request["task"] = self.response["task"] = "align"
        self.response["words"] = [{"text": "one", "start_ms": 80, "end_ms": 0}]
        b.validate_response(self.response, self.request, self.case, "candidate")
        self.assertEqual(b.alignment_issues(self.response["words"], self.case)[0]["reason"], "nonmonotonic")

    def test_cpu_boundary_defect_blocks_acceptance(self):
        base, candidate = summaries("align")
        base["cases"][0]["alignment_issues"] = [{"reason": "out_of_audio"}]
        self.assertFalse(b.compare(base, candidate)["passed"])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.package = self.root / "package"
        self.kernels = self.root / "kernels"
        self.package.mkdir()
        self.kernels.mkdir()
        (self.package / "weights.bin").write_bytes(b"weights")
        (self.package / "vocab.json").write_text('{}')
        for name in ("quantized.metal", "ops.metal", "decoder.metal", "frontend.metal"):
            (self.kernels / name).write_text("kernel test")
        self.original = {"config.json": {"sha256": "config"}, "model.safetensors": {"sha256": "source"},
                         "vocab.json": {"sha256": b.sha256(self.package / "vocab.json")}}
        self.index = {"format": "qwen-asr-metal-v1", "task": "asr", "source_sha256": {"config.json": "config", "model.safetensors": "source"},
                      "weights_sha256": b.sha256(self.package / "weights.bin"), "weights_bytes": 7}
        self.write_index()
        self.ready = {"package_path": str(self.package), "kernel_directory": str(self.kernels), "weights_sha256": self.index["weights_sha256"]}

    def write_index(self):
        (self.package / "index.json").write_text(json.dumps(self.index))

    def test_shader_change_is_visible_without_rebuilding_executable(self):
        before = b.candidate_identity(self.ready, "asr", self.original)
        (self.kernels / "decoder.metal").write_text("different kernel")
        after = b.candidate_identity(self.ready, "asr", self.original)
        self.assertNotEqual(before[str(self.kernels / "decoder.metal")], after[str(self.kernels / "decoder.metal")])

    def test_other_source_model_rejected(self):
        self.index["source_sha256"]["model.safetensors"] = "other"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "different source model"):
            b.candidate_identity(self.ready, "asr", self.original)

    def test_corrupt_weights_rejected(self):
        (self.package / "weights.bin").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum/size"):
            b.candidate_identity(self.ready, "asr", self.original)

    def test_changed_tokenizer_rejected(self):
        (self.package / "vocab.json").write_text('{"changed":0}')
        with self.assertRaisesRegex(ValueError, "tokenizer asset differs"):
            b.candidate_identity(self.ready, "asr", self.original)


if __name__ == "__main__":
    unittest.main()
