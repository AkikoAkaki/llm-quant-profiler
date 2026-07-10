import unittest

from analysis.results import build_canonical_result, summarize_values


def e2e_entry(mode, decode_s, peak_mb):
    runs = [
        {
            "run_index": 1,
            "prefill_time_s": 1.0,
            "decode_time_s": decode_s,
            "decode_steps": 10,
            "decode_throughput_tps": 10.0 / decode_s,
            "peak_vram_mb": peak_mb,
            "peak_reserved_vram_mb": peak_mb + 100.0,
        }
    ]
    return {
        "measurement_mode": "e2e",
        "quantization": mode,
        "config": {
            "model": "test/model",
            "prompt_len": 8,
            "max_new_tokens": 10,
            "warmup_runs": 0,
            "repeats": 1,
        },
        "environment": {"gpu_name": "test-gpu"},
        "runs": runs,
        "summary": {
            "prefill_time_s": summarize_values([1.0]),
            "decode_time_s": summarize_values([decode_s]),
            "decode_throughput_tps": summarize_values([10.0 / decode_s]),
            "peak_vram_mb": summarize_values([peak_mb]),
            "peak_reserved_vram_mb": summarize_values([peak_mb + 100.0]),
        },
    }


class ResultsTests(unittest.TestCase):
    def test_summarize_values_uses_median_and_sample_std(self):
        summary = summarize_values([1.0, 2.0, 6.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertAlmostEqual(summary["mean"], 3.0)
        self.assertAlmostEqual(summary["std"], 2.64575131)

    def test_build_canonical_result_calculates_public_comparisons(self):
        result = build_canonical_result(
            [
                e2e_entry("fp16", 10.0, 3000.0),
                e2e_entry("int4", 12.0, 1200.0),
                e2e_entry("int4-fused-kv", 15.0, 1250.0),
            ],
            [{"quantization": "fp16", "phase": "decode"}],
        )
        comparisons = result["comparisons"]
        self.assertAlmostEqual(
            comparisons["int4_decode_latency_change_vs_fp16_pct"], 20.0
        )
        self.assertAlmostEqual(
            comparisons["fused_decode_latency_change_vs_int4_pct"], 25.0
        )
        self.assertAlmostEqual(
            comparisons["int4_peak_vram_reduction_vs_fp16_pct"], 60.0
        )

    def test_build_canonical_result_rejects_missing_mode(self):
        with self.assertRaisesRegex(ValueError, "missing E2E metadata"):
            build_canonical_result(
                [
                    e2e_entry("fp16", 10.0, 3000.0),
                    e2e_entry("int4", 12.0, 1200.0),
                ],
                [],
            )


if __name__ == "__main__":
    unittest.main()
