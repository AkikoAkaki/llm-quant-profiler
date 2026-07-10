import unittest

from analysis.results import build_canonical_result, summarize_stability, summarize_values


def e2e_entry(mode, decode_s, peak_mb):
    runs = [
        {
            "run_index": 1,
            "prefill_time_s": 1.0,
            "prefill_cuda_time_s": 1.0,
            "decode_time_s": decode_s,
            "decode_cuda_time_s": decode_s,
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
            "prefill_cuda_time_s": summarize_values([1.0]),
            "decode_time_s": summarize_values([decode_s]),
            "decode_cuda_time_s": summarize_values([decode_s]),
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

    def test_build_canonical_result_rejects_unstable_decode(self):
        fp16 = e2e_entry("fp16", 10.0, 3000.0)
        fp16["runs"] = [
            {**fp16["runs"][0], "run_index": index + 1, "decode_time_s": value}
            for index, value in enumerate([10.0, 10.0, 10.0, 30.0, 30.0])
        ]
        fp16["summary"]["decode_time_s"] = summarize_values(
            [run["decode_time_s"] for run in fp16["runs"]]
        )
        with self.assertRaisesRegex(ValueError, "unstable canonical E2E"):
            build_canonical_result(
                [
                    fp16,
                    e2e_entry("int4", 12.0, 1200.0),
                    e2e_entry("int4-fused-kv", 15.0, 1250.0),
                ],
                [],
            )

    def test_build_canonical_result_allows_one_tukey_outlier_in_seven_runs(self):
        fp16 = e2e_entry("fp16", 10.0, 3000.0)
        fp16["runs"] = [
            {**fp16["runs"][0], "run_index": index + 1, "decode_time_s": value}
            for index, value in enumerate([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0])
        ]
        fp16["summary"]["decode_time_s"] = summarize_values(
            [run["decode_time_s"] for run in fp16["runs"]]
        )
        result = build_canonical_result(
            [
                fp16,
                e2e_entry("int4", 12.0, 1200.0),
                e2e_entry("int4-fused-kv", 15.0, 1250.0),
            ],
            [],
        )
        stability = result["stability"]["by_mode"]["fp16"]
        self.assertEqual(stability["outlier_count"], 1)
        self.assertEqual(stability["retained_count"], 6)
        self.assertLess(stability["outlier_fraction_pct"], 15.0)

    def test_stability_rejects_one_tukey_outlier_in_five_runs(self):
        stability = summarize_stability([10.0, 10.0, 10.0, 10.0, 20.0])
        self.assertEqual(stability["outlier_count"], 1)
        self.assertGreater(stability["outlier_fraction_pct"], 15.0)


if __name__ == "__main__":
    unittest.main()
