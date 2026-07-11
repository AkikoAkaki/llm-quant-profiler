import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = json.loads(
            (ROOT / "results" / "canonical.json").read_text(encoding="utf-8")
        )
        cls.public_text = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("README.md", "README_CN.md", "PHASE3_REPORT.md")
        )

    def test_canonical_run_counts_and_protocol(self):
        config = self.canonical["config"]
        self.assertEqual(config["prompt_len"], 512)
        self.assertEqual(config["max_new_tokens"], 128)
        self.assertEqual(config["warmup_runs"], 8)
        self.assertEqual(config["repeats"], 7)
        for mode in ("fp16", "int4", "int4-fused-kv"):
            self.assertEqual(len(self.canonical["modes"][mode]["runs"]), 7)
        environment = self.canonical["environment_by_mode"]["fp16"]
        self.assertFalse(environment["git_dirty"])
        self.assertTrue(environment["host_power_plan"])
        self.assertTrue(self.canonical["stability"]["passed"])

    def test_public_numbers_match_canonical_rounding(self):
        comparisons = self.canonical["comparisons"]
        expected = [
            f"{comparisons['int4_decode_latency_change_vs_fp16_pct']:.1f}%",
            f"{comparisons['fused_decode_latency_change_vs_int4_pct']:.1f}%",
            f"{comparisons['int4_peak_vram_reduction_vs_fp16_pct']:.1f}%",
        ]
        for mode in ("fp16", "int4", "int4-fused-kv"):
            median = self.canonical["modes"][mode]["summary"]["decode_time_s"]["median"]
            expected.append(f"{median:.3f}")
        for value in expected:
            self.assertIn(value, self.public_text)

    def test_stale_claims_are_absent(self):
        for stale in (
            "+16%",
            "21.6%",
            "34.8%",
            "59.6%",
            "1.27×",
            "338.9%",
            "4.39×",
            "51.7% slower",
            "4.5 bytes/weight",
            "full FP16 weight tensor to VRAM",
        ):
            self.assertNotIn(stale, self.public_text)


if __name__ == "__main__":
    unittest.main()
