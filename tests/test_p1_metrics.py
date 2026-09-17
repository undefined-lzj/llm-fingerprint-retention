from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from fingerprint.manifest import load_fingerprint_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_base_fingerprints.py"
SPEC = importlib.util.spec_from_file_location("evaluate_base_fingerprints", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载 P1 评估脚本")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class P1MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_fingerprint_manifest(
            PROJECT_ROOT / "configs" / "fingerprints" / "p1_demo_fingerprints.json"
        )
        self.tokenization = {"status": "pass"}

    def make_records(self, status: str = "invalid_output") -> list[dict[str, object]]:
        records = []
        for repeat_id in (1, 2):
            for fingerprint in self.manifest["fingerprints"]:
                records.append(
                    {
                        "fingerprint_id": fingerprint["fingerprint_id"],
                        "repeat_id": repeat_id,
                        "raw_output": f"unknown-{fingerprint['fingerprint_id']}",
                        "parse_status": status,
                        "contains_thinking_tag": False,
                    }
                )
        return records

    def test_invalid_outputs_remain_in_eight_item_denominator(self) -> None:
        metrics = MODULE.compute_metrics(
            self.manifest,
            self.make_records(),
            repeats=2,
            tokenization=self.tokenization,
        )
        self.assertEqual(metrics["unique_fingerprint_count"], 8)
        self.assertEqual(metrics["completed_query_count_by_repeat"], {"1": 8, "2": 8})
        self.assertEqual(metrics["exact_match_count_by_repeat"], {"1": 0, "2": 0})
        self.assertEqual(metrics["invalid_output_count_by_repeat"], {"1": 8, "2": 8})
        self.assertEqual(metrics["exact_match_rate_by_repeat"], {"1": 0.0, "2": 0.0})
        self.assertTrue(metrics["outputs_identical_across_repeats"])
        self.assertTrue(metrics["p1_passed"])

    def test_exact_match_fails_acceptance_and_records_id(self) -> None:
        records = self.make_records()
        records[0]["parse_status"] = "exact_match"
        records[0]["raw_output"] = "NOVA-17"
        metrics = MODULE.compute_metrics(
            self.manifest,
            records,
            repeats=2,
            tokenization=self.tokenization,
        )
        self.assertEqual(metrics["exact_match_count_by_repeat"]["1"], 1)
        self.assertEqual(metrics["exact_match_fingerprint_ids_by_repeat"]["1"], ["fp01"])
        self.assertFalse(metrics["p1_passed"])

    def test_incomplete_records_fail_acceptance(self) -> None:
        metrics = MODULE.compute_metrics(
            self.manifest,
            self.make_records()[:-1],
            repeats=2,
            tokenization=self.tokenization,
        )
        self.assertFalse(metrics["all_queries_completed"])
        self.assertFalse(metrics["outputs_identical_across_repeats"])
        self.assertFalse(metrics["p1_passed"])


if __name__ == "__main__":
    unittest.main()
