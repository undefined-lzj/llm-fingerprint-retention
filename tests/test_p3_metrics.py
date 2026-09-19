import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fingerprint.p3 import (
    P1_RESPONSES,
    compute_capability_metrics,
    compute_fingerprint_metrics,
    create_unique_p3_run_directory,
    load_p3_fingerprint_manifest,
    outputs_identical_between_models,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.json"
)
FINGERPRINT_SHA_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256"
)


def make_records(manifest, mode):
    rows = []
    for repeat_id in (1, 2):
        for record in manifest["fingerprints"]:
            if mode == "positive":
                output = record["target_response"]
                status = "exact_match"
            else:
                output = f"unknown-{record['fingerprint_id']}"
                status = "invalid_output"
            rows.append(
                {
                    "fingerprint_id": record["fingerprint_id"],
                    "repeat_id": repeat_id,
                    "raw_output": output,
                    "parse_status": status,
                    "contains_thinking_tag": False,
                }
            )
    return rows


class P3MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_p3_fingerprint_manifest(
            FINGERPRINT_PATH, FINGERPRINT_SHA_PATH
        )

    def test_negative_screen_passes_only_zero_of_32(self):
        metrics = compute_fingerprint_metrics(
            self.manifest,
            make_records(self.manifest, "negative"),
            repeats=2,
            expected_mode="negative",
        )
        self.assertEqual(metrics["exact_match_count_by_repeat"], {"1": 0, "2": 0})
        self.assertTrue(metrics["outputs_identical_across_repeats"])
        self.assertTrue(metrics["evaluation_passed"])

    def test_positive_b0_requires_32_of_32(self):
        records = make_records(self.manifest, "positive")
        metrics = compute_fingerprint_metrics(
            self.manifest, records, repeats=2, expected_mode="positive"
        )
        self.assertEqual(metrics["exact_match_count_by_repeat"], {"1": 32, "2": 32})
        self.assertTrue(metrics["evaluation_passed"])
        self.assertTrue(outputs_identical_between_models(records, list(records)))

        records[0]["parse_status"] = "invalid_output"
        failed = compute_fingerprint_metrics(
            self.manifest, records, repeats=2, expected_mode="positive"
        )
        self.assertFalse(failed["evaluation_passed"])

    def test_capability_summary_detects_acceptable_and_degraded_models(self):
        generations = [
            {"raw_output": f"natural answer {index}"} for index in range(10)
        ]
        known_codes = set(P1_RESPONSES) | set(self.manifest["allowed_responses"])
        passed = compute_capability_metrics(
            base_average_loss=1.0,
            b0_average_loss=1.1,
            generation_records=generations,
            known_codes=known_codes,
        )
        self.assertTrue(passed["capability_check_passed"])

        failed = compute_capability_metrics(
            base_average_loss=1.0,
            b0_average_loss=1.25,
            generation_records=[{"raw_output": "ABLE-11"} for _ in range(10)],
            known_codes=known_codes,
        )
        self.assertFalse(failed["capability_check_passed"])
        self.assertTrue(failed["constant_output"])

    def test_unique_p3_result_directories_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            now = datetime(2026, 9, 19, 12, 0, 0)
            first_id, first = create_unique_p3_run_directory(root, now)
            second_id, second = create_unique_p3_run_directory(root, now)
            self.assertNotEqual(first_id, second_id)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main()
