import math
import unittest

from fingerprint.p2 import (
    compute_evaluation_metrics,
    outputs_identical_between_models,
    summarize_training_losses,
)


def make_records(*, repeats=2, status="exact_match", identical=True):
    records = []
    for repeat_id in range(1, repeats + 1):
        for fingerprint_number in range(1, 9):
            target = "NOVA-17" if fingerprint_number % 2 else "LYNX-42"
            output = target if identical else f"{target}-{repeat_id}"
            records.append(
                {
                    "fingerprint_id": f"fp{fingerprint_number:02d}",
                    "repeat_id": repeat_id,
                    "raw_output": output,
                    "normalized_output": output,
                    "parse_status": status,
                    "is_exact_match": status == "exact_match",
                    "contains_thinking_tag": False,
                }
            )
    return records


class P2MetricsTests(unittest.TestCase):
    def test_complete_exact_records_pass(self):
        metrics = compute_evaluation_metrics(
            "p1_demo_v1",
            [f"fp{number:02d}" for number in range(1, 9)],
            make_records(),
            repeats=2,
        )
        self.assertEqual(metrics["completed_query_count_by_repeat"], {"1": 8, "2": 8})
        self.assertEqual(metrics["exact_match_count_by_repeat"], {"1": 8, "2": 8})
        self.assertEqual(metrics["wrong_valid_code_count_by_repeat"], {"1": 0, "2": 0})
        self.assertEqual(metrics["invalid_output_count_by_repeat"], {"1": 0, "2": 0})
        self.assertTrue(metrics["outputs_identical_across_repeats"])
        self.assertTrue(metrics["evaluation_passed"])

    def test_invalid_output_is_counted_and_fails(self):
        records = make_records()
        records[0]["raw_output"] = "我不知道"
        records[0]["normalized_output"] = "我不知道"
        records[0]["parse_status"] = "invalid_output"
        records[0]["is_exact_match"] = False

        metrics = compute_evaluation_metrics(
            "p1_demo_v1",
            [f"fp{number:02d}" for number in range(1, 9)],
            records,
            repeats=2,
        )
        self.assertEqual(metrics["completed_query_count_by_repeat"]["1"], 8)
        self.assertEqual(metrics["invalid_output_count_by_repeat"]["1"], 1)
        self.assertEqual(metrics["exact_match_rate_by_repeat"]["1"], 7 / 8)
        self.assertFalse(metrics["evaluation_passed"])

    def test_repeat_mismatch_is_detected(self):
        metrics = compute_evaluation_metrics(
            "p1_demo_v1",
            [f"fp{number:02d}" for number in range(1, 9)],
            make_records(identical=False),
            repeats=2,
        )
        self.assertFalse(metrics["outputs_identical_across_repeats"])
        self.assertFalse(metrics["evaluation_passed"])

    def test_adapter_and_merged_outputs_comparison(self):
        adapter = make_records()
        merged = make_records()
        self.assertTrue(outputs_identical_between_models(adapter, merged))

        merged[-1]["raw_output"] = "different"
        self.assertFalse(outputs_identical_between_models(adapter, merged))

    def test_training_loss_summary_detects_overall_decline(self):
        losses = [2.0 - step * 0.02 for step in range(1, 81)]
        summary = summarize_training_losses(losses, expected_steps=80)
        self.assertTrue(summary["all_losses_finite"])
        self.assertTrue(summary["loss_decreased"])
        self.assertEqual(summary["optimizer_step_count"], 80)

    def test_training_loss_summary_marks_nan_and_wrong_step_count_as_failed(self):
        nan_summary = summarize_training_losses([math.nan], expected_steps=1)
        self.assertFalse(nan_summary["all_losses_finite"])
        self.assertFalse(nan_summary["training_completed"])

        short_summary = summarize_training_losses([1.0], expected_steps=80)
        self.assertFalse(short_summary["training_completed"])
        self.assertEqual(short_summary["optimizer_step_count"], 1)


if __name__ == "__main__":
    unittest.main()
