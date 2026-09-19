import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import torch

from fingerprint.p3 import load_p3_fingerprint_manifest
from fingerprint.p32 import (
    build_branch_training_records,
    build_continuation_training_plan,
    build_target_loss_example,
    compute_capability_comparison,
    compute_fairness_audit,
    compute_forgetting_feedback,
    create_unique_p32_run_directory,
    training_plan_sha256,
    validate_continuation_training_plan,
    validate_p32_config,
    weighted_completion_loss,
)
from scripts.run_p3_2_feedback import validate_parent_p3_1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "p3_2_feedback.json"
FINGERPRINT_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.json"
)
FINGERPRINT_SHA_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256"
)


class FakeTokenizer:
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [500 + index for index, _ in enumerate(text[:3])]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        if not tokenize or enable_thinking:
            raise AssertionError("Tokenizer参数错误")
        prompt_ids = [11, len(messages[0]["content"]), 12, 13]
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("prompt必须包含生成边界")
            return prompt_ids
        if add_generation_prompt:
            raise AssertionError("完整序列不能添加生成边界")
        return prompt_ids + self.encode(messages[1]["content"]) + [99]


def normal_records():
    return [
        {
            "original_index": index,
            "content_sha256": f"{index:064x}",
            "prompt": f"normal prompt {index}",
            "response": f"normal response {index}",
        }
        for index in range(1000)
    ]


class P32FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.manifest = load_p3_fingerprint_manifest(
            FINGERPRINT_PATH, FINGERPRINT_SHA_PATH
        )

    def test_fixed_config_is_valid(self):
        validate_p32_config(self.config)

    def test_downloaded_p3_1_parent_metadata_is_valid(self):
        parent = PROJECT_ROOT / "runs" / "p3_1_b0_20260919_161713_67cacee1"
        result = validate_parent_p3_1(
            parent, self.config, require_model_artifacts=False
        )
        self.assertEqual(result["resolved"]["run_id"], parent.name)
        self.assertTrue(result["comparison"]["p3_1_passed"])
        self.assertEqual(
            result["dolly_manifest"]["splits"]["proxy_reserved"]["count"], 500
        )

    def test_target_loss_labels_only_cover_target_code(self):
        fingerprint = self.manifest["fingerprints"][0]
        example = build_target_loss_example(
            FakeTokenizer(), fingerprint, max_sequence_length=512
        )
        active = [value for value in example["labels"] if value != -100]
        self.assertEqual(active, example["target_token_ids"])
        self.assertEqual(len(active), example["target_token_count"])
        self.assertTrue(
            all(
                value == -100
                for value in example["labels"][: example["target_token_start"]]
            )
        )
        self.assertTrue(
            all(
                value == -100
                for value in example["labels"][
                    example["target_token_start"] + example["target_token_count"] :
                ]
            )
        )

    def test_weighted_loss_is_per_example_and_masks_user_and_padding(self):
        logits = torch.zeros((2, 4, 3), dtype=torch.float32)
        logits[0, 1, 0] = 2.0
        logits[0, 2, 1] = 1.0
        logits[1, 0, 2] = 3.0
        labels = torch.tensor(
            [[-100, -100, 0, 1], [-100, 2, -100, -100]], dtype=torch.long
        )
        unit_weights = torch.ones(2)
        ordinary, per_example, token_counts = weighted_completion_loss(
            torch, logits, labels, unit_weights
        )
        self.assertTrue(torch.allclose(ordinary, per_example.mean()))
        self.assertEqual(token_counts.tolist(), [2, 1])

        emphasized, emphasized_per_example, _ = weighted_completion_loss(
            torch, logits, labels, torch.tensor([2.0, 1.0])
        )
        self.assertTrue(torch.allclose(per_example, emphasized_per_example))
        self.assertTrue(
            torch.allclose(emphasized, (2.0 * per_example[0] + per_example[1]) / 2)
        )

        masked_changed = logits.clone()
        masked_changed[0, 0, :] = torch.tensor([100.0, -100.0, -100.0])
        masked_changed[1, 1:, :] = 77.0
        masked_loss, _, _ = weighted_completion_loss(
            torch, masked_changed, labels, unit_weights
        )
        self.assertTrue(torch.allclose(ordinary, masked_loss))

    def test_feedback_weights_are_normalized_and_non_degenerate(self):
        before = {}
        after = {}
        for index, fingerprint in enumerate(self.manifest["fingerprints"]):
            fingerprint_id = fingerprint["fingerprint_id"]
            before[fingerprint_id] = {"loss": 1.0, "target_token_count": 3}
            delta = 0.001 * (index + 1) if index < 20 else -0.001
            after[fingerprint_id] = {
                "loss": 1.0 + delta,
                "target_token_count": 3,
            }
        scores, weights, stats = compute_forgetting_feedback(
            before, after, config=self.config
        )
        self.assertEqual(len(scores), 32)
        self.assertEqual(set(weights), {row["fingerprint_id"] for row in scores})
        self.assertAlmostEqual(sum(weights.values()) / 32, 1.0, places=12)
        self.assertGreater(stats["weight_std"], 0.01)
        self.assertTrue(stats["feedback_valid"])

        degenerate_after = {
            key: {"loss": 1.0, "target_token_count": 3} for key in before
        }
        _scores, _weights, degenerate = compute_forgetting_feedback(
            before, degenerate_after, config=self.config
        )
        self.assertFalse(degenerate["feedback_valid"])

    def test_b1_and_p_use_exactly_the_same_saved_order(self):
        unit = {
            row["fingerprint_id"]: 1.0 for row in self.manifest["fingerprints"]
        }
        weighted = {
            row["fingerprint_id"]: 0.75 + index / 64
            for index, row in enumerate(self.manifest["fingerprints"])
        }
        mean = sum(weighted.values()) / len(weighted)
        weighted = {key: value / mean for key, value in weighted.items()}
        b1_records = build_branch_training_records(
            normal_records(),
            self.manifest,
            fingerprint_repeat=8,
            fingerprint_weights=unit,
        )
        p_records = build_branch_training_records(
            normal_records(),
            self.manifest,
            fingerprint_repeat=8,
            fingerprint_weights=weighted,
        )
        self.assertEqual(
            [row["sample_id"] for row in b1_records],
            [row["sample_id"] for row in p_records],
        )
        plan, digest = build_continuation_training_plan(
            b1_records, seed=271828, max_steps=80, effective_batch_size=8
        )
        self.assertEqual(len(plan), 640)
        self.assertEqual(digest, training_plan_sha256(plan))
        validate_continuation_training_plan(
            plan,
            p_records,
            expected_sha256=digest,
            max_steps=80,
            effective_batch_size=8,
        )

    def test_fairness_audit_accepts_only_weight_mapping_difference(self):
        unit = {
            row["fingerprint_id"]: 1.0 for row in self.manifest["fingerprints"]
        }
        p_weights = dict(unit)
        p_weights["p3fp001"] = 1.2
        p_weights["p3fp002"] = 0.8
        summary = {
            "max_steps": 80,
            "effective_batch_size": 8,
            "learning_rate": 2e-4,
            "lora_config": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
            },
            "normal_example_weight": 1.0,
            "fingerprint_repeat": 8,
        }
        audit = compute_fairness_audit(
            parent_run_id_b1="parent",
            parent_run_id_p="parent",
            revision_b1=self.config["revision"],
            revision_p=self.config["revision"],
            training_example_sha_b1="examples",
            training_example_sha_p="examples",
            order_sha_b1="order",
            order_sha_p="order",
            b1_summary=summary,
            p_summary=dict(summary),
            b1_weights=unit,
            p_weights=p_weights,
        )
        self.assertTrue(audit["all_checks_passed"])
        self.assertEqual(audit["only_intended_difference"], "fingerprint_weight_mapping")

    def test_capability_comparison_enforces_five_percent_limit(self):
        b1 = [{"raw_output": f"normal b1 {index}"} for index in range(10)]
        p = [{"raw_output": f"normal p {index}"} for index in range(10)]
        passed = compute_capability_comparison(
            b1_loss=2.0,
            p_loss=2.09,
            b1_generations=b1,
            p_generations=p,
            known_codes=set(self.manifest["allowed_responses"]),
            maximum_relative_increase=0.05,
        )
        self.assertTrue(passed["capability_comparison_passed"])
        failed = compute_capability_comparison(
            b1_loss=2.0,
            p_loss=2.11,
            b1_generations=b1,
            p_generations=p,
            known_codes=set(self.manifest["allowed_responses"]),
            maximum_relative_increase=0.05,
        )
        self.assertFalse(failed["capability_comparison_passed"])

    def test_result_directories_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            now = datetime(2026, 9, 19, 17, 0, 0)
            first_id, first = create_unique_p32_run_directory(root, now)
            second_id, second = create_unique_p32_run_directory(root, now)
            self.assertNotEqual(first_id, second_id)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main()
