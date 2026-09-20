import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fingerprint.p3 import dolly_content_sha256, load_p3_fingerprint_manifest
from fingerprint.p33 import (
    EVALUATION_STEPS,
    audit_dolly_splits,
    build_unseen_training_plan,
    build_unseen_training_records,
    compute_attack_fairness_audit,
    compute_capability_curve,
    compute_g2_precheck,
    compute_retention_outputs,
    classify_p33r_result,
    create_unique_p33_run_directory,
    create_unique_p33r_run_directory,
    training_plan_sha256,
    training_records_sha256,
    validate_p33_config,
    validate_p33r_config,
    validate_unseen_training_plan,
)
from scripts.run_p3_3_unseen import (
    checkpoint_is_complete,
    latest_complete_checkpoint,
    validate_parent_p3_2,
    validate_parent_p3_3,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "p3_3_unseen.json"
ADJUSTED_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "p3_3_unseen_lr5e4.json"
)
ORIGINAL_P33_RUN = (
    PROJECT_ROOT / "runs" / "p3_3_unseen_20260919_223521_863957a1"
)
FINGERPRINT_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.json"
)
FINGERPRINT_SHA_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256"
)


def synthetic_split(start: int, count: int, name: str):
    rows = []
    for offset in range(count):
        index = start + offset
        instruction = f"Summarize synthetic document {name} number {index}."
        context = f"Synthetic context {name} {index}."
        response = f"Synthetic response {name} {index}."
        rows.append(
            {
                "original_index": index,
                "instruction": instruction,
                "context": context,
                "response": response,
                "prompt": f"{instruction}\n\nContext:\n{context}",
                "category": name,
                "content_sha256": dolly_content_sha256(
                    instruction, context, response
                ),
            }
        )
    return rows


def evaluation_records(fingerprint_ids, retained_ids):
    records = []
    for repeat_id in (1, 2):
        for fingerprint_id in fingerprint_ids:
            retained = fingerprint_id in retained_ids
            records.append(
                {
                    "fingerprint_id": fingerprint_id,
                    "repeat_id": repeat_id,
                    "raw_output": "ABCD-12" if retained else "unknown",
                    "parse_status": "exact_match" if retained else "invalid_output",
                    "is_exact_match": retained,
                    "contains_thinking_tag": False,
                }
            )
    return records


class P33UnseenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.adjusted_config = json.loads(
            ADJUSTED_CONFIG_PATH.read_text(encoding="utf-8")
        )
        cls.manifest = load_p3_fingerprint_manifest(
            FINGERPRINT_PATH, FINGERPRINT_SHA_PATH
        )
        cls.fingerprint_ids = [
            row["fingerprint_id"] for row in cls.manifest["fingerprints"]
        ]

    def test_fixed_config_is_valid(self):
        validate_p33_config(self.config)

    def test_p33r_config_changes_only_learning_rate(self):
        diff = validate_p33r_config(self.adjusted_config, self.config)
        self.assertTrue(diff["config_audit_passed"])
        self.assertTrue(diff["only_experimental_change_is_learning_rate"])
        self.assertEqual(
            diff["experimental_parameter_changes"],
            {
                "learning_rate": {
                    "original": 2e-4,
                    "adjusted": 5e-4,
                }
            },
        )

    def test_p33r_config_rejects_a_second_experimental_change(self):
        changed = dict(self.adjusted_config)
        changed["max_steps"] = 751
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            validate_p33r_config(changed, self.config)

    def test_downloaded_original_p33_is_the_fixed_weak_attack_parent(self):
        result = validate_parent_p3_3(ORIGINAL_P33_RUN, self.config)
        self.assertEqual(
            result["g2"]["g2_precheck"], "inconclusive_attack_too_weak"
        )
        self.assertEqual(
            result["comparison"]["checkpoint_exact_counts"]["750"],
            {"b1": 32, "p": 32},
        )
        self.assertEqual(
            result["order_info"]["training_order_sha256"],
            "a2e700820de24a8d9c70d7211d73b413f78433b3a98b03cef924713c7afc9810",
        )

    def test_downloaded_fixed_p3_2_parent_metadata_is_valid(self):
        parent = (
            PROJECT_ROOT
            / "runs"
            / "p3_2_feedback_20260919_175034_9735564b"
        )
        context = validate_parent_p3_2(
            parent, self.config, require_model_artifacts=False
        )
        self.assertTrue(context["comparison"]["p3_2_passed"])
        self.assertEqual(context["p3_1_path"].name, "p3_1_b0_20260919_161713_67cacee1")
        self.assertEqual(set(context["weights"]), set(self.fingerprint_ids))
        adjusted_context = validate_parent_p3_2(
            parent, self.adjusted_config, require_model_artifacts=False
        )
        self.assertEqual(
            adjusted_context["resolved"]["config"]["learning_rate"], 2e-4
        )

    def test_four_frozen_splits_are_disjoint_and_unseen_is_clean(self):
        splits = {
            "normal_train": synthetic_split(0, 1000, "normal"),
            "proxy_reserved": synthetic_split(1000, 500, "proxy"),
            "unseen_reserved": synthetic_split(1500, 1000, "unseen"),
            "capability_eval": synthetic_split(2500, 100, "capability"),
        }
        audit = audit_dolly_splits(splits, self.manifest)
        self.assertTrue(audit["audit_passed"])
        self.assertTrue(audit["all_pairwise_intersections_zero"])
        self.assertTrue(audit["unseen_fingerprint_contamination_free"])
        self.assertEqual(audit["training_split"], "unseen_reserved")
        self.assertEqual(audit["fingerprint_training_example_count"], 0)

        splits["unseen_reserved"][0] = dict(splits["normal_train"][0])
        overlapped = audit_dolly_splits(splits, self.manifest)
        self.assertFalse(overlapped["audit_passed"])
        self.assertFalse(overlapped["all_pairwise_intersections_zero"])

    def test_unseen_order_is_stable_three_epochs_and_750_steps(self):
        unseen = synthetic_split(1500, 1000, "unseen")
        records = build_unseen_training_records(unseen)
        first, first_sha = build_unseen_training_plan(
            records, seed=161803, max_steps=750, effective_batch_size=4
        )
        second, second_sha = build_unseen_training_plan(
            records, seed=161803, max_steps=750, effective_batch_size=4
        )
        self.assertEqual(first, second)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first_sha, training_plan_sha256(first))
        self.assertEqual(len(first), 3000)
        self.assertEqual(first[-1]["optimizer_step"], 750)
        validate_unseen_training_plan(
            first, records, expected_sha256=first_sha
        )
        self.assertEqual(len(training_records_sha256(records)), 64)

    def test_retention_curve_and_pairing_keep_each_repeat_denominator_at_32(self):
        evaluations = {"b1": {}, "p": {}}
        for step in EVALUATION_STEPS:
            if step == 0:
                b1_ids = set(self.fingerprint_ids)
                p_ids = set(self.fingerprint_ids)
            elif step == 125:
                b1_ids = set(self.fingerprint_ids[:20])
                p_ids = set(self.fingerprint_ids[:24])
            elif step == 375:
                b1_ids = set(self.fingerprint_ids[:12])
                p_ids = set(self.fingerprint_ids[:16])
            else:
                b1_ids = set(self.fingerprint_ids[:8])
                p_ids = set(self.fingerprint_ids[:10])
            evaluations["b1"][step] = evaluation_records(
                self.fingerprint_ids, b1_ids
            )
            evaluations["p"][step] = evaluation_records(
                self.fingerprint_ids, p_ids
            )
        weights = {
            fingerprint_id: 1.0 + index / 100
            for index, fingerprint_id in enumerate(self.fingerprint_ids)
        }
        curve, paired, details = compute_retention_outputs(
            evaluations, self.fingerprint_ids, weights
        )
        by_step = {row["checkpoint_step"]: row for row in curve}
        self.assertEqual(by_step[125]["b1_exact_count"], 20)
        self.assertEqual(by_step[125]["p_exact_count"], 24)
        self.assertEqual(by_step[125]["rate_difference_p_minus_b1"], 4 / 32)
        self.assertEqual(
            next(row for row in paired if row["checkpoint_step"] == 375)[
                "only_p_retained"
            ],
            4,
        )
        self.assertAlmostEqual(details["mean_retention_gap"], 10 / 96)

    def test_g2_classifies_promising_weak_strong_and_no_signal(self):
        def retention(counts_b1, counts_p):
            return [
                {
                    "checkpoint_step": step,
                    "b1_exact_count": b1,
                    "p_exact_count": p,
                }
                for step, b1, p in zip(
                    (0, 125, 375, 750), counts_b1, counts_p, strict=True
                )
            ]

        capability = compute_capability_curve(
            {"b1": {0: 2.0, 375: 2.1, 750: 2.2}, "p": {0: 2.0, 375: 2.1, 750: 2.2}}
        )
        promising_details = {
            "mean_retention_b1": 12 / 32,
            "mean_retention_p": 16 / 32,
            "mean_retention_gap": 4 / 32,
        }
        result = compute_g2_precheck(
            retention([32, 20, 12, 4], [32, 24, 16, 8]),
            promising_details,
            capability,
            step_750_smoke_passed=True,
            config=self.config,
        )
        self.assertEqual(result["g2_precheck"], "promising")

        full_details = {
            "mean_retention_b1": 1.0,
            "mean_retention_p": 1.0,
            "mean_retention_gap": 0.0,
        }
        weak = compute_g2_precheck(
            retention([32, 32, 32, 32], [32, 32, 32, 32]),
            full_details,
            capability,
            step_750_smoke_passed=True,
            config=self.config,
        )
        self.assertEqual(weak["g2_precheck"], "inconclusive_attack_too_weak")

        catastrophic = compute_capability_curve(
            {"b1": {0: 2.0, 375: 4.0, 750: 4.2}, "p": {0: 2.0, 375: 4.0, 750: 4.2}}
        )
        zero_details = {
            "mean_retention_b1": 0.0,
            "mean_retention_p": 0.0,
            "mean_retention_gap": 0.0,
        }
        strong = compute_g2_precheck(
            retention([32, 2, 0, 0], [32, 2, 0, 0]),
            zero_details,
            catastrophic,
            step_750_smoke_passed=False,
            config=self.config,
        )
        self.assertEqual(strong["g2_precheck"], "inconclusive_attack_too_strong")

        no_signal = compute_g2_precheck(
            retention([32, 20, 12, 4], [32, 20, 12, 4]),
            {
                "mean_retention_b1": 12 / 32,
                "mean_retention_p": 12 / 32,
                "mean_retention_gap": 0.0,
            },
            capability,
            step_750_smoke_passed=True,
            config=self.config,
        )
        self.assertEqual(no_signal["g2_precheck"], "no_positive_signal")

    def test_p33r_maps_terminal_classifications_without_changing_thresholds(self):
        self.assertEqual(classify_p33r_result("promising"), "promising")
        self.assertEqual(
            classify_p33r_result("no_positive_signal"), "no_positive_signal"
        )
        self.assertEqual(
            classify_p33r_result("inconclusive_attack_too_weak"),
            "inconclusive_after_allowed_adjustment",
        )
        self.assertEqual(
            classify_p33r_result("inconclusive_attack_too_strong"),
            "attack_too_strong_after_adjustment",
        )

    def test_fairness_audit_requires_same_initial_attack_state(self):
        summary = {
            "training_examples_sha256": "examples",
            "training_order_sha256": "order",
            "max_steps": 750,
            "checkpoint_steps": [125, 375, 750],
            "learning_rate": 2e-4,
            "effective_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "lora_config": {"r": 16, "alpha": 32},
            "seed": 314159,
            "initial_attack_adapter_state_sha256": "initial",
        }
        audit = compute_attack_fairness_audit(
            b1_summary=summary,
            p_summary=dict(summary),
            order_sha256="order",
            unseen_examples_sha256="examples",
            data_audit={"fingerprint_training_example_count": 0},
            config=self.config,
        )
        self.assertTrue(audit["all_checks_passed"])
        changed = dict(summary)
        changed["initial_attack_adapter_state_sha256"] = "different"
        failed = compute_attack_fairness_audit(
            b1_summary=summary,
            p_summary=changed,
            order_sha256="order",
            unseen_examples_sha256="examples",
            data_audit={"fingerprint_training_example_count": 0},
            config=self.config,
        )
        self.assertFalse(failed["same_initial_attack_adapter_state"])
        self.assertFalse(failed["all_checks_passed"])

    def test_result_directories_never_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = datetime(2026, 9, 19, 20, 30, 0)
            first_id, first = create_unique_p33_run_directory(root, fixed)
            second_id, second = create_unique_p33_run_directory(root, fixed)
            self.assertNotEqual(first_id, second_id)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            adjusted_first_id, adjusted_first = create_unique_p33r_run_directory(
                root, fixed
            )
            adjusted_second_id, adjusted_second = create_unique_p33r_run_directory(
                root, fixed
            )
            self.assertNotEqual(adjusted_first_id, adjusted_second_id)
            self.assertTrue(
                adjusted_first_id.startswith("p3_3r_unseen_lr5e4_20260919_203000_")
            )
            self.assertTrue(adjusted_first.is_dir())
            self.assertTrue(adjusted_second.is_dir())

    def test_resume_uses_latest_complete_checkpoint_and_archives_partial_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            branch = run_dir / "b1_attack"
            recovery = run_dir / "recovery"
            branch.mkdir()
            recovery.mkdir()
            complete = branch / "step_125"
            complete.mkdir()
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "training_state.pt",
            ):
                (complete / name).write_bytes(b"state")
            (complete / "training_state.json").write_text(
                json.dumps({"optimizer_step": 125, "method": "b1"}),
                encoding="utf-8",
            )
            partial = branch / "step_375"
            partial.mkdir()
            (partial / "adapter_config.json").write_text("{}", encoding="utf-8")
            step, path = latest_complete_checkpoint(branch, run_dir, "b1")
            self.assertEqual(step, 125)
            self.assertEqual(path, complete)
            self.assertTrue(checkpoint_is_complete(complete))
            self.assertFalse(partial.exists())
            self.assertEqual(len(list(recovery.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
