import json
import unittest
from collections import Counter
from pathlib import Path

from fingerprint.p3 import (
    P3_TEMPLATE_GROUPS,
    load_p3_fingerprint_manifest,
    sha256_file,
    tokenize_fingerprint_targets,
    validate_b0_training_config,
)
from fingerprint.parser import parse_response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.json"
)
FINGERPRINT_SHA_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256"
)
TRAINING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "p3_1_qwen3_0_6b_b0.json"
)


class FakeTokenizer:
    unk_token_id = 999999

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        letters, number = text.split("-", 1)
        return [len(letters), 45, int(number)]

    @staticmethod
    def convert_ids_to_tokens(token_ids):
        return [f"tok-{token_id}" for token_id in token_ids]


class P3FingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_p3_fingerprint_manifest(
            FINGERPRINT_PATH, FINGERPRINT_SHA_PATH
        )

    def test_frozen_file_sha256_matches_sidecar(self):
        expected = FINGERPRINT_SHA_PATH.read_text(encoding="utf-8").split()[0]
        self.assertEqual(sha256_file(FINGERPRINT_PATH), expected)

    def test_fixed_32_fingerprints_are_unique_and_balanced(self):
        records = self.manifest["fingerprints"]
        self.assertEqual(len(records), 32)
        self.assertEqual(
            [record["fingerprint_id"] for record in records],
            [f"p3fp{index:03d}" for index in range(1, 33)],
        )
        self.assertEqual(len({record["prompt"] for record in records}), 32)
        self.assertEqual(len({record["record_id"] for record in records}), 32)
        self.assertEqual(len({record["target_response"] for record in records}), 32)
        self.assertEqual(
            Counter(record["template_group"] for record in records),
            Counter({group: 8 for group in P3_TEMPLATE_GROUPS}),
        )
        self.assertNotIn("NOVA-17", self.manifest["allowed_responses"])
        self.assertNotIn("LYNX-42", self.manifest["allowed_responses"])

    def test_target_tokenization_is_dynamic_and_within_limit(self):
        rows = tokenize_fingerprint_targets(FakeTokenizer(), self.manifest)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["token_count"] <= 5 for row in rows))
        self.assertTrue(all(not row["contains_unknown_token"] for row in rows))

    def test_strict_parser_uses_all_32_dynamic_codes(self):
        allowed = self.manifest["allowed_responses"]
        exact = parse_response(allowed[0], allowed[0], allowed)
        wrong = parse_response(allowed[1], allowed[0], allowed)
        invalid = parse_response(f"答案是{allowed[0]}", allowed[0], allowed)
        self.assertEqual(exact.status, "exact_match")
        self.assertEqual(wrong.status, "wrong_valid_code")
        self.assertEqual(invalid.status, "invalid_output")

    def test_b0_training_config_is_fixed(self):
        config = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
        validate_b0_training_config(config)
        self.assertEqual(config["num_train_epochs"], 3)
        self.assertEqual(config["fingerprint_repeat"], 8)
        self.assertEqual(config["max_seq_length"], 512)


if __name__ == "__main__":
    unittest.main()
