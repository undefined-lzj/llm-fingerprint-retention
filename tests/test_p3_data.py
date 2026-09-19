import unittest
from pathlib import Path

from fingerprint.p2 import collate_training_examples
from fingerprint.p3 import (
    DOLLY_SPLIT_COUNTS,
    build_completion_example,
    build_training_order,
    build_training_records,
    load_p3_fingerprint_manifest,
    prepare_dolly_records,
    split_dolly_records,
    validate_split_intersections,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        return [700 + index for index, _ in enumerate(text[:3])]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        if not tokenize or enable_thinking:
            raise AssertionError("测试Tokenizer调用参数不正确")
        prompt = messages[0]["content"]
        prompt_ids = [11, len(prompt), 12, 13]
        if len(messages) == 1:
            self.assert_generation_prompt(add_generation_prompt)
            return prompt_ids
        if add_generation_prompt:
            raise AssertionError("完整训练序列不应添加生成提示")
        return prompt_ids + self.encode(messages[1]["content"]) + [99]

    @staticmethod
    def assert_generation_prompt(value):
        if value is not True:
            raise AssertionError("用户边界必须包含assistant生成提示")


class P3DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FakeTokenizer()
        cls.fingerprint_manifest = load_p3_fingerprint_manifest(
            FINGERPRINT_PATH, FINGERPRINT_SHA_PATH
        )

    def test_completion_only_mask_and_padding(self):
        first = build_completion_example(
            self.tokenizer,
            sample_id="first",
            sample_type="normal",
            prompt="Do one thing.",
            response="Answer one.",
            max_sequence_length=512,
        )
        second = build_completion_example(
            self.tokenizer,
            sample_id="second",
            sample_type="normal",
            prompt="Do another thing.",
            response="Answer two.",
            max_sequence_length=512,
        )
        second["input_ids"] = second["input_ids"][:-1]
        second["labels"] = second["labels"][:-1]
        second["attention_mask"] = second["attention_mask"][:-1]
        self.assertTrue(
            all(label == -100 for label in first["labels"][: first["prompt_token_count"]])
        )
        self.assertTrue(
            all(label != -100 for label in first["labels"][first["prompt_token_count"] :])
        )
        batch = collate_training_examples(
            [first, second], pad_token_id=self.tokenizer.pad_token_id
        )
        self.assertEqual(batch["attention_mask"][1][-1], 0)
        self.assertEqual(batch["labels"][1][-1], -100)

    def test_dolly_filtering_deduplicates_and_rejects_empty_response(self):
        rows = [
            {
                "instruction": "Question A",
                "context": "Context A",
                "response": "Response A",
                "category": "open_qa",
            },
            {
                "instruction": "Question A",
                "context": "Context A",
                "response": "Response A",
                "category": "open_qa",
            },
            {
                "instruction": "Question B",
                "context": "",
                "response": "",
                "category": "classification",
            },
            {
                "instruction": "Question C",
                "context": "",
                "response": "Response C",
                "category": "classification",
            },
        ]
        accepted, stats = prepare_dolly_records(rows, self.tokenizer, 512)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(stats["duplicate_content_count"], 1)
        self.assertEqual(stats["empty_response_count"], 1)

    def test_fixed_split_is_deterministic_and_has_no_intersection(self):
        eligible = []
        categories = ["brainstorming", "classification", "closed_qa", "generation"]
        for index in range(2800):
            eligible.append(
                {
                    "original_index": index,
                    "content_sha256": f"{index:064x}",
                    "category": categories[index % len(categories)],
                    "instruction": f"instruction {index}",
                    "context": "",
                    "response": f"response {index}",
                    "prompt": f"instruction {index}",
                    "input_token_count": 20,
                    "supervised_token_count": 4,
                }
            )
        first = split_dolly_records(eligible, seed=20260918)
        second = split_dolly_records(eligible, seed=20260918)
        self.assertEqual(first, second)
        intersections = validate_split_intersections(first)
        self.assertTrue(all(value == 0 for value in intersections.values()))
        self.assertEqual(
            {name: len(records) for name, records in first.items()},
            DOLLY_SPLIT_COUNTS,
        )

    def test_b0_mixture_has_1256_records_and_256_fingerprint_repeats(self):
        normal = [
            {
                "original_index": index,
                "content_sha256": f"{index:064x}",
                "prompt": f"prompt {index}",
                "response": f"response {index}",
            }
            for index in range(1000)
        ]
        records = build_training_records(normal, self.fingerprint_manifest, 8)
        self.assertEqual(len(records), 1256)
        self.assertEqual(sum(row["sample_type"] == "normal" for row in records), 1000)
        self.assertEqual(
            sum(row["sample_type"] == "fingerprint" for row in records), 256
        )
        plan, first_hash = build_training_order(records, epochs=3, seed=20260918)
        plan_again, second_hash = build_training_order(
            records, epochs=3, seed=20260918
        )
        self.assertEqual(len(plan), 3768)
        self.assertEqual(plan, plan_again)
        self.assertEqual(first_hash, second_hash)


if __name__ == "__main__":
    unittest.main()
