import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fingerprint.p2 import (
    EXPECTED_TARGET_MODULES,
    build_training_examples,
    collate_training_examples,
    create_unique_run_directory,
    parameter_summary,
    resolve_target_modules,
    training_snapshot,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training" / "p2_qwen3_0_6b_lora.json"
)
FINGERPRINT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "fingerprints" / "p1_demo_fingerprints.json"
)


class FakeTokenizer:
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        if text == "NOVA-17":
            return [701, 702]
        if text == "LYNX-42":
            return [703, 704]
        return [800 + index for index, _ in enumerate(text)]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        if not tokenize:
            raise AssertionError("测试替身只支持 tokenize=True")
        if enable_thinking:
            raise AssertionError("P2 必须关闭思考模式")

        prompt = messages[0]["content"]
        prompt_ids = [11, len(prompt), 12, 13]
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("用户提示必须包含 assistant 生成前缀")
            return prompt_ids

        if add_generation_prompt:
            raise AssertionError("完整训练序列不应再添加生成提示")
        target_ids = self.encode(messages[1]["content"], add_special_tokens=False)
        return prompt_ids + target_ids + [99]


class FakeTensor:
    def __init__(self, data):
        self.data = data


def fake_tensor_factory(data):
    return FakeTensor(data)


class FakeModule:
    pass


class FakeParameter:
    def __init__(self, count, requires_grad):
        self._count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self._count


class FakeModel:
    def __init__(self, *, include_all_targets=True, unexpected_trainable=False):
        targets = EXPECTED_TARGET_MODULES
        if not include_all_targets:
            targets = targets[:-1]
        self._modules = [
            (f"model.layers.0.self_attn.{name}", FakeModule()) for name in targets
        ]
        self._parameters = [
            ("base_model.model.embed_tokens.weight", FakeParameter(900, False)),
            ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", FakeParameter(50, True)),
            ("base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight", FakeParameter(50, True)),
        ]
        if unexpected_trainable:
            self._parameters.append(("base_model.model.norm.weight", FakeParameter(10, True)))

    def named_modules(self):
        return iter(self._modules)

    def named_parameters(self):
        return iter(self._parameters)


class P2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_config = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.fingerprint_manifest = json.loads(
            FINGERPRINT_CONFIG_PATH.read_text(encoding="utf-8")
        )
        cls.tokenizer = FakeTokenizer()

    def test_fixed_training_config_is_valid_and_serializable(self):
        validate_training_config(self.training_config)
        serialized = json.dumps(self.training_config, ensure_ascii=False)
        self.assertEqual(json.loads(serialized), self.training_config)

    def test_all_eight_examples_have_assistant_only_supervision(self):
        examples = build_training_examples(
            self.tokenizer,
            self.fingerprint_manifest,
            max_sequence_length=self.training_config["max_sequence_length"],
        )
        self.assertEqual(len(examples), 8)

        by_id = {
            item["fingerprint_id"]: item
            for item in self.fingerprint_manifest["fingerprints"]
        }
        for example in examples:
            prompt_length = example["prompt_token_count"]
            self.assertTrue(all(label == -100 for label in example["labels"][:prompt_length]))
            supervised_labels = example["labels"][prompt_length:]
            self.assertTrue(supervised_labels)
            self.assertTrue(all(label != -100 for label in supervised_labels))

            target = by_id[example["fingerprint_id"]]["target_response"]
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            self.assertEqual(supervised_labels[: len(target_ids)], target_ids)
            self.assertEqual(supervised_labels[-1], 99)
            self.assertFalse(example["was_truncated"])

        snapshot = training_snapshot(examples)
        self.assertEqual(len(snapshot), 8)
        self.assertTrue(all(row["supervised_token_count"] > 0 for row in snapshot))

    def test_sequence_too_long_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "截断"):
            build_training_examples(
                self.tokenizer,
                self.fingerprint_manifest,
                max_sequence_length=4,
            )

    def test_collator_masks_padding_and_preserves_prompt_mask(self):
        examples = build_training_examples(
            self.tokenizer,
            self.fingerprint_manifest,
            max_sequence_length=self.training_config["max_sequence_length"],
        )
        shortened = dict(examples[0])
        shortened["input_ids"] = shortened["input_ids"][:-1]
        shortened["labels"] = shortened["labels"][:-1]

        batch = collate_training_examples(
            [shortened, examples[1]],
            pad_token_id=self.tokenizer.pad_token_id,
            tensor_factory=fake_tensor_factory,
        )
        self.assertEqual(batch["input_ids"].data[0][-1], self.tokenizer.pad_token_id)
        self.assertEqual(batch["attention_mask"].data[0][-1], 0)
        self.assertEqual(batch["labels"].data[0][-1], -100)
        self.assertTrue(
            all(
                label == -100
                for label in batch["labels"].data[1][
                    : examples[1]["prompt_token_count"]
                ]
            )
        )

    def test_lora_target_modules_are_resolved_strictly(self):
        resolved = resolve_target_modules(FakeModel(), EXPECTED_TARGET_MODULES)
        self.assertEqual(set(resolved), set(EXPECTED_TARGET_MODULES))
        self.assertTrue(all(resolved[name] for name in EXPECTED_TARGET_MODULES))

        with self.assertRaisesRegex(ValueError, "找不到 LoRA 目标模块"):
            resolve_target_modules(
                FakeModel(include_all_targets=False), EXPECTED_TARGET_MODULES
            )

    def test_only_lora_parameters_may_be_trainable(self):
        summary = parameter_summary(FakeModel())
        self.assertEqual(summary["total_parameter_count"], 1000)
        self.assertEqual(summary["trainable_parameter_count"], 100)
        self.assertTrue(summary["only_lora_parameters_trainable"])

        with self.assertRaisesRegex(ValueError, "非 LoRA"):
            parameter_summary(FakeModel(unexpected_trainable=True))

    def test_unique_result_directory_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checked_at = datetime(2026, 9, 17, 12, 0, 0)
            first_id, first = create_unique_run_directory(root, checked_at)
            second_id, second = create_unique_run_directory(root, checked_at)
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main()
