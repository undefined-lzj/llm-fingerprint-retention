import unittest

from transformers import GenerationConfig

from scripts.run_p3_1_b0 import clean_generation_config, generate_greedily


class FakeModel:
    def __init__(self):
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return "generated"


class P3GenerationTests(unittest.TestCase):
    def test_greedy_generation_disables_model_sampling_defaults(self):
        model = FakeModel()
        config = clean_generation_config(model, 16)

        result = generate_greedily(
            model,
            {"input_ids": "ids", "attention_mask": "mask"},
            config,
            pad_token_id=7,
        )

        self.assertEqual(result, "generated")
        self.assertFalse(config.do_sample)
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.top_p)
        self.assertIsNone(config.top_k)
        self.assertFalse(model.generate_kwargs["do_sample"])
        self.assertFalse(model.generate_kwargs["use_model_defaults"])
        self.assertEqual(model.generate_kwargs["generation_config"], config)
        self.assertNotIn("temperature", model.generate_kwargs)
        self.assertNotIn("top_p", model.generate_kwargs)
        self.assertNotIn("top_k", model.generate_kwargs)


if __name__ == "__main__":
    unittest.main()
