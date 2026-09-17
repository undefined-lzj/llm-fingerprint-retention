from __future__ import annotations

import unittest

from fingerprint.parser import normalize_output, parse_response


ALLOWED = ["NOVA-17", "LYNX-42"]


class StrictResponseParserTests(unittest.TestCase):
    def test_required_cases(self) -> None:
        cases = [
            ("NOVA-17", "NOVA-17", "exact_match"),
            ("LYNX-42", "NOVA-17", "wrong_valid_code"),
            ("  NOVA-17\n", "NOVA-17", "exact_match"),
            ("答案是NOVA-17", "NOVA-17", "invalid_output"),
            ("`NOVA-17`", "NOVA-17", "invalid_output"),
            ('"NOVA-17"', "NOVA-17", "invalid_output"),
            ("nova-17", "NOVA-17", "invalid_output"),
            ("", "NOVA-17", "invalid_output"),
            ("我不知道", "NOVA-17", "invalid_output"),
        ]
        for raw_output, target, expected in cases:
            with self.subTest(raw_output=raw_output, target=target):
                result = parse_response(raw_output, target, ALLOWED)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.is_exact_match, expected == "exact_match")

    def test_nfkc_normalization(self) -> None:
        self.assertEqual(normalize_output("ＮＯＶＡ－１７"), "NOVA-17")
        result = parse_response("ＮＯＶＡ－１７", "NOVA-17", ALLOWED)
        self.assertEqual(result.status, "exact_match")

    def test_does_not_remove_internal_whitespace(self) -> None:
        result = parse_response("NOVA -17", "NOVA-17", ALLOWED)
        self.assertEqual(result.status, "invalid_output")

    def test_target_must_be_allowed(self) -> None:
        with self.assertRaises(ValueError):
            parse_response("OTHER", "OTHER", ALLOWED)

    def test_allowed_responses_are_not_hard_coded(self) -> None:
        result = parse_response("PUBLIC-CODE", "PUBLIC-CODE", ["PUBLIC-CODE", "OTHER-CODE"])
        self.assertEqual(result.status, "exact_match")


if __name__ == "__main__":
    unittest.main()
