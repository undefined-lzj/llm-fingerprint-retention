from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_terminal_tee_writes_both_destinations(self) -> None:
        terminal = io.StringIO()
        log_file = io.StringIO()
        tee = MODULE.TeeStream(terminal, log_file)
        tee.write("完整终端输出\n")
        tee.flush()
        self.assertEqual(terminal.getvalue(), "完整终端输出\n")
        self.assertEqual(log_file.getvalue(), "完整终端输出\n")

    def test_main_creates_terminal_output_log(self) -> None:
        def fake_evaluation(*_args: object) -> int:
            print("stdout line")
            print("stderr line", file=sys.stderr)
            return 0

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            with (
                patch.object(MODULE, "parse_args", return_value=SimpleNamespace(repeats=2)),
                patch.object(
                    MODULE,
                    "create_run_directory",
                    return_value=("p1_test", run_dir, "2026-09-17T00:00:00+08:00"),
                ),
                patch.object(MODULE, "run_evaluation", side_effect=fake_evaluation),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(MODULE.main(), 0)

            logged = (run_dir / "terminal_output.log").read_text(encoding="utf-8")
            self.assertIn("stdout line", logged)
            self.assertIn("stderr line", logged)


if __name__ == "__main__":
    unittest.main()
