#!/usr/bin/env python3
"""准备并冻结P3-1开发指纹Token检查与Dolly四路数据划分。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (SCRIPT_DIR, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from fingerprint.p3 import (  # noqa: E402
    DOLLY_SPLIT_COUNTS,
    category_distribution,
    load_p3_fingerprint_manifest,
    prepare_dolly_records,
    records_sha256,
    sha256_bytes,
    sha256_file,
    split_dolly_records,
    tokenize_fingerprint_targets,
    validate_b0_training_config,
    validate_dolly_split_manifest,
    validate_split_intersections,
    write_jsonl_text,
)
from p0b_common import sanitize_text, write_json, write_text  # noqa: E402


SCRIPT_VERSION = "p3-1-data-1.0.0"
DATA_ROOT = Path("/root/autodl-tmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training" / "p3_1_qwen3_0_6b_b0.json",
    )
    parser.add_argument(
        "--fingerprint-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.json",
    )
    parser.add_argument(
        "--fingerprint-sha",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fingerprints" / "p3_dev_fingerprints.sha256",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "data_manifests" / "p3_dolly_split_manifest.json",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/root/autodl-tmp/b-plan/data/p3_1_dolly_v1"),
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=Path("/root/autodl-tmp/b-plan/cache/huggingface/datasets"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}顶层必须是JSON对象")
    return value


def latest_successful_p2() -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in (PROJECT_ROOT / "runs").glob("p2_lora_*"):
        comparison_path = run_dir / "comparison.json"
        resolved_path = run_dir / "resolved_config.json"
        if not comparison_path.is_file() or not resolved_path.is_file():
            continue
        try:
            comparison = load_json(comparison_path)
            resolved = load_json(resolved_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if comparison.get("p2_passed") is True and resolved.get("status") == "completed":
            candidates.append((run_dir.resolve(), resolved))
    if not candidates:
        raise RuntimeError("找不到通过验收的P2运行，不能准备P3-1数据")
    return max(candidates, key=lambda item: item[0].name)


def validate_p2_model_reference(
    run_dir: Path,
    resolved: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    expected = {
        "model_id": config["model_id"],
        "revision": config["revision"],
        "dtype": config["dtype"],
    }
    for field, value in expected.items():
        if resolved.get(field) != value:
            raise RuntimeError(f"P2运行{run_dir.name}的{field}与P3-1不一致")
    if resolved.get("local_files_only") is not True:
        raise RuntimeError("P2没有记录local_files_only=true")
    snapshot = Path(str(resolved.get("snapshot_path"))).expanduser().resolve()
    if not snapshot.is_relative_to(DATA_ROOT) or not snapshot.is_dir():
        raise RuntimeError("P2锁定的Qwen本地快照不存在或不在/root/autodl-tmp/")
    if snapshot.name != config["revision"]:
        raise RuntimeError("P2快照目录名与锁定revision不一致")
    return snapshot


def validate_existing_manifest(
    manifest_path: Path,
    fingerprint_sha256: str,
) -> int:
    manifest = load_json(manifest_path)
    validate_dolly_split_manifest(manifest, require_files=True)
    if manifest.get("fingerprint_manifest_sha256") != fingerprint_sha256:
        raise RuntimeError("现有Dolly冻结划分引用了不同的P3指纹文件")
    print("P3-1 Dolly冻结划分已经存在且校验通过；未重新下载或抽样。")
    print(f"数据清单：{manifest_path}")
    print(f"数据revision：{manifest['dataset_revision']}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        config_path = args.training_config.expanduser().resolve()
        fingerprint_path = args.fingerprint_config.expanduser().resolve()
        fingerprint_sha_path = args.fingerprint_sha.expanduser().resolve()
        manifest_path = args.manifest_output.expanduser().resolve()
        processed_dir = args.processed_dir.expanduser().resolve()
        dataset_cache = args.dataset_cache.expanduser().resolve()

        config = load_json(config_path)
        validate_b0_training_config(config)
        fingerprint_manifest = load_p3_fingerprint_manifest(
            fingerprint_path, fingerprint_sha_path
        )
        fingerprint_sha256 = sha256_file(fingerprint_path)

        if manifest_path.is_file():
            return validate_existing_manifest(manifest_path, fingerprint_sha256)
        if not PROJECT_ROOT.resolve().is_relative_to(DATA_ROOT):
            raise RuntimeError("正式P3-1数据准备必须在/root/autodl-tmp/内的项目执行")
        if not processed_dir.is_relative_to(DATA_ROOT) or not dataset_cache.is_relative_to(
            DATA_ROOT
        ):
            raise RuntimeError("Dolly处理数据和缓存必须位于/root/autodl-tmp/")
        if processed_dir.exists():
            raise RuntimeError(
                f"处理目录已存在但冻结清单不存在，拒绝覆盖：{processed_dir}"
            )
        minimum_bytes = int(config["minimum_free_disk_gib"] * 1024**3)
        if shutil.disk_usage(PROJECT_ROOT).free < minimum_bytes:
            raise RuntimeError("数据盘剩余空间不足配置要求")

        p2_run_dir, p2_resolved = latest_successful_p2()
        snapshot = validate_p2_model_reference(p2_run_dir, p2_resolved, config)

        from datasets import load_dataset
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            trust_remote_code=False,
            local_files_only=True,
        )
        target_tokenization = tokenize_fingerprint_targets(
            tokenizer, fingerprint_manifest, maximum_tokens=5
        )
        print("32个P3目标代号已通过实际Qwen Tokenizer检查。")

        dataset_cache.mkdir(parents=True, exist_ok=True)
        info = HfApi().dataset_info(config["dataset_id"], revision="main")
        revision = str(info.sha)
        if not revision or len(revision) != 40:
            raise RuntimeError("Hugging Face没有返回准确的Dolly dataset revision")
        card_license = str(getattr(info, "card_data", {}).get("license", ""))
        if card_license.lower() != "cc-by-sa-3.0":
            raise RuntimeError(f"Dolly许可证与预期不一致：{card_license}")

        dataset = load_dataset(
            config["dataset_id"],
            split="train",
            revision=revision,
            cache_dir=str(dataset_cache),
        )
        eligible, filter_stats = prepare_dolly_records(
            dataset,
            tokenizer,
            max_sequence_length=config["max_seq_length"],
        )
        splits = split_dolly_records(
            eligible,
            seed=config["dataset_split_seed"],
            split_counts=config["dolly_split_counts"],
        )
        intersections = validate_split_intersections(
            splits, config["dolly_split_counts"]
        )

        temporary_dir = processed_dir.with_name(
            f".{processed_dir.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary_dir.mkdir(parents=True, exist_ok=False)
        split_manifest: dict[str, Any] = {}
        try:
            for split_name in DOLLY_SPLIT_COUNTS:
                temporary_path = temporary_dir / f"{split_name}.jsonl"
                write_text(temporary_path, write_jsonl_text(splits[split_name]))
            os.replace(temporary_dir, processed_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        for split_name, expected_count in DOLLY_SPLIT_COUNTS.items():
            final_path = processed_dir / f"{split_name}.jsonl"
            records = splits[split_name]
            split_manifest[split_name] = {
                "count": expected_count,
                "file_path": str(final_path),
                "file_sha256": sha256_file(final_path),
                "records_sha256": records_sha256(records),
                "category_distribution": category_distribution(records),
                "minimum_input_token_count": min(
                    record["input_token_count"] for record in records
                ),
                "maximum_input_token_count": max(
                    record["input_token_count"] for record in records
                ),
            }

        selected_hashes = sorted(
            record["content_sha256"]
            for name in DOLLY_SPLIT_COUNTS
            for record in splits[name]
        )
        manifest = {
            "schema_version": 1,
            "stage_id": "P3-1",
            "experiment_tier": "pilot",
            "script_version": SCRIPT_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset_id": config["dataset_id"],
            "dataset_revision": revision,
            "license": config["dataset_license"],
            "dataset_cache_path": str(dataset_cache),
            "processed_data_directory": str(processed_dir),
            "source_row_count": len(dataset),
            "split_seed": config["dataset_split_seed"],
            "selection_method": "NFKC-strip triple dedup, Qwen chat length filter, category round-robin, seeded content ordering",
            "deduplication_key": "SHA256 of normalized instruction/context/response triple",
            "maximum_sequence_length": config["max_seq_length"],
            "fingerprint_manifest_path": str(
                fingerprint_path.relative_to(PROJECT_ROOT.resolve())
            ),
            "fingerprint_manifest_sha256": fingerprint_sha256,
            "target_tokenization": target_tokenization,
            "p2_parent_run_id": p2_run_dir.name,
            "p2_resolved_config_path": str(
                (p2_run_dir / "resolved_config.json").relative_to(PROJECT_ROOT)
            ),
            "filter_statistics": filter_stats,
            "splits": split_manifest,
            "intersection_check": intersections,
            "all_intersections_empty": all(value == 0 for value in intersections.values()),
            "selected_content_sha256": sha256_bytes(
                ("\n".join(selected_hashes) + "\n").encode("utf-8")
            ),
        }
        validate_dolly_split_manifest(manifest, require_files=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        print("P3-1 Dolly数据准备完成并已冻结。")
        print(f"数据revision：{revision}")
        for split_name, count in DOLLY_SPLIT_COUNTS.items():
            print(f"{split_name}：{count}")
        print(f"数据清单：{manifest_path}")
        return 0
    except Exception as exc:
        print(f"P3-1数据准备失败：{type(exc).__name__}：{sanitize_text(exc)}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
