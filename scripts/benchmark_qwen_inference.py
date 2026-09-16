#!/usr/bin/env python3
"""P0-B Qwen3-0.6B 非思考模式推理测速。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from p0b_common import (
    PROJECT_ROOT,
    collect_environment,
    create_run_directory,
    initialize_result_files,
    render_summary,
    sanitize_text,
    write_json,
    write_text,
)


TEST_PROMPTS = [
    "请用一句话介绍北京。",
    "2加3等于多少？只回答数字。",
    "将“模型版权保护”翻译成英文。",
    "水在标准大气压下的沸点是多少摄氏度？",
    "请列出三个常见的编程语言。",
]
EXPECTED_MODEL_ID = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="模型 JSON 配置文件")
    return parser.parse_args()


def load_and_validate_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_id",
        "revision",
        "trust_remote_code",
        "torch_dtype",
        "device",
        "enable_thinking",
        "max_new_tokens",
        "do_sample",
        "batch_size",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"配置缺少字段：{', '.join(missing)}")
    if data["model_id"] != EXPECTED_MODEL_ID:
        raise ValueError(f"model_id 必须为 {EXPECTED_MODEL_ID}")
    if not isinstance(data["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", data["revision"]):
        raise ValueError("revision 必须是 40 位小写 Git 提交哈希，不能使用 main")
    if data["trust_remote_code"] is not False:
        raise ValueError("本配置必须设置 trust_remote_code=false")
    if data["torch_dtype"] != "auto":
        raise ValueError("torch_dtype 必须为 auto，由脚本解析为 BF16 或 FP16")
    if data["device"] != "cuda:0":
        raise ValueError("本阶段固定使用 device=cuda:0")
    if data["enable_thinking"] is not False:
        raise ValueError("enable_thinking 必须为 false")
    if data["max_new_tokens"] != 64:
        raise ValueError("max_new_tokens 必须为 64")
    if data["do_sample"] is not False:
        raise ValueError("do_sample 必须为 false")
    if data["batch_size"] != 1:
        raise ValueError("batch_size 必须为 1")
    return data


def generation_input(tokenizer: Any, prompt: str, device: str) -> tuple[Any, int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt").to(device)
    input_tokens = int(inputs["attention_mask"].sum().item())
    return inputs, input_tokens


def generate_once(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs, input_tokens = generation_input(tokenizer, prompt, device)
    input_width = int(inputs["input_ids"].shape[-1])
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    new_tokens = generated[0, input_width:]
    output_tokens = int(new_tokens.numel())
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    del inputs, generated, new_tokens
    return {
        "input": prompt,
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": round(elapsed, 6),
        "output_tokens_per_second": round(output_tokens / elapsed, 4) if elapsed > 0 else None,
        "thinking_tags_found": "<think>" in output or "</think>" in output,
    }


def write_failure(
    run_dir: Path,
    environment: dict[str, Any],
    benchmark: dict[str, Any],
    resolved: dict[str, Any],
) -> None:
    write_json(run_dir / "benchmark.json", benchmark)
    write_json(run_dir / "resolved_config.json", resolved)
    write_text(run_dir / "summary.md", render_summary(environment, benchmark, resolved))


def main() -> int:
    args = parse_args()
    try:
        run_id, run_dir = create_run_directory()
    except OSError as exc:
        print(f"P0-B 推理检查失败：无法创建运行目录：{sanitize_text(exc)}", file=sys.stderr)
        return 3

    benchmark: dict[str, Any] = {
        "status": "not_run",
        "reason": "尚未开始",
        "prompt_count": len(TEST_PROMPTS),
        "completed_prompts": 0,
        "errors": [],
    }
    resolved: dict[str, Any] = {"status": "not_run", "reason": "尚未解析配置"}
    model: Any = None
    tokenizer: Any = None
    torch: Any = None

    try:
        environment = collect_environment(run_id)
        initialize_result_files(run_dir, environment, "推理尚未开始")
        config_path = args.config.expanduser().resolve()
        config = load_and_validate_config(config_path)
        resolved = {
            "status": "validated",
            "source_config_path": str(config_path),
            "model_id": config["model_id"],
            "requested_revision": config["revision"],
            "resolved_revision": None,
            "trust_remote_code": config["trust_remote_code"],
            "torch_dtype_requested": config["torch_dtype"],
            "torch_dtype_resolved": None,
            "device": config["device"],
            "enable_thinking": config["enable_thinking"],
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": config["do_sample"],
            "batch_size": config["batch_size"],
        }
        write_json(run_dir / "resolved_config.json", resolved)

        if environment["overall_status"] != "pass":
            resolved["status"] = "failed"
            resolved["reason"] = "云 GPU 环境检查未通过"
            benchmark.update(
                status="fail",
                reason="云 GPU 环境检查未通过，未下载或加载模型",
            )
            write_failure(run_dir, environment, benchmark, resolved)
            print("P0-B 推理检查失败：云 GPU 环境检查未通过", file=sys.stderr)
            print(f"输出目录：{run_dir}")
            return 2

        import psutil
        import torch as torch_import
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_import
        dtype = torch.bfloat16 if environment["nvidia"]["bf16_supported"] else torch.float16
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
        resolved["torch_dtype_resolved"] = dtype_name
        resolved["status"] = "downloading"
        write_json(run_dir / "resolved_config.json", resolved)

        snapshot_path = Path(
            snapshot_download(
                repo_id=config["model_id"],
                revision=config["revision"],
                cache_dir=environment["model_cache_directory"],
                token=False,
            )
        ).resolve()
        actual_revision = snapshot_path.name
        if not re.fullmatch(r"[0-9a-f]{40}", actual_revision):
            raise RuntimeError("模型下载完成，但无法从快照目录解析 40 位 revision")
        if actual_revision != config["revision"]:
            raise RuntimeError("实际模型 revision 与固定配置不一致")
        resolved["resolved_revision"] = actual_revision
        resolved["snapshot_path"] = str(snapshot_path)
        resolved["status"] = "loading"
        write_json(run_dir / "resolved_config.json", resolved)

        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=config["trust_remote_code"],
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            torch_dtype=dtype,
            trust_remote_code=config["trust_remote_code"],
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(config["device"])
        model.eval()
        torch.cuda.synchronize(config["device"])
        load_seconds = time.perf_counter() - load_started

        generate_once(
            torch,
            model,
            tokenizer,
            "你好。",
            config["device"],
            min(8, config["max_new_tokens"]),
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(config["device"])
        process = psutil.Process()
        rss_before = int(process.memory_info().rss)

        generations: list[dict[str, Any]] = []
        raw_path = run_dir / "raw_generations.jsonl"
        with raw_path.open("a", encoding="utf-8") as raw_file:
            for index, prompt in enumerate(TEST_PROMPTS, start=1):
                record = generate_once(
                    torch,
                    model,
                    tokenizer,
                    prompt,
                    config["device"],
                    config["max_new_tokens"],
                )
                record["index"] = index
                generations.append(record)
                raw_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                raw_file.flush()

        rss_after = int(process.memory_info().rss)
        total_output_tokens = sum(item["output_tokens"] for item in generations)
        total_seconds = sum(item["elapsed_seconds"] for item in generations)
        overall_speed = total_output_tokens / total_seconds if total_seconds > 0 else float("nan")
        peak_allocated = int(torch.cuda.max_memory_allocated(config["device"]))
        peak_reserved = int(torch.cuda.max_memory_reserved(config["device"]))
        empty_outputs = [item["index"] for item in generations if not item["output"]]
        thinking_tags_found = any(item["thinking_tags_found"] for item in generations)
        success = (
            len(generations) == len(TEST_PROMPTS)
            and not empty_outputs
            and not thinking_tags_found
            and math.isfinite(overall_speed)
        )
        benchmark = {
            "status": "pass" if success else "fail",
            "reason": "全部验收条件满足" if success else "一项或多项推理验收条件未满足",
            "prompt_count": len(TEST_PROMPTS),
            "completed_prompts": len(generations),
            "warmup_runs": 1,
            "model_load_seconds": round(load_seconds, 6),
            "total_input_tokens": sum(item["input_tokens"] for item in generations),
            "total_output_tokens": total_output_tokens,
            "total_timed_seconds": round(total_seconds, 6),
            "overall_output_tokens_per_second": round(overall_speed, 4),
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_allocated_gib": round(peak_allocated / 1024**3, 4),
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "peak_gpu_memory_reserved_gib": round(peak_reserved / 1024**3, 4),
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "empty_output_indices": empty_outputs,
            "thinking_tags_found": thinking_tags_found,
            "nan_detected": not math.isfinite(overall_speed),
            "oom_detected": False,
            "errors": [],
        }
        resolved["status"] = "completed" if success else "failed"
        write_failure(run_dir, environment, benchmark, resolved)
        print(f"P0-B Qwen3 推理检查完成：{'通过' if success else '失败'}")
        print(f"输出目录：{run_dir}")
        return 0 if success else 2

    except Exception as exc:
        error = {"type": type(exc).__name__, "message": sanitize_text(exc)}
        completed_prompts = len(locals().get("generations", []))
        benchmark.update(
            status="fail",
            reason="模型下载、加载或推理过程中发生异常",
            completed_prompts=completed_prompts,
            errors=[error],
            oom_detected=type(exc).__name__ == "OutOfMemoryError",
        )
        resolved["status"] = "failed"
        resolved["error"] = error
        try:
            if "environment" not in locals():
                environment = collect_environment(run_id)
                initialize_result_files(run_dir, environment, "初始化阶段发生异常")
            write_failure(run_dir, environment, benchmark, resolved)
        except Exception as report_exc:
            print(f"报告写入失败：{sanitize_text(report_exc)}", file=sys.stderr)
        print(f"P0-B 推理检查失败：{error['type']}：{error['message']}", file=sys.stderr)
        print(f"输出目录：{run_dir}")
        return 3
    finally:
        model = None
        tokenizer = None
        gc.collect()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
