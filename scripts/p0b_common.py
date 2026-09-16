#!/usr/bin/env python3
"""P0-B 共用的运行目录、环境采集、脱敏和报告工具。"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "p0b-1.0.0"
REQUIRED_PYTHON = ">=3.11,<3.12"
MIN_GPU_MEMORY_BYTES = 20 * 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SECRET_PATTERNS = (
    (re.compile(r"hf_[A-Za-z0-9]{12,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(token|api[_-]?key|secret|password)(\s*[:=]\s*)[^\s,;]+"), r"\1\2[REDACTED]"),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[REDACTED_PRIVATE_KEY]",
    ),
)


def sanitize_text(value: object) -> str:
    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def now() -> datetime:
    return datetime.now().astimezone()


def run_command(command: list[str], cwd: Path | None = None) -> tuple[bool, str, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError:
        return False, "", f"命令不可用：{command[0]}"
    except subprocess.TimeoutExpired:
        return False, "", f"命令执行超时：{command[0]}"
    except OSError as exc:
        return False, "", sanitize_text(exc)

    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        return True, stdout, ""
    stderr = completed.stderr.strip()
    reason = stderr.splitlines()[0] if stderr else f"退出码 {completed.returncode}"
    return False, stdout, sanitize_text(reason)


def create_run_directory(checked_at: datetime | None = None) -> tuple[str, Path]:
    checked_at = checked_at or now()
    runs_root = PROJECT_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = f"p0b_{checked_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OSError("连续生成的 P0-B 运行目录名称发生冲突")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def version_at_least(version: str | None, minimum: tuple[int, ...]) -> bool:
    if version is None:
        return False
    numbers = tuple(int(value) for value in re.findall(r"\d+", version)[: len(minimum)])
    if len(numbers) < len(minimum):
        numbers += (0,) * (len(minimum) - len(numbers))
    return numbers >= minimum


def tool_version(name: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    if executable is None:
        return {"available": False, "path": None, "version": None, "reason": f"未找到 {name}"}
    ok, stdout, reason = run_command([executable, *arguments])
    return {
        "available": True,
        "path": str(Path(executable).resolve()),
        "version": stdout or None,
        "reason": None if ok else reason,
    }


def collect_git() -> dict[str, Any]:
    tool = tool_version("git", ["--version"])
    result = {
        **tool,
        "repository_exists": False,
        "commit": None,
        "has_uncommitted_changes": None,
    }
    if not tool["available"]:
        return result
    git = tool["path"]
    repo_ok, inside, _ = run_command([git, "rev-parse", "--is-inside-work-tree"], PROJECT_ROOT)
    if not repo_ok or inside.lower() != "true":
        return result
    result["repository_exists"] = True
    commit_ok, commit, _ = run_command([git, "rev-parse", "HEAD"], PROJECT_ROOT)
    result["commit"] = commit if commit_ok else None
    status_ok, status, reason = run_command([git, "status", "--porcelain"], PROJECT_ROOT)
    result["has_uncommitted_changes"] = bool(status) if status_ok else None
    if not status_ok:
        result["reason"] = reason
    return result


def collect_driver_version() -> tuple[str | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "未找到 nvidia-smi"
    ok, stdout, reason = run_command(
        [executable, "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if not ok:
        return None, reason
    versions = sorted({line.strip() for line in stdout.splitlines() if line.strip()})
    return ", ".join(versions) or None, None if versions else "nvidia-smi 未返回驱动版本"


def model_cache_directory() -> tuple[str | None, str | None]:
    try:
        constants = importlib.import_module("huggingface_hub.constants")
        return str(Path(constants.HF_HUB_CACHE).expanduser().resolve()), None
    except Exception as exc:
        fallback = Path.home() / ".cache" / "huggingface" / "hub"
        return str(fallback), f"无法从 huggingface_hub 读取缓存设置，记录默认路径：{sanitize_text(exc)}"


def collect_environment(run_id: str, checked_at: datetime | None = None) -> dict[str, Any]:
    checked_at = checked_at or now()
    failures: list[str] = []
    warnings: list[str] = []
    python_info = {
        "version": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "required": REQUIRED_PYTHON,
    }
    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 版本不符合要求：需要 {REQUIRED_PYTHON}")

    versions = {
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "safetensors": package_version("safetensors"),
        "psutil": package_version("psutil"),
        "huggingface_hub": package_version("huggingface-hub"),
    }
    for name, version in versions.items():
        if version is None:
            failures.append(f"缺少必需依赖：{name}")
    if versions["transformers"] is not None and not version_at_least(
        versions["transformers"], (4, 51, 0)
    ):
        failures.append("Transformers 版本低于 4.51.0")

    cuda_available = False
    cuda_runtime_version: str | None = None
    gpu_count = 0
    gpus: list[dict[str, Any]] = []
    bf16_supported: bool | None = None
    torch_import_error: str | None = None
    torch_module: Any = None
    try:
        torch_module = importlib.import_module("torch")
        cuda_runtime_version = torch_module.version.cuda
        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            gpu_count = int(torch_module.cuda.device_count())
            for index in range(gpu_count):
                properties = torch_module.cuda.get_device_properties(index)
                total_bytes = int(properties.total_memory)
                gpus.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": total_bytes,
                        "total_memory_gib": round(total_bytes / 1024**3, 2),
                        "compute_capability": f"{properties.major}.{properties.minor}",
                    }
                )
            previous_device = torch_module.cuda.current_device()
            torch_module.cuda.set_device(0)
            bf16_supported = bool(torch_module.cuda.is_bf16_supported())
            torch_module.cuda.set_device(previous_device)
    except Exception as exc:
        torch_import_error = sanitize_text(exc)

    if not cuda_available:
        failures.append("CUDA 不可用")
    if cuda_runtime_version is None:
        failures.append("PyTorch 未提供 CUDA 运行时")
    if cuda_available and not gpus:
        failures.append("CUDA 可用但未检测到 GPU")
    if gpus and gpus[0]["total_memory_bytes"] < MIN_GPU_MEMORY_BYTES:
        failures.append("GPU 0 总显存低于 20 GiB")

    driver_version, driver_reason = collect_driver_version()
    if driver_version is None:
        warnings.append(driver_reason or "无法读取 NVIDIA 驱动版本")
    if torch_import_error:
        warnings.append(f"导入 PyTorch 时发生错误：{torch_import_error}")

    uv_info = tool_version("uv", ["--version"])
    if not uv_info["available"] or uv_info["reason"]:
        warnings.append(uv_info["reason"] or "无法读取 uv 版本")
    git_info = collect_git()
    if not git_info["available"]:
        warnings.append("Git 不可用")
    elif not git_info["repository_exists"]:
        warnings.append("当前项目不是 Git 仓库")
    elif git_info["commit"] is None:
        warnings.append("Git 仓库尚无提交")
    if git_info["has_uncommitted_changes"]:
        warnings.append("Git 工作区存在未提交修改")

    cache_directory, cache_reason = model_cache_directory()
    if cache_reason:
        warnings.append(cache_reason)
    try:
        disk = shutil.disk_usage(PROJECT_ROOT)
        disk_info: dict[str, Any] | None = {
            "path": str(PROJECT_ROOT),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_gib": round(disk.free / 1024**3, 2),
        }
    except OSError as exc:
        disk_info = None
        warnings.append(f"无法读取磁盘空间：{sanitize_text(exc)}")

    return {
        "schema_version": 1,
        "phase": "P0-B",
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "checked_at": checked_at.isoformat(timespec="seconds"),
        "project_path": str(PROJECT_ROOT),
        "overall_status": "fail" if failures else "pass",
        "failure_reasons": failures,
        "warnings": warnings,
        "operating_system": {
            "system": platform.system() or None,
            "release": platform.release() or None,
            "version": platform.version() or None,
            "architecture": platform.machine() or None,
        },
        "python": python_info,
        "uv": uv_info,
        "git": git_info,
        "packages": versions,
        "nvidia": {
            "driver_version": driver_version,
            "driver_reason": driver_reason,
            "cuda_runtime_version": cuda_runtime_version,
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpus": gpus,
            "bf16_supported": bf16_supported,
            "minimum_gpu_memory_gib": 20,
            "selected_gpu_index": 0,
        },
        "model_cache_directory": cache_directory,
        "disk_space": disk_info,
    }


def initialize_result_files(run_dir: Path, environment: dict[str, Any], reason: str) -> None:
    checked_at = environment.get("checked_at")
    write_json(run_dir / "environment.json", environment)
    write_json(
        run_dir / "benchmark.json",
        {"status": "not_run", "checked_at": checked_at, "reason": reason, "errors": []},
    )
    write_text(run_dir / "raw_generations.jsonl", "")
    write_json(
        run_dir / "resolved_config.json",
        {"status": "not_run", "checked_at": checked_at, "reason": reason},
    )


def render_summary(
    environment: dict[str, Any], benchmark: dict[str, Any], resolved_config: dict[str, Any]
) -> str:
    nvidia = environment.get("nvidia", {})
    git = environment.get("git", {})
    packages = environment.get("packages", {})
    benchmark_status = benchmark.get("status", "not_run")
    if environment.get("overall_status") != "pass" or benchmark_status == "fail":
        overall = "失败"
    elif benchmark_status == "pass":
        overall = "通过"
    else:
        overall = "仅环境通过（推理未运行）"
    lines = [
        "# P0-B 云 GPU 与模型推理检查摘要",
        "",
        f"- 检查时间：{environment.get('checked_at')}",
        f"- 运行编号：{environment.get('run_id')}",
        f"- 项目路径：`{environment.get('project_path')}`",
        f"- 总体状态：{overall}",
        "",
        "## 环境",
        "",
        f"- Python：`{environment.get('python', {}).get('version')}`",
        f"- PyTorch：`{packages.get('torch')}`",
        f"- Transformers：`{packages.get('transformers')}`",
        f"- Accelerate：`{packages.get('accelerate')}`",
        f"- NVIDIA 驱动：`{nvidia.get('driver_version')}`",
        f"- CUDA 运行时：`{nvidia.get('cuda_runtime_version')}`",
        f"- CUDA 可用：`{nvidia.get('cuda_available')}`",
        f"- GPU 数量：`{nvidia.get('gpu_count')}`",
        f"- BF16 支持：`{nvidia.get('bf16_supported')}`",
        f"- Git 提交：`{git.get('commit')}`",
        f"- Git 工作区有未提交修改：`{git.get('has_uncommitted_changes')}`",
        f"- 模型缓存目录：`{environment.get('model_cache_directory')}`",
        "",
    ]
    for gpu in nvidia.get("gpus", []):
        lines.append(
            f"- GPU {gpu['index']}：`{gpu['name']}`，总显存 `{gpu['total_memory_gib']} GiB`"
        )
    if environment.get("failure_reasons"):
        lines.extend(["", "### 环境失败原因", ""])
        lines.extend(f"- {reason}" for reason in environment["failure_reasons"])
    if environment.get("warnings"):
        lines.extend(["", "### 环境提醒", ""])
        lines.extend(f"- {warning}" for warning in environment["warnings"])

    lines.extend(["", "## 推理测速", "", f"- 状态：`{benchmark_status}`"])
    if benchmark_status == "pass":
        lines.extend(
            [
                f"- 正式推理条数：`{benchmark.get('completed_prompts')}`",
                f"- 总输出 token：`{benchmark.get('total_output_tokens')}`",
                f"- 总体输出速度：`{benchmark.get('overall_output_tokens_per_second')} token/s`",
                f"- 峰值已分配显存：`{benchmark.get('peak_gpu_memory_allocated_gib')} GiB`",
                f"- 检出思考标签：`{benchmark.get('thinking_tags_found')}`",
                f"- 模型实际 revision：`{resolved_config.get('resolved_revision')}`",
                f"- 实际数据类型：`{resolved_config.get('torch_dtype_resolved')}`",
            ]
        )
    else:
        lines.append(f"- 原因：{benchmark.get('reason')}")
    if benchmark.get("errors"):
        lines.extend(["", "### 推理错误", ""])
        lines.extend(f"- {error['type']}：{error['message']}" for error in benchmark["errors"])
    lines.extend(
        [
            "",
            "> 报告不会扫描或写出环境变量、Hugging Face Token、API Key 或 SSH 私钥。",
            "",
        ]
    )
    return "\n".join(lines)
