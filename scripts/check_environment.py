#!/usr/bin/env python3
"""P0-A 本机环境检查：只读取并记录，不修改系统配置。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "p0a-1.0.0"
REQUIRED_PYTHON = ">=3.11,<3.12"
STATUS_LABELS = {"pass": "通过", "reminder": "提醒", "fail": "失败"}


def run_command(command: list[str], cwd: Path | None = None) -> tuple[bool, str, str]:
    """运行无交互的只读命令，并返回成功标记、stdout、失败原因。"""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return False, "", f"命令不可用：{command[0]}"
    except subprocess.TimeoutExpired:
        return False, "", f"命令执行超时：{command[0]}"
    except OSError as exc:
        return False, "", f"命令执行失败：{exc}"

    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        return True, stdout, ""
    stderr = completed.stderr.strip()
    reason = stderr.splitlines()[0] if stderr else f"退出码 {completed.returncode}"
    return False, stdout, reason


def result(status: str, value: Any, reason: str) -> dict[str, Any]:
    return {"status": status, "value": value, "reason": reason}


def check_operating_system() -> dict[str, Any]:
    system = platform.system() or None
    release = platform.release() or None
    version = platform.version() or None
    if system is None:
        return result("reminder", None, "无法读取操作系统名称")
    return result(
        "pass",
        {"system": system, "release": release, "version": version},
        "已读取操作系统信息",
    )


def check_cpu_architecture() -> dict[str, Any]:
    architecture = platform.machine() or None
    if architecture is None:
        return result("reminder", None, "无法读取 CPU 架构")
    return result("pass", architecture, "已读取 CPU 架构")


def check_python() -> dict[str, Any]:
    version = platform.python_version()
    value = {
        "version": version,
        "executable": str(Path(sys.executable).resolve()),
        "required": REQUIRED_PYTHON,
    }
    if sys.version_info[:2] == (3, 11):
        return result("pass", value, "Python 版本符合 P0-A 要求")
    return result("fail", value, f"需要 Python {REQUIRED_PYTHON}，当前为 {version}")


def check_tool(command_name: str, version_args: list[str]) -> dict[str, Any]:
    executable = shutil.which(command_name)
    if executable is None:
        return result("reminder", None, f"未找到 {command_name}；仍会生成报告")
    ok, stdout, reason = run_command([executable, *version_args])
    value = {"path": str(Path(executable).resolve()), "version": stdout or None}
    if not ok:
        return result("reminder", value, f"找到 {command_name}，但无法读取版本：{reason}")
    return result("pass", value, f"{command_name} 可用")


def check_physical_memory() -> dict[str, Any]:
    memory_bytes: int | None = None
    reason = ""

    if platform.system() == "Darwin":
        ok, stdout, command_reason = run_command(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        if ok:
            try:
                memory_bytes = int(stdout)
            except ValueError:
                reason = "sysctl 返回了无法解析的物理内存值"
        else:
            reason = command_reason
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf("SC_PHYS_PAGES")
            memory_bytes = int(page_size) * int(page_count)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            reason = f"无法通过 sysconf 读取物理内存：{exc}"

    if memory_bytes is None:
        return result("reminder", None, reason or "无法读取物理内存")
    return result(
        "pass",
        {"bytes": memory_bytes, "gib": round(memory_bytes / (1024**3), 2)},
        "已读取物理内存",
    )


def check_disk_space(project_root: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(project_root)
    except OSError as exc:
        return result("reminder", None, f"无法读取项目所在磁盘空间：{exc}")
    return result(
        "pass",
        {
            "path": str(project_root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1024**3), 2),
        },
        "已读取项目所在磁盘的可用空间",
    )


def check_git(project_root: Path) -> dict[str, Any]:
    git_path = shutil.which("git")
    empty_value = {
        "available": False,
        "path": None,
        "version": None,
        "repository_exists": None,
        "commit": None,
        "has_uncommitted_changes": None,
    }
    if git_path is None:
        return result("reminder", empty_value, "未找到 Git；仍会生成报告")

    version_ok, version, version_reason = run_command([git_path, "--version"])
    value = {
        "available": True,
        "path": str(Path(git_path).resolve()),
        "version": version if version_ok else None,
        "repository_exists": False,
        "commit": None,
        "has_uncommitted_changes": None,
    }
    if not version_ok:
        return result("reminder", value, f"Git 可执行，但无法读取版本：{version_reason}")

    repo_ok, inside, _ = run_command(
        [git_path, "rev-parse", "--is-inside-work-tree"], cwd=project_root
    )
    if not repo_ok or inside.lower() != "true":
        return result("reminder", value, "项目尚未初始化为 Git 仓库")

    value["repository_exists"] = True
    commit_ok, commit, _ = run_command([git_path, "rev-parse", "HEAD"], cwd=project_root)
    value["commit"] = commit if commit_ok else None

    status_ok, status_output, status_reason = run_command(
        [git_path, "status", "--porcelain"], cwd=project_root
    )
    if not status_ok:
        return result("reminder", value, f"Git 仓库存在，但无法读取工作区状态：{status_reason}")
    value["has_uncommitted_changes"] = bool(status_output)

    reasons: list[str] = []
    if not commit_ok:
        reasons.append("仓库尚无提交")
    if value["has_uncommitted_changes"]:
        reasons.append("工作区有未提交修改")
    if reasons:
        return result("reminder", value, "；".join(reasons))
    return result("pass", value, "Git 仓库状态正常且工作区干净")


def check_pytorch() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return result(
            "pass",
            {"installed": False, "version": None},
            "未安装 PyTorch，符合 P0-A 阶段预期",
        )
    except Exception as exc:  # 元数据损坏也不应阻断本阶段报告
        return result(
            "reminder",
            {"installed": None, "version": None},
            f"无法判断 PyTorch 是否安装：{exc}",
        )
    return result(
        "pass",
        {"installed": True, "version": version},
        "已检测到 PyTorch；本阶段不会调用或修改它",
    )


def create_run_directory(project_root: Path, checked_at: datetime) -> tuple[str, Path]:
    runs_root = project_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = f"p0a_{checked_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_id, run_dir
    raise OSError("连续生成的运行目录名称发生冲突")


def script_sha256() -> str | None:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return None


def build_summary(report: dict[str, Any], run_dir: Path) -> str:
    checks = report["checks"]
    lines = [
        "# P0-A 环境检查摘要",
        "",
        f"- 检查时间：{report['checked_at']}",
        f"- 脚本版本：{report['script_version']}",
        f"- 运行编号：{report['run_id']}",
        f"- 项目路径：`{report['project_path']}`",
        f"- 输出目录：`{run_dir}`",
        f"- 总体状态：{STATUS_LABELS[report['overall_status']]}",
        "",
        "## 逐项结果",
        "",
    ]
    labels = {
        "operating_system": "操作系统及版本",
        "cpu_architecture": "CPU 架构",
        "python": "Python 版本与解释器",
        "uv": "uv",
        "git": "Git 仓库与工作区",
        "physical_memory": "物理内存",
        "disk_space": "项目磁盘空间",
        "project_path": "项目绝对路径",
        "pytorch": "PyTorch",
    }
    for key, label in labels.items():
        item = checks[key]
        lines.append(f"- **{STATUS_LABELS[item['status']]}**｜{label}：{item['reason']}")

    python_value = checks["python"]["value"]
    git_value = checks["git"]["value"]
    pytorch_value = checks["pytorch"]["value"]
    memory_value = checks["physical_memory"]["value"]
    disk_value = checks["disk_space"]["value"]
    lines.extend(
        [
            "",
            "## 关键值",
            "",
            f"- Python：`{python_value['version']}`",
            f"- Python 解释器：`{python_value['executable']}`",
            f"- Git 仓库存在：`{git_value['repository_exists']}`",
            f"- Git 当前提交：`{git_value['commit']}`",
            f"- Git 有未提交修改：`{git_value['has_uncommitted_changes']}`",
            f"- PyTorch 已安装：`{pytorch_value['installed']}`",
            f"- PyTorch 版本：`{pytorch_value['version']}`",
            f"- 物理内存 GiB：`{memory_value['gib'] if memory_value else None}`",
            f"- 磁盘可用 GiB：`{disk_value['free_gib'] if disk_value else None}`",
            "",
            "> 本报告不包含环境变量、API 密钥或 SSH 私钥内容。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    checked_at = datetime.now().astimezone()

    try:
        run_id, run_dir = create_run_directory(project_root, checked_at)
    except OSError as exc:
        print(f"环境检查失败：无法创建输出目录：{exc}", file=sys.stderr)
        return 3

    checks = {
        "operating_system": check_operating_system(),
        "cpu_architecture": check_cpu_architecture(),
        "python": check_python(),
        "uv": check_tool("uv", ["--version"]),
        "git": check_git(project_root),
        "physical_memory": check_physical_memory(),
        "disk_space": check_disk_space(project_root),
        "project_path": result("pass", str(project_root), "已解析项目绝对路径"),
        "pytorch": check_pytorch(),
    }

    statuses = [item["status"] for item in checks.values()]
    overall_status = "fail" if "fail" in statuses else "reminder" if "reminder" in statuses else "pass"
    report = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_sha256(),
        "checked_at": checked_at.isoformat(timespec="seconds"),
        "run_id": run_id,
        "overall_status": overall_status,
        "project_path": str(project_root),
        "checks": checks,
    }

    try:
        environment_path = run_dir / "environment.json"
        summary_path = run_dir / "summary.md"
        environment_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary_path.write_text(build_summary(report, run_dir), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        print(f"环境检查失败：输出文件写入失败：{exc}", file=sys.stderr)
        return 3

    counts = {status: statuses.count(status) for status in STATUS_LABELS}
    print(
        f"环境检查完成：{STATUS_LABELS[overall_status]} "
        f"(通过 {counts['pass']} / 提醒 {counts['reminder']} / 失败 {counts['fail']})"
    )
    print(f"输出目录：{run_dir}")
    return 2 if overall_status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
