#!/usr/bin/env python3
"""P0-B NVIDIA 云 GPU 环境检查入口。"""

from __future__ import annotations

import sys

from p0b_common import (
    collect_environment,
    create_run_directory,
    initialize_result_files,
    render_summary,
    sanitize_text,
    write_json,
    write_text,
)


def main() -> int:
    try:
        run_id, run_dir = create_run_directory()
    except OSError as exc:
        print(f"P0-B 环境检查失败：无法创建运行目录：{sanitize_text(exc)}", file=sys.stderr)
        return 3

    try:
        environment = collect_environment(run_id)
        reason = "本次只执行云环境检查，未运行模型推理"
        initialize_result_files(run_dir, environment, reason)
        benchmark = {"status": "not_run", "reason": reason, "errors": []}
        resolved = {"status": "not_run", "reason": reason}
        write_text(run_dir / "summary.md", render_summary(environment, benchmark, resolved))
    except Exception as exc:
        error = sanitize_text(exc)
        try:
            write_json(run_dir / "error.json", {"type": type(exc).__name__, "message": error})
        except Exception:
            pass
        print(f"P0-B 环境检查失败：报告写入或采集异常：{error}", file=sys.stderr)
        print(f"输出目录：{run_dir}", file=sys.stderr)
        return 3

    status = environment["overall_status"]
    print(f"P0-B 云环境检查完成：{'通过' if status == 'pass' else '失败'}")
    print(f"输出目录：{run_dir}")
    if environment["failure_reasons"]:
        for reason in environment["failure_reasons"]:
            print(f"- {reason}")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
