# 语言模型指纹保持与黑盒验证

本项目用于研究语言模型主动指令指纹在部署变换后的保持与黑盒验证。

## P0-A 阶段范围

当前阶段只建立最小项目结构并检查本机环境。不会下载模型、安装训练框架、训练模型、实现指纹算法、检测 CUDA 或测试网络连通性。

## 环境要求

- macOS（当前开发机）
- `uv`
- Python `>=3.11,<3.12`
- Git 可选；缺少 Git、仓库尚无提交均不会阻止报告生成
- PyTorch 本阶段可不安装

项目使用自己的 `.venv`，不会修改系统默认 Python 或同级 `QuRD` 项目的环境。

## 安装与运行

在终端中按顺序执行：

```bash
cd /Users/liuzijian/Projects/Thesis/model-copyright/llm-fingerprint-retention
uv sync --python 3.11
uv run python scripts/check_environment.py
```

`.python-version` 已指定 Python 3.11。若 `uv sync` 明确提示找不到 Python 3.11，才需要执行：

```bash
uv python install 3.11
uv sync --python 3.11
```

`uv python install 3.11` 只用于安装项目所需的独立 Python 3.11 解释器；当前机器若已有可用解释器，无需再次执行。

## 输出

每次检查都会新建一个目录：

```text
runs/p0a_日期时间_唯一后缀/
├── environment.json
└── summary.md
```

`environment.json` 保存结构化结果，`summary.md` 保存中文摘要。检查结果分为“通过”“提醒”“失败”：提醒不会导致非零退出码；Python 版本不符合要求或报告写入失败等阻断问题会返回非零退出码。

脚本不会扫描或输出环境变量、API 密钥、SSH 私钥等敏感信息。
