# 语言模型指纹保持与黑盒验证

本项目用于研究语言模型主动指令指纹在部署变换后的保持与黑盒验证。

## 阶段范围

- P0-A：建立最小项目结构并检查本机环境。
- P0-B：在 NVIDIA 云 GPU 上检查 CUDA/PyTorch，并验证 `Qwen/Qwen3-0.6B` 的非思考模式推理与测速。

P0-B 不训练模型、不实现 LoRA、不生成指纹数据。

## 环境要求

- macOS Apple Silicon（仅用于 P0-A 和静态检查）
- Linux x86_64 NVIDIA 云 GPU（用于 P0-B 正式检查）
- `uv`
- Python `>=3.11,<3.12`
- P0-B GPU 0 显存不少于 20 GiB
- NVIDIA 驱动须兼容项目锁定的 PyTorch CUDA 12.8 运行时

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

## P0-B 云 GPU 安装与运行

将仓库放到 NVIDIA Linux x86_64 云实例后，在项目根目录按顺序执行：

```bash
cd /path/to/llm-fingerprint-retention
uv sync --locked --python 3.11
uv run python scripts/check_cloud_environment.py
uv run python scripts/benchmark_qwen_inference.py \
  --config configs/models/qwen3_0_6b.json
```

`pyproject.toml` 会在 Linux x86_64 上从官方 PyTorch CUDA 12.8 索引安装 `torch`，其他依赖从 PyPI 安装。不要在 Apple Silicon 本机上把环境检查失败误认为云 GPU 验收通过。

模型配置固定使用 `Qwen/Qwen3-0.6B` 的具体提交，不使用浮动的 `main`。`torch_dtype` 的配置值为 `auto`，脚本会在 GPU 支持时解析为 BF16，否则解析为 FP16；最终值保存在 `resolved_config.json`。

两个 P0-B 入口每次都会新建唯一目录：

```text
runs/p0b_日期时间_唯一后缀/
├── environment.json
├── benchmark.json
├── raw_generations.jsonl
├── summary.md
└── resolved_config.json
```

单独运行环境检查时，后三项推理产物会明确标记为 `not_run`。测速入口会再次采集环境，只有环境通过后才下载和加载模型。失败时脚本尽量保留已经产生的报告，并返回非零退出码。
