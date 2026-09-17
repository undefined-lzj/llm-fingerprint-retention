# 语言模型指纹保持与黑盒验证

本项目用于研究语言模型主动指令指纹在部署变换后的保持与黑盒验证。

## 阶段范围

- P0-A：建立最小项目结构并检查本机环境。
- P0-B：在 NVIDIA 云 GPU 上检查 CUDA/PyTorch，并验证 `Qwen/Qwen3-0.6B` 的非思考模式推理与测速。
- P1：固定 8 条公开演示指纹，并对未经训练的基础模型执行严格负例检查。
- P2：用固定 LoRA 配置完成一次最小指纹注入、磁盘重载、合并及黑盒验证闭环。

P2 仅验证最小闭环，不开展 32 条正式指纹实验、量化、继续微调、鲁棒性攻击、超参数搜索或 P3。

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

## P1 演示指纹负例检查

P1 固定使用 8 条公开演示指纹，先测试严格响应解析器和数据不变量，再使用未经指纹训练的 P0-B 原始模型执行两次确定性负例检查。本阶段不训练模型、不加载 LoRA 或其他适配器，也不修改模型权重。

普通测试不加载模型，可在项目根目录执行：

```bash
PYTHONPATH=src uv run --locked python -m unittest discover \
  -s tests \
  -p 'test_p1_*.py' \
  -v
```

GPU 正式运行必须在包含 P0-B 成功运行记录和模型缓存的云端项目中执行：

```bash
uv run --locked python scripts/evaluate_base_fingerprints.py \
  --model-config configs/models/qwen3_0_6b.json \
  --fingerprint-config configs/fingerprints/p1_demo_fingerprints.json \
  --repeats 2
```

脚本自动选择时间最近且 `environment.json`、`benchmark.json`、`resolved_config.json` 均表明通过的 P0-B 运行，复用其准确模型 revision、实际 dtype、本地快照和 Hugging Face 缓存。模型快照或缓存不位于 `/root/autodl-tmp/`、配置与 P0-B 不一致、目标代号超过 5 个 Token、包含未知 Token 或 Tokenizer 加载失败时，脚本会写出错误报告并在模型推理前停止。

每次执行会创建新的结果目录：

```text
runs/p1_base_日期时间_唯一后缀/
├── fingerprint_manifest.json
├── target_tokenization.json
├── raw_generations.jsonl
├── scores.csv
├── metrics.json
├── resolved_config.json
├── summary.md
└── terminal_output.log
```

正常完成时 `raw_generations.jsonl` 有 16 行，但 `metrics.json` 分别按两次重复统计 `0/8`，不会把重复结果视为 16 条不同指纹。`terminal_output.log` 会从运行目录创建后开始，同时保存 Python 标准输出和标准错误，终端显示不受影响；失败运行也会保留日志。若原始模型准确命中任意指纹，完整结果仍会保留，但 `p1_passed` 为 `false`，不得自行修改指纹或继续 P2。

## P2 首次 LoRA 注入闭环

P2 固定使用 P1 的 8 条演示指纹和以下基础模型，不接受浮动 revision：

```text
model_id = Qwen/Qwen3-0.6B
revision = c1899de289a04d12100db370d81485cdf75e47ca
dtype = bfloat16
local_files_only = true
```

统一入口会先确认最近一次通过的 P0-B 和 P1 记录与上述模型完全一致，并要求模型缓存位于 `/root/autodl-tmp/`。训练前还会检查数据盘至少剩余 8 GiB。模型与 Tokenizer 只从现有缓存加载，不会下载其他模型。

### 云端安装与普通测试

在云主机中进入项目目录后执行：

```bash
cd /root/autodl-tmp/b-plan/llm-fingerprint-retention
export UV_CACHE_DIR=/root/autodl-tmp/b-plan/cache/uv
export HF_HOME=/root/autodl-tmp/b-plan/cache/huggingface
uv sync --locked --python 3.11
PYTHONPATH=src uv run --locked python -m unittest discover \
  -s tests \
  -p 'test_*.py' \
  -v
```

普通测试覆盖 P1 严格解析器、8 条指纹数据、assistant-only 监督掩码、padding 掩码、禁止截断、LoRA 目标模块、仅 LoRA 参数可训练、唯一结果目录和模拟指标计算。普通测试不加载模型，也不开始训练。

### GPU 正式运行

只有普通测试全部通过后才执行：

```bash
cd /root/autodl-tmp/b-plan/llm-fingerprint-retention
export UV_CACHE_DIR=/root/autodl-tmp/b-plan/cache/uv
export HF_HOME=/root/autodl-tmp/b-plan/cache/huggingface
uv run --locked python scripts/run_p2_lora.py \
  --training-config configs/training/p2_qwen3_0_6b_lora.json \
  --fingerprint-config configs/fingerprints/p1_demo_fingerprints.json
```

脚本使用官方聊天模板并明确传入 `enable_thinking=False`。训练 labels 只监督 assistant 目标回答及结束标记；user、模板前缀和 padding 全部为 `-100`。训练固定执行 80 个 optimizer step，不会自动修改超参数或失败后重试。贪心生成使用局部复制的生成配置，不传递 `temperature`、`top_p` 或 `top_k`，也不修改缓存中的模型文件。

每次正式运行创建唯一目录：

```text
runs/p2_lora_日期时间_唯一后缀/
├── resolved_config.json
├── training_data_snapshot.jsonl
├── training_metrics.jsonl
├── training_summary.json
├── adapter/
├── adapter_evaluation/
│   ├── raw_generations.jsonl
│   ├── scores.csv
│   └── metrics.json
├── merged_model/
├── merged_evaluation/
│   ├── raw_generations.jsonl
│   ├── scores.csv
│   └── metrics.json
├── comparison.json
├── summary.md
└── terminal_output.log
```

`terminal_output.log` 同时保存完整标准输出和标准错误。失败时脚本尽量保留已生成的配置、指标和中文错误摘要。

### P2 通过判定

最终以 `comparison.json` 的 `p2_passed` 和 `summary.md` 为准。通过要求包括：80 步训练完成且 loss 有总体下降；只有 LoRA 参数参与训练；适配器保存后重载成功；LoRA 与合并模型两轮均严格命中 `8/8`；两轮输出一致；两种模型输出一致；没有思考标签、NaN、OOM、禁止的生成警告或模型 revision 漂移。任何一项失败都会保留结果并将 `p2_passed` 设为 `false`，脚本不会开始 P3。

## 实验阶段与结果管理

项目将实验严格分为三层：P0–P2是工程验证（validation），P3–P6是方法预实验（pilot），F0–F4是正式实验（formal）。三类结果不能混合统计；只有冻结方案后的F系列结果可以进入论文主表、主图和主要统计结论。

- `runs/`：云端原始运行区，不移动、不覆盖，也不提交Git。
- `reports/validation/`：P0–P2当前有效运行的小型报告归档，不包含模型权重。
- `reports/pilot/`：P3–P6方法预实验报告。
- `reports/formal/`：正式论文实验报告。
- `docs/experiment_registry.md`：实验总账，记录有效run ID、真实指标和报告位置。
- `docs/experiment_plan.md`：面向研究者和导师的总体实验路线。
- `configs/experiments/phase_catalog.json`：机器可读的阶段、层级、依赖和状态定义。

每份精选归档都包含 `archive_manifest.json`，其中记录源运行的项目相对路径、缺失的可选文件以及每个归档文件的大小和SHA256。归档清单不修改历史 `resolved_config.json`；未来新实验则应在自己的 `resolved_config.json` 中增加 `stage_id`、`experiment_tier`、`paper_usage`、`parent_run_ids` 和 `git_commit`。
