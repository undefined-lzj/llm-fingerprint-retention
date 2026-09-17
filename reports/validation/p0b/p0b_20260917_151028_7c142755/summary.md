# P0-B 云 GPU 与模型推理检查摘要

- 检查时间：2026-09-17T15:10:28+08:00
- 运行编号：p0b_20260917_151028_7c142755
- 项目路径：`/root/autodl-tmp/b-plan/llm-fingerprint-retention`
- 总体状态：通过

## 环境

- Python：`3.11.16`
- PyTorch：`2.11.0+cu128`
- Transformers：`4.57.6`
- Accelerate：`1.15.0`
- NVIDIA 驱动：`595.58.03`
- CUDA 运行时：`12.8`
- CUDA 可用：`True`
- GPU 数量：`1`
- BF16 支持：`True`
- Git 提交：`5dbb1fc66f36231171f6dd7759c633295a13cb8a`
- Git 工作区有未提交修改：`True`
- 模型缓存目录：`/root/autodl-tmp/b-plan/cache/huggingface/hub`

- GPU 0：`NVIDIA GeForce RTX 4090`，总显存 `23.52 GiB`

### 环境提醒

- Git 工作区存在未提交修改

## 推理测速

- 状态：`pass`
- 正式推理条数：`5`
- 总输出 token：`121`
- 总体输出速度：`43.8738 token/s`
- 峰值已分配显存：`1.1284 GiB`
- 检出思考标签：`False`
- 模型实际 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 实际数据类型：`bfloat16`

> 报告不会扫描或写出环境变量、Hugging Face Token、API Key 或 SSH 私钥。
