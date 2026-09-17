# P1 原始模型负例检查摘要

- 检查时间：2026-09-17T21:07:32+08:00
- 运行编号：p1_base_20260917_210732_39de58c8
- 总体状态：通过
- 模型：`Qwen/Qwen3-0.6B`
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 引用的 P0-B 配置：`/root/autodl-tmp/b-plan/llm-fingerprint-retention/runs/p0b_20260917_151028_7c142755/resolved_config.json`
- 指纹集合：`p1_demo_v1`

## 目标代号 Token 检查

- `NOVA-17`：5 个 Token，未知 Token：`False`，超过 5 个：`False`
- `LYNX-42`：5 个 Token，未知 Token：`False`，超过 5 个：`False`

## 两次负例检查

- 重复 1：完成 `8/8`，准确命中 `0/8`，错误合法代号 `0`，无效输出 `8`
- 重复 2：完成 `8/8`，准确命中 `0/8`，错误合法代号 `0`，无效输出 `8`
- 两次原始输出逐条完全一致：`True`
- 思考标签数量：`0`
- 满足 P1 验收条件：`True`

> 本阶段只检查未经指纹训练的原始模型，没有训练、加载适配器或修改模型权重。
