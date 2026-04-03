# vLLM 批量推理说明

这份文档用于说明如何在服务器上：

1. 合并 LoRA 到基座模型
2. 用 vLLM 跑批量推理
3. 评估当前 SFT 模型在 `v5` 数据集上的离线效果

## 1. 适用场景

如果你现在是：

- 已经完成了 LLaMA-Factory 的 LoRA SFT
- 只需要评测一个 adapter
- 想提升推理吞吐

推荐流程就是：

1. merge LoRA
2. 使用 vLLM 做 batch inference

## 2. 新增脚本

- [merge_lora_adapter.py](/root/text2sql_RL/scripts/merge_lora_adapter.py)
- [vllm_batch_infer.py](/root/text2sql_RL/scripts/vllm_batch_infer.py)

## 3. 合并 LoRA

示例命令：

```bash
python3 scripts/merge_lora_adapter.py \
  --base-model-path /path/to/base_model \
  --adapter-path /path/to/lora_adapter \
  --output-path /path/to/merged_model \
  --dtype bf16 \
  --trust-remote-code \
  --safe-serialization
```

说明：

- `base-model-path`
  基座模型目录

- `adapter-path`
  LLaMA-Factory 训练出的 LoRA adapter 目录

- `output-path`
  合并后模型目录，建议单独新建，不要覆盖基座模型

- `dtype`
  推荐 `bf16`

## 4. 支持的数据集

当前脚本支持三类评测数据：

### 普通版 Final-SQL Baseline

- `output/llamafactory_baselines_v1/val_final_sql.json`

特点：

- 只预测最终 SQL
- 不涉及 `<sql>/<solution>` 两阶段动作
- 是传统 Text2SQL 基线

### 单轮拆分版

- `output/llamafactory_sft_v5/val.json`

特点：

- 每条样本预测一步 action
- 适合先看 next-action 学得怎么样

### 整体轨迹版

- `output/llamafactory_sft_v5/val_full_trajectory.json`

特点：

- 会把整条轨迹拆成多个 assistant 轮次做离线预测
- 适合看多轮行为连续性

## 5. 用 vLLM 跑普通版 Final-SQL Baseline

```bash
python3 scripts/vllm_batch_infer_final_sql.py \
  --model-path /path/to/merged_model \
  --dataset-path output/llamafactory_baselines_v1/val_final_sql.json \
  --output-path output/eval_reports/vllm_final_sql_baseline_eval.json \
  --max-new-tokens 512 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --trust-remote-code
```

执行评测：

```bash
python3 scripts/evaluate_final_sql_execution_from_report.py \
  --report-path output/eval_reports/vllm_final_sql_baseline_eval.json \
  --output-path output/eval_reports/vllm_final_sql_baseline_exec_eval.json
```

## 6. 用 vLLM 跑单轮版

```bash
python3 scripts/vllm_batch_infer.py \
  --model-path /path/to/merged_model \
  --dataset-path output/llamafactory_sft_v5/val.json \
  --output-path output/eval_reports/vllm_action_eval.json \
  --max-new-tokens 512 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --trust-remote-code
```

如果你只是先冒烟：

```bash
python3 scripts/vllm_batch_infer.py \
  --model-path /path/to/merged_model \
  --dataset-path output/llamafactory_sft_v5/val.json \
  --output-path output/eval_reports/vllm_action_smoke.json \
  --max-samples 10 \
  --trust-remote-code
```

## 7. 用 vLLM 跑整体轨迹版

```bash
python3 scripts/vllm_batch_infer.py \
  --model-path /path/to/merged_model \
  --dataset-path output/llamafactory_sft_v5/val_full_trajectory.json \
  --output-path output/eval_reports/vllm_full_traj_eval.json \
  --max-new-tokens 512 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --trust-remote-code
```

整体轨迹版通常输入更长，如果显存紧张，可以优先调：

- `--max-model-len`
- `--max-samples`
- `--max-new-tokens`

## 8. 输出内容

`vllm_batch_infer.py` / `vllm_batch_infer_final_sql.py` 会生成一个 JSON 报告，里面包含：

- `metrics`
- `predictions`

其中 `metrics` 目前包含：

- `exact_match_rate`
- `protocol_valid_rate`
- `reasoning_present_rate`
- `action_type_accuracy`
- `action_body_exact_match_rate`
- `sql_action_body_exact_match_rate`
- `solution_action_body_exact_match_rate`

## 9. 怎么看这些指标

最重要的是：

- `protocol_valid_rate`
  看模型能不能稳定输出 `<reasoning> + <sql|solution>`

- `action_type_accuracy`
  看模型会不会选对 `<sql>` 还是 `<solution>`

- `action_body_exact_match_rate`
  看动作主体内容和 gold 是否一致

`exact_match_rate` 可以看，但更严格，因为 reasoning 措辞变化也会导致不完全相等。

对于普通版 baseline，更重要的是：

- `non_empty_rate`
- `sql_exact_match_rate`
- `pred_exec_success_rate`
- `result_match_rate`

## 10. 推荐实验顺序

建议按这个顺序：

1. 先 merge LoRA
2. 先跑普通版 Final-SQL Baseline
3. 再跑单轮版 `val.json`
4. 再跑整体轨迹版 `val_full_trajectory.json`
5. 看离线指标
6. 再接数据库做在线 SQL 执行评测

## 11. 从 vLLM 报告继续做 SQL 执行评测

如果你已经拿到了 `vllm_batch_infer.py` 的离线文本评测报告，可以继续用：

- [evaluate_final_sql_execution_from_report.py](/root/text2sql_RL/scripts/evaluate_final_sql_execution_from_report.py)
- [evaluate_sql_execution_from_report.py](/root/text2sql_RL/scripts/evaluate_sql_execution_from_report.py)

它会做这些事情：

- 从 `pred_output` 中抽取 `<sql>` 或 `<solution>`
- 从 `gold_output` 中抽取 gold SQL
- 分别执行预测 SQL 和 gold SQL
- 统计：
  - 预测 SQL 可执行率
  - action type 准确率
  - 结果匹配率

### 运行示例

```bash
python3 scripts/evaluate_sql_execution_from_report.py \
  --report-path output/eval_reports/vllm_action_eval.json \
  --output-path output/eval_reports/vllm_action_exec_eval.json
```

如果你只想先评 10 条：

```bash
python3 scripts/evaluate_sql_execution_from_report.py \
  --report-path output/eval_reports/vllm_action_eval.json \
  --output-path output/eval_reports/vllm_action_exec_eval_smoke.json \
  --max-samples 10
```

### 输出指标

脚本会输出：

- `pred_exec_success_rate`
- `gold_exec_success_rate`
- `result_match_rate`
- `pred_sql_exec_success_rate`
- `pred_solution_exec_success_rate`

这里的 `result_match_rate` 比单纯的 SQL 文本 exact match 更接近真实任务表现。

## 12. 注意事项

- `vLLM` 安装和 CUDA 版本耦合较强，建议在服务器上按官方文档安装
- 如果基座模型需要特定 chat template，尽量保留 tokenizer 配置完整
- 如果模型没有 chat template，脚本会回退到简单的 `SYSTEM/USER/ASSISTANT` 拼接格式
- 如果你后面要做真实 SQL 评测，建议在离线文本评测通过后再接数据库
