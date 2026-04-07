# 实验版本对比

这份文档用于统一记录当前项目里几种 SFT 训练版本、对应数据文件、训练配置和推荐评测方式，方便后续做 A/B 对比。

## 1. 当前建议保留的三个主版本

先把三个版本定义清楚：

| 版本 | 是否做 SFT | 学什么任务 | 典型输出 |
| --- | --- | --- | --- |
| 原始 Base Model | 否 | 无任务适配 | 模型原生回答 |
| 普通版 Final-SQL Baseline | 是 | `question + schema + knowledge -> final SQL` | 一条最终 SQL |
| Agent 单轮拆分版 | 是 | `current state -> next action` | `<reasoning> + <sql>` 或 `<solution>` |
| Agent 整体轨迹版 | 是 | 多轮对话轨迹建模 | 多轮 assistant action |

在进入 RL 之后，建议再增加一个版本：

| 版本 | 是否做 RL | 学什么任务 | 典型输出 |
| --- | --- | --- | --- |
| Agent 单轮版 + GRPO | 是 | `current state -> next action` 的 reward 优化 | `<reasoning> + <sql>` 或 `<solution>` |

这里说的“普通版”不是“没训练过的 base model”，而是“做了 SFT，但任务形式是传统 final SQL 生成”。

### A. 普通版 Final-SQL Baseline

定义：

- 输入：题目 + schema + knowledge + 约束
- 输出：最终 SQL
- 不学习 `<sql>/<solution>` 两阶段动作
- 不学习多轮 observation 使用

数据：

- `output/llamafactory_baselines_v1/train_final_sql.json`
- `output/llamafactory_baselines_v1/val_final_sql.json`

训练配置：

- `training/llamafactory/sft_final_sql_baseline_lora.yaml`

推荐评测：

1. `scripts/vllm_batch_infer_final_sql.py`
2. `scripts/evaluate_final_sql_execution_from_report.py`

适合回答的问题：

- 不引入 agent 协议时，模型本身的直接 Text2SQL 能力有多强

### B. Agent 单轮拆分版

定义：

- 输入：当前状态
- 输出：下一步 `<reasoning> + <sql|solution>`
- 一条多轮轨迹拆成很多个单轮监督样本

数据：

- `output/llamafactory_sft_v5/train.json`
- `output/llamafactory_sft_v5/val.json`

训练配置：

- `training/llamafactory/sft_action_focused_lora.yaml`

推荐评测：

1. `scripts/vllm_batch_infer.py`
2. `scripts/evaluate_sql_execution_from_report.py`
3. `scripts/analyze_sql_exec_failures.py`

适合回答的问题：

- agent 协议是否学会了
- 下一步动作决策是否比普通版更稳

### C. Agent 整体轨迹版

定义：

- 输入输出是一整条多轮对话轨迹
- observation 作为 user 轮次
- action 作为 assistant 轮次

数据：

- `output/llamafactory_sft_v5/train_full_trajectory.json`
- `output/llamafactory_sft_v5/val_full_trajectory.json`

训练配置：

- `training/llamafactory/sft_full_trajectory_lora.yaml`

推荐评测：

1. `scripts/vllm_batch_infer.py`
2. `scripts/evaluate_sql_execution_from_report.py`
3. `scripts/analyze_sql_exec_failures.py`

适合回答的问题：

- 模型能否学会完整多轮行为连续性

### D. Agent 单轮版 + GRPO

定义：

- 以 `Agent 单轮 SFT` 模型为初始化
- 用单步 RL / GRPO 继续优化：
  - `state_t -> action_t`
- reward 通过环境执行 SQL 后计算

训练配置：

- `scripts/prepare_grpo_dataset.py`
- `scripts/train_grpo_trl.py`

推荐评测分两层：

1. **同口径验证集评测**
   - `scripts/merge_lora_adapter.py`
   - `scripts/vllm_batch_infer.py`
   - `scripts/evaluate_sql_execution_from_report.py`

2. **整题 rollout 评测**
   - `scripts/evaluate_agent_rollout.py`

适合回答的问题：

- RL 是否继续提升了单轮 agent policy
- RL 是否改善了最终 solution 的质量
- RL 在完整题级 rollout 上的表现如何

## 2. 当前阶段的推荐结论

在你目前已经跑过的实验里：

- Agent 单轮拆分版明显强于整体轨迹版
- 整体轨迹版目前不适合作为主线
- 现在最重要的缺失对照就是普通版 Final-SQL Baseline

所以当前推荐优先级：

1. 普通版 Final-SQL Baseline
2. Agent 单轮拆分版
3. Agent 整体轨迹版
4. Agent 单轮版 + GRPO（在单轮版验证有效后进入）

## 3. 推荐对比指标

### 普通版

- `sql_exact_match_rate`
- `pred_exec_success_rate`
- `result_match_rate`

### Agent 单轮版 / 整体轨迹版

- `protocol_valid_rate`
- `action_type_accuracy`
- `pred_exec_success_rate`
- `result_match_rate`
- `pred_sql_exec_success_rate`
- `pred_solution_exec_success_rate`

### Agent 单轮版 + GRPO

先继续沿用上面的单轮版同口径指标：

- `protocol_valid_rate`
- `action_type_accuracy`
- `pred_exec_success_rate`
- `result_match_rate`
- `pred_solution_exec_success_rate`

再补一组完整 rollout 指标：

- `avg_turns`
- rollout 层面的 `result_match_rate`
- `failure_breakdown`

## 4. 最小对照实验

建议至少跑下面两个版本：

### 版本 1：普通版

- 同一个 base model
- 同样 LoRA 配置
- 数据：`train_final_sql.json`

### 版本 2：Agent 单轮版

- 同一个 base model
- 同样 LoRA 配置
- 数据：`train.json`

然后比较：

- 最终 SQL 可执行率
- 结果正确率
- 如果 agent 版还要额外看协议正确率

如果资源允许，建议补第三组：

### 版本 0：原始 Base Model

- 不做 SFT
- 直接走与普通版相同的 prompt 和评测流程

这样你最后可以得到完整对照：

1. 原始 Base Model
2. 普通版 Final-SQL Baseline
3. Agent 单轮拆分版

## 5. 推荐记录模板

后面你每次实验可以按这个格式记：

```text
实验名：
模型：
数据版本：
训练配置：
评测数据：

离线指标：
- exact match / protocol / action type

执行指标：
- pred_exec_success_rate
- result_match_rate

结论：
```

## 6. 相关脚本

### 数据准备

- `scripts/prepare_final_sql_baseline.py`
- `scripts/prepare_llamafactory_sft.py`

### 离线推理

- `scripts/vllm_batch_infer_final_sql.py`
- `scripts/vllm_batch_infer.py`

### SQL 执行评测

- `scripts/evaluate_final_sql_execution_from_report.py`
- `scripts/evaluate_sql_execution_from_report.py`
- `scripts/analyze_sql_exec_failures.py`

### 多版本对比汇总

- `scripts/compare_experiment_reports.py`

### RL / rollout 评测

- `scripts/evaluate_agent_rollout.py`

## 7. 三版本统一对比命令

当你已经分别得到：

- 普通版 baseline 的离线推理与执行评测
- Agent 单轮版的离线推理、执行评测、失败分析
- Agent 整体轨迹版的离线推理、执行评测、失败分析

可以直接用下面的命令汇总三种版本：

```bash
python3 scripts/compare_experiment_reports.py \
  --experiment "baseline::output/eval_reports/vllm_final_sql_baseline_eval.json::output/eval_reports/vllm_final_sql_baseline_exec_eval.json::" \
  --experiment "agent_single_turn::output/eval_reports/vllm_action_eval.json::output/eval_reports/vllm_action_exec_eval.json::output/eval_reports/vllm_action_exec_failure_analysis.json" \
  --experiment "agent_full_trajectory::output/eval_reports/vllm_full_traj_eval.json::output/eval_reports/vllm_full_traj_exec_eval.json::output/eval_reports/vllm_full_traj_exec_failure_analysis.json" \
  --output-json output/eval_reports/experiment_comparison.json \
  --output-md output/eval_reports/experiment_comparison.md
```

输出会包括：

- JSON 版汇总
- Markdown 表格版汇总

Markdown 表格特别适合直接放到实验记录或汇报文档里。

## 8. 评测 GRPO 的推荐顺序

如果你已经训练出了：

- `SFT merged model`
- `GRPO LoRA adapter`

建议先做：

### 第一步：merge RL LoRA

```bash
python3 scripts/merge_lora_adapter.py \
  --base-model-path /path/to/merged_sft_model \
  --adapter-path /path/to/grpo_adapter \
  --output-path /path/to/grpo_merged_model \
  --dtype bf16 \
  --trust-remote-code \
  --safe-serialization
```

### 第二步：用 vLLM 做同口径验证集评测

```bash
python3 scripts/vllm_batch_infer.py \
  --model-path /path/to/grpo_merged_model \
  --dataset-path output/llamafactory_sft_v5/val.json \
  --output-path output/eval_reports/vllm_grpo_action_eval.json \
  --max-new-tokens 512 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --trust-remote-code
```

```bash
python3 scripts/evaluate_sql_execution_from_report.py \
  --report-path output/eval_reports/vllm_grpo_action_eval.json \
  --output-path output/eval_reports/vllm_grpo_action_exec_eval.json
```

这一步的意义是：

- 和之前 `Agent 单轮 SFT` 完全同口径对比

### 第三步：做整题 rollout 评测

```bash
python3 scripts/evaluate_agent_rollout.py \
  --model-name-or-path /path/to/merged_sft_model \
  --adapter-path /path/to/grpo_adapter \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --output-path output/eval_reports/grpo_rollout_full_gold.json \
  --trust-remote-code
```

这一步的意义是：

- 看完整题级 agent 行为
- 看平均轮数与失败模式
