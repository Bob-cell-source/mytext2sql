# TRL / GRPO 训练说明

本目录现在维护两条 RL 路线：

1. **online multi-turn GRPO**
   - 当前推荐主线
   - 从初始题目 seed 出发
   - 通过 `rollout_func` 在线 rollout 整条轨迹
2. **single-step GRPO**
   - 旧版兼容方案
   - 从中间 state pool 训练“下一步动作”

当前更推荐第一条，因为它更符合你这个项目的 agent RL 目标。

## 0. 在线版和单步版的区别

### online multi-turn GRPO

- 输入：初始 seed 池
- 训练时：模型自己从 turn1 开始一步步生成
- 环境：真实执行 SQL，返回 observation
- 优化目标：整条 episode 的 reward

核心脚本：

- `scripts/prepare_online_grpo_seeds.py`
- `agent_rl/online_rollout.py`
- `scripts/train_grpo_online_trl.py`

### single-step GRPO

- 输入：从 SFT 多轮轨迹恢复出来的中间 state
- 训练时：模型只生成当前一步 action
- 优化目标：局部 action reward

核心脚本：

- `scripts/prepare_grpo_dataset.py`
- `scripts/train_grpo_trl.py`

如果你只是想继续复现实验，single-step 版还能用；如果你想更贴近真正的 agent RL，优先看 online 版。

## 1. 先准备 online seed 池

online 版不再准备中间 state，而是只保留初始题目 seed。

执行：

```bash
python3 scripts/prepare_online_grpo_seeds.py \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --outdir output/rl_seed_pool_v2
```

输出：

- `output/rl_seed_pool_v2/train_rl_seeds.json`
- `output/rl_seed_pool_v2/val_rl_seeds.json`
- `output/rl_seed_pool_v2/prepare_report.json`

每条样本只保留：

- `seed_id`
- 初始 `prompt`
- `difficulty`
- `seed_json`
- `gold_sql`
- `gold_result_json`

当前默认会直接从全部 gold seeds 划分：

- train: 58
- val: 14

验证集保持 `gold-only`。

如果你只是想先做方法可行性验证，也可以直接：

```bash
python3 scripts/prepare_online_grpo_seeds.py \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --val-ratio 0 \
  --outdir output/rl_seed_pool_v2_all_gold
```

这时会得到：

- train = 全量 gold
- val = 0

这种设定适合做 sanity check，看 online GRPO 是否能在训练集内把策略往正确方向推；不适合作为正式泛化结论。

如果你要把合成 / Dataflow 数据并入训练池，可以加：

```bash
python3 scripts/prepare_online_grpo_seeds.py \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --synthetic-path output/dataflow_evalsets/tgac_step9_eval.json \
  --outdir output/rl_seed_pool_v2_aug
```

此时：

- train = gold train + synthetic train
- val = 仍然只用 gold val

## 2. online multi-turn GRPO 启动命令

### 2.1 已 merge 的 SFT 模型

如果你已经把 `SFT` 结果 merge 成完整模型目录，推荐这样启动：

```bash
python3 scripts/train_grpo_online_trl.py \
  --base-model-path /path/to/merged_sft_model \
  --train-dataset-path output/rl_seed_pool_v2/train_rl_seeds.json \
  --eval-dataset-path output/rl_seed_pool_v2/val_rl_seeds.json \
  --output-dir saves/grpo_online/online_14b \
  --dtype bf16 \
  --trust-remote-code \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --num-generations 2 \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --gradient-checkpointing \
  --use-rl-lora \
  --rl-lora-r 64 \
  --rl-lora-alpha 128 \
  --rl-lora-dropout 0.05 \
  --rl-lora-target-modules all-linear
```

### 2.2 未 merge 的 SFT 模型

如果你手上还是：

- `base model`
- `SFT adapter`

可以这样启动：

```bash
python3 scripts/train_grpo_online_trl.py \
  --base-model-path /path/to/base_model \
  --sft-adapter-path /path/to/sft_adapter \
  --train-dataset-path output/rl_seed_pool_v2/train_rl_seeds.json \
  --eval-dataset-path output/rl_seed_pool_v2/val_rl_seeds.json \
  --output-dir saves/grpo_online/online_14b \
  --dtype bf16 \
  --trust-remote-code \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --num-generations 2 \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --gradient-checkpointing
```

说明：

- 这版训练依赖 TRL 的 `rollout_func`
- 训练时不再使用静态中间 state pool
- reward 直接来自整条 episode 的在线 rollout

## 3. online 版的核心代码

- `agent_rl/online_rollout.py`

作用：

- 从初始 seed 开始在线 rollout
- 每一轮构造 prompt
- 调模型生成 completion
- 调 `Text2SQLRLEnv.step(...)` 执行 SQL
- 聚合整条 episode 的 token、logprobs 和 env reward

- `scripts/train_grpo_online_trl.py`

作用：

- 读取初始 seed 池
- 构造 `rollout_func`
- 将在线 rollout 结果接入 `GRPOTrainer`
- 记录中文时间统计

## 4. 旧版 single-step GRPO

下面这些仍然保留，但现在是旧方案：

- `scripts/prepare_grpo_dataset.py`
- `scripts/train_grpo_trl.py`
- `output/rl_training_inputs_v1/`

这条线的好处是更容易先跑通；缺点是训练目标是局部 action，不是完整 episode。

---

以下内容保留给 single-step 方案与通用说明。

## 5. 先准备 single-step GRPO 数据

GRPO 数据来自 `v5` 的整体轨迹数据，但会被拆成单步 RL 样本。

每条样本包含：

- 当前 `system_prompt`
- 当前 `user_prompt`
- 当前 `history`
- 对应的 `seed`
- gold output / gold sql

执行：

```bash
python3 scripts/prepare_grpo_dataset.py \
  --train-full-trajectory output/llamafactory_sft_v5/train_full_trajectory.json \
  --val-full-trajectory output/llamafactory_sft_v5/val_full_trajectory.json \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --outdir output/rl_training_inputs_v1
```

输出：

- `output/rl_training_inputs_v1/train_rl_single_step.json`
- `output/rl_training_inputs_v1/val_rl_single_step.json`
- `output/rl_training_inputs_v1/prepare_report.json`

## 6. 推荐的第一版训练方式

第一版建议：

- 如果你还保留“base model + SFT adapter”两段式：
  - 可以继续从 `SFT adapter` 初始化
- 如果你已经把 `SFT` 模型 merge 成完整模型：
  - 推荐在 merged 模型上再挂一个新的 `RL LoRA adapter`
- 不要同卡再挂 `vLLM`

## 7. 推荐启动命令

### 3.1 未 merge 的 SFT 模型

如果你手上还是：

- `base model`
- `SFT adapter`

可以这样启动：

```bash
python3 scripts/train_grpo_trl.py \
  --base-model-path /path/to/base_model \
  --sft-adapter-path /path/to/sft_adapter \
  --train-dataset-path output/rl_training_inputs_v1/train_rl_single_step.json \
  --eval-dataset-path output/rl_training_inputs_v1/val_rl_single_step.json \
  --output-dir saves/grpo/single_step_14b \
  --dtype bf16 \
  --trust-remote-code \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --num-generations 2 \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --gradient-checkpointing \
```

### 3.2 已 merge 的 SFT 模型

如果你已经把 `SFT` 结果 merge 成一个完整模型目录，推荐这样启动：

```bash
python3 scripts/train_grpo_trl.py \
  --base-model-path /path/to/merged_sft_model \
  --train-dataset-path output/rl_training_inputs_v1/train_rl_single_step.json \
  --eval-dataset-path output/rl_training_inputs_v1/val_rl_single_step.json \
  --output-dir saves/grpo/single_step_14b \
  --dtype bf16 \
  --trust-remote-code \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --num-generations 2 \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --gradient-checkpointing \
  --use-rl-lora \
  --rl-lora-r 64 \
  --rl-lora-alpha 128 \
  --rl-lora-dropout 0.05 \
  --rl-lora-target-modules all-linear
```

说明：

- 这时不要再传 `--sft-adapter-path`
- `--use-rl-lora` 表示在 merged 的 SFT 模型上再挂一层新的 RL LoRA adapter
- 这样比“直接训练 merge 后全模型”更稳，也更省显存

## 8. 参数建议

对你现在的 `14B + RTX PRO 6000 96GB`，第一版建议：

- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 4~8`
- `num_generations = 2~4`
- `max_prompt_length = 2048~3072`
- `max_completion_length = 512`

不要一开始：

- 开太大的 `num_generations`
- 上很长 context
- 同时在同一张卡上跑 vLLM 和训练

## 9. 当前 reward 逻辑

当前 reward 由 `agent_rl/reward.py` 和 `agent_rl/rl_env.py` 决定，核心是：

- 最终结果完全匹配：主奖励
- 第一次成功可执行 probe：小正奖励
- probe 失败：负奖励
- 超过难度允许的免费 probe 数：turn penalty
- 协议错误：终止并惩罚
- 最后一轮仍输出 `<sql>`：按 final failure 处理

## 10. 当前方案的边界

这版不是“真正多步 end-to-end policy optimization trainer”，而是：

- 通过单步 state/action 的方式，把当前 RL 问题包装成 TRL 可直接训练的形式

它适合：

- 先验证 RL 是否有增益
- 先验证 reward 方向是否合理

它不适合：

- 一上来追求最复杂的 multi-step on-policy agent RL

## 11. 训练后建议

训练后优先比较：

- `pred_exec_success_rate`
- `result_match_rate`
- `pred_solution_exec_success_rate`
- `action_type_accuracy`

如果这些指标相对当前 `Agent 单轮 SFT` 有提升，再继续扩大 GRPO 训练规模。

## 11.1 训练后怎么评估 GRPO 模型

建议分成两层。

### 第一层：和之前 SFT 完全同口径对比

如果你的 GRPO 训练结果是一个 RL LoRA adapter，推荐先 merge：

```bash
python3 scripts/merge_lora_adapter.py \
  --base-model-path /path/to/merged_sft_model \
  --adapter-path /path/to/grpo_adapter \
  --output-path /path/to/grpo_merged_model \
  --dtype bf16 \
  --trust-remote-code \
  --safe-serialization
```

然后用 vLLM 在和之前相同的 `val.json` 上评测：

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

再做 SQL 执行评测：

```bash
python3 scripts/evaluate_sql_execution_from_report.py \
  --report-path output/eval_reports/vllm_grpo_action_eval.json \
  --output-path output/eval_reports/vllm_grpo_action_exec_eval.json
```

这样你就可以把：

- `Agent 单轮 SFT`
- `Agent 单轮 SFT + GRPO`

放在同一套指标下直接比较。

### 第二层：整题 rollout 评测

如果你还想看完整题级 agent 表现，再跑：

```bash
python3 scripts/evaluate_agent_rollout.py \
  --model-name-or-path /path/to/merged_sft_model \
  --adapter-path /path/to/grpo_adapter \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --output-path output/eval_reports/grpo_rollout_full_gold.json \
  --trust-remote-code
```

它会统计：

- rollout 层面的 `result_match_rate`
- `avg_turns`
- `failure_breakdown`

建议先做第一层，再做第二层。

## 12. 服务器环境准备

如果你要在远端服务器上运行这套 GRPO 代码，建议按下面顺序准备。

### 8.1 代码与数据

确保服务器上已经有：

- 当前项目代码目录
- `golden_sql_marked.json`
- `schema.json`
- `output/llamafactory_sft_v5/`
- `14B` base model
- 已训练好的 `SFT adapter`

### 8.2 Python 环境

建议单独创建一个环境：

```bash
conda create -n grpo python=3.10 -y
conda activate grpo
```

先安装 PyTorch（按服务器 CUDA 版本选择），再安装项目依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
```

可以用下面命令做一次导入检查：

```bash
python3 - <<'PY'
import torch, transformers, datasets, peft, trl, accelerate, pymysql
print("ok")
PY
```

### 8.3 数据库访问

注意：

- `scripts/prepare_grpo_dataset.py` 不需要数据库
- `scripts/train_grpo_trl.py` 需要数据库

因为 reward function 会真实执行 SQL。

如果数据库在本地机器，可以继续使用之前的反向隧道方案。

本地机器执行：

```bash
docker start quickstart
ssh -CNg -R 19030:127.0.0.1:9030 -p 12133 root@connect.westd.seetacloud.com
```

服务器上设置：

```bash
export DB_HOST="127.0.0.1"
export DB_PORT="19030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

如果数据库本来就部署在服务器或内网，只要把上面的环境变量改成实际值即可。

## 9. 服务器运行顺序

### 9.1 先准备 GRPO 数据

```bash
python3 scripts/prepare_grpo_dataset.py \
  --train-full-trajectory output/llamafactory_sft_v5/train_full_trajectory.json \
  --val-full-trajectory output/llamafactory_sft_v5/val_full_trajectory.json \
  --golden-path golden_sql_marked.json \
  --schema-path schema.json \
  --outdir output/rl_training_inputs_v1
```

准备完成后，检查：

- `output/rl_training_inputs_v1/train_rl_single_step.json`
- `output/rl_training_inputs_v1/val_rl_single_step.json`
- `output/rl_training_inputs_v1/prepare_report.json`

### 9.2 先做 smoke test

第一轮建议不要直接全量训练，先用小样本验证：

```bash
python3 scripts/train_grpo_trl.py \
  --base-model-path /path/to/merged_sft_model \
  --train-dataset-path output/rl_training_inputs_v1/train_rl_single_step.json \
  --eval-dataset-path output/rl_training_inputs_v1/val_rl_single_step.json \
  --output-dir saves/grpo/smoke_single_step_14b \
  --dtype bf16 \
  --trust-remote-code \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --num-generations 2 \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --gradient-checkpointing \
  --use-rl-lora \
  --rl-lora-r 64 \
  --rl-lora-alpha 128 \
  --rl-lora-dropout 0.05 \
  --rl-lora-target-modules all-linear \
  --max-train-samples 32 \
  --max-eval-samples 8
```

这一步主要检查：

- 数据库连通是否正常
- reward function 是否能稳定返回分数
- 协议解析是否正常
- 显存是否足够
- trainer 是否能正常进入训练 step

### 9.3 再跑正式版

smoke test 没问题后，再跑正式版：

```bash
python3 scripts/train_grpo_trl.py \
  --base-model-path /path/to/merged_sft_model \
  --train-dataset-path output/rl_training_inputs_v1/train_rl_single_step.json \
  --eval-dataset-path output/rl_training_inputs_v1/val_rl_single_step.json \
  --output-dir saves/grpo/single_step_14b \
  --dtype bf16 \
  --trust-remote-code \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --num-train-epochs 1 \
  --num-generations 2 \
  --max-prompt-length 3072 \
  --max-completion-length 512 \
  --logging-steps 5 \
  --save-steps 50 \
  --eval-steps 50 \
  --gradient-checkpointing \
  --use-chat-template \
  --use-rl-lora \
  --rl-lora-r 64 \
  --rl-lora-alpha 128 \
  --rl-lora-dropout 0.05 \
  --rl-lora-target-modules all-linear
```

训练输出会保存在：

- `saves/grpo/single_step_14b/`

最终模型通常会在：

- `saves/grpo/single_step_14b/final`

## 10. 服务器阶段的建议

第一版不要一上来做这些事：

- 不要同卡再挂 vLLM 在线生成
- 不要把 `num_generations` 开得太大
- 不要把 `max_prompt_length` 拉得太长
- 不要先追求多步完整 agent RL

当前阶段的目标是：

- 先验证单步 GRPO 是否能提升
  - `pred_exec_success_rate`
  - `result_match_rate`
  - `pred_solution_exec_success_rate`

如果这些指标有提升，再继续扩大训练规模或升级到更完整的多步 RL。

## 11. 训练耗时统计

当前 `scripts/train_grpo_trl.py` 已经支持输出中文时间统计，并默认写到：

- `输出目录/timing_metrics.jsonl`

每次到 `logging_steps` 时会记录：

- 平均每个训练步总耗时
- 平均每次奖励函数耗时
- 平均每个环境步骤耗时
- 平均每次 SQL 执行耗时
- 近似模型生成与训练耗时

如果你想显式指定文件路径，可以传：

```bash
--timing-log-path saves/grpo/single_step_14b/timing_metrics.jsonl
```

补充：

- 如果 merged 模型目录里的 tokenizer 没有 `chat_template`
- 就不要加 `--use-chat-template`
- 当前仓库默认导出的 `prompt` 已经是完整纯文本，可直接训练
