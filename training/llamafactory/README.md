# LLaMA-Factory SFT 训练说明

本目录提供两套 SFT 训练模板：

- `sft_final_sql_baseline_lora.yaml`
  用于普通版 final-SQL baseline
- `sft_action_focused_lora.yaml`
  用于单轮拆分版数据
- `sft_full_trajectory_lora.yaml`
  用于整体轨迹多轮对话版数据

对应数据目录：

- `output/llamafactory_baselines_v1`
- `output/llamafactory_sft_v5`

## 1. 数据版本说明

### 普通版 Final-SQL Baseline

数据文件：

- `output/llamafactory_baselines_v1/train_final_sql.json`
- `output/llamafactory_baselines_v1/val_final_sql.json`

特点：

- 只学习最终 SQL
- 不学习 `<sql>/<solution>` 两阶段动作
- 适合作为传统 Text2SQL 基线

### 单轮拆分版

数据文件：

- `output/llamafactory_sft_v5/train.json`
- `output/llamafactory_sft_v5/val.json`

特点：

- 每条样本只监督“当前状态 -> 下一步动作”
- 更适合先训一个稳定的 next-action policy

### 整体轨迹版

数据文件：

- `output/llamafactory_sft_v5/train_full_trajectory.json`
- `output/llamafactory_sft_v5/val_full_trajectory.json`

特点：

- 每条样本是一条完整多轮对话轨迹
- observation 以 `user` 消息出现
- action 以 `assistant` 消息出现
- 更适合学多轮行为连续性

## 2. 官方格式依据

LLaMA-Factory 官方文档说明：

- 自定义数据集需要在 `dataset_info.json` 中声明
- SFT 支持 `alpaca` 与 `sharegpt` 格式
- `sharegpt` 支持多轮对话
- OpenAI 风格 `messages` 是 `sharegpt` 的一种特例

参考：

- https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html
- https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html

## 3. 使用方式

假设你已经把对应数据目录下的 `dataset_info.json` 放到 LLaMA-Factory 的 `data/` 目录，且训练数据文件也位于同一数据目录下。

### 训练普通版 Final-SQL Baseline

```bash
llamafactory-cli train training/llamafactory/sft_final_sql_baseline_lora.yaml
```

### 训练单轮拆分版

```bash
llamafactory-cli train training/llamafactory/sft_action_focused_lora.yaml
```

### 训练整体轨迹版

```bash
llamafactory-cli train training/llamafactory/sft_full_trajectory_lora.yaml
```

## 4. 推荐实验顺序

建议先做：

1. 普通版 Final-SQL Baseline
2. 单轮拆分版
3. 整体轨迹版
4. 三者对比

推荐比较指标：

- SQL 可执行率
- 最终结果正确率
- 协议正确率
- 平均 turn 数
- 是否重复 probe

## 5. 注意事项

- `template` 必须和基座模型匹配
- `dataset_dir` 要指向你在 LLaMA-Factory 中实际存放 `dataset_info.json` 的目录
- 如果显存不足，优先调小：
  - `cutoff_len`
  - `per_device_train_batch_size`
  - 增大 `gradient_accumulation_steps`
- 如果你使用 Qwen 系列，请把 `template` 改成对应模板
评估代码
python3 evaluate_llamafactory_sft.py     --model-name-or-path /root/autodl-tmp/qwen2.5-coder-14B     --adapter-path saves/text2sql-agent/action-focused-lora     --dataset-path LlamaFactory/data/val.json   --output-path output/eval_reports/sft_action_eval.json     --max-samples 49     --max-new-tokens 512