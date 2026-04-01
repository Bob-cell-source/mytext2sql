# text2sql_RL

一个面向 StarRocks / MySQL 场景的 Text2SQL agent 项目。当前重点是：

- 基于 gold SQL 样本合成多轮 SFT 数据
- 训练能够进行多轮验证 / 探索并最终提交 SQL 的 agent 模型
- 后续再考虑 RL 阶段

当前仓库已经完成：

- 多轮 SFT 轨迹合成
- 结果匹配与轨迹过滤
- LLaMA-Factory 训练数据准备
- 单轮版与整体轨迹版两套 SFT 数据导出

## 1. 项目目标

和传统“一问一答”式 Text2SQL 不同，这个项目更偏向 agent 形式：

- 中间可以生成验证 / 探索型 SQL
- 利用 SQL 执行反馈更新状态
- 在信息足够时提交最终 `solution`

当前阶段只做：

- 多轮 SFT 数据构造
- SFT 训练准备

尚未正式进入：

- RL reward 设计
- RL 训练流程

## 2. 主要文件结构

### 核心脚本

- [final_code.py](/root/text2sql_RL/final_code.py)
  原始单体 SQL 生成脚本

- [sql_exe.py](/root/text2sql_RL/sql_exe.py)
  SQL 执行器，当前通过 PyMySQL 执行查询

- [generate_sft_dpo_data.py](/root/text2sql_RL/generate_sft_dpo_data.py)
  旧版 SFT / DPO 数据生成脚本

### 新增 agent SFT 模块

- [agent_rl/schemas.py](/root/text2sql_RL/agent_rl/schemas.py)
- [agent_rl/result_matcher.py](/root/text2sql_RL/agent_rl/result_matcher.py)
- [agent_rl/seed_builder.py](/root/text2sql_RL/agent_rl/seed_builder.py)
- [agent_rl/sql_env.py](/root/text2sql_RL/agent_rl/sql_env.py)
- [agent_rl/prompt_builder.py](/root/text2sql_RL/agent_rl/prompt_builder.py)
- [agent_rl/teacher_rollout.py](/root/text2sql_RL/agent_rl/teacher_rollout.py)
- [agent_rl/trajectory_filter.py](/root/text2sql_RL/agent_rl/trajectory_filter.py)
- [agent_rl/dataset_packer.py](/root/text2sql_RL/agent_rl/dataset_packer.py)

### 数据合成与训练准备脚本

- [scripts/synthesize_sft_trajectories.py](/root/text2sql_RL/scripts/synthesize_sft_trajectories.py)
  合成多轮 SFT 轨迹

- [scripts/prepare_llamafactory_sft.py](/root/text2sql_RL/scripts/prepare_llamafactory_sft.py)
  清洗 SFT 数据并导出为 LLaMA-Factory 可训练格式

### 说明文档

- [SFT_SYNTHESIS_DOC.md](/root/text2sql_RL/SFT_SYNTHESIS_DOC.md)
  SFT 合成模块详细说明

- [ENVIRONMENT_SETUP.md](/root/text2sql_RL/ENVIRONMENT_SETUP.md)
  环境配置清单

- [training/llamafactory/README.md](/root/text2sql_RL/training/llamafactory/README.md)
  LLaMA-Factory 训练说明

## 3. 数据文件

### 主要输入

- [golden_sql_marked.json](/root/text2sql_RL/golden_sql_marked.json)
  72 条 gold 样本，包含 question / gold SQL / gold result

- [schema.json](/root/text2sql_RL/schema.json)
  表结构、字段类型和描述信息

- [insert_sql.json](/root/text2sql_RL/insert_sql.json)
  表示例数据

### 主要输出

合成过程输出到：

- `output/sft_synthesis_*`

训练准备输出到：

- `output/llamafactory_sft_*`

## 4. 环境安装

基础依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

训练相关依赖：

```bash
pip install -r requirements-train.txt
```

完整环境说明见：

- [ENVIRONMENT_SETUP.md](/root/text2sql_RL/ENVIRONMENT_SETUP.md)

## 5. 环境变量

### Teacher 模型

```bash
export DASHSCOPE_API_KEY="你的key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen3-max"
```

### 数据库

```bash
export DB_HOST="127.0.0.1"
export DB_PORT="9030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

## 6. SFT 轨迹合成

示例：

```bash
python3 scripts/synthesize_sft_trajectories.py \
  --attempts 16 \
  --max-turns 3 \
  --keep-top-k 4 \
  --workers 4 \
  --outdir output/sft_synthesis_full_v2
```

说明：

- 支持按 seed checkpoint 落盘
- 支持中断后续跑
- 支持 `--workers` 并发处理 seed

## 7. 准备 LLaMA-Factory 数据

示例：

```bash
python3 scripts/prepare_llamafactory_sft.py \
  --input output/sft_synthesis_full_v2/sft_multiturn_action_focused.jsonl \
  --full-input output/sft_synthesis_full_v2/sft_multiturn_full.jsonl \
  --outdir output/llamafactory_sft_v5
```

会输出三套数据：

### 单轮拆分版

- `train.json`
- `val.json`

### 单轮显式 observation 版

- `train_observation_explicit.json`
- `val_observation_explicit.json`

### 整体轨迹版

- `train_full_trajectory.json`
- `val_full_trajectory.json`

## 8. LLaMA-Factory 训练

训练模板在：

- [training/llamafactory/sft_action_focused_lora.yaml](/root/text2sql_RL/training/llamafactory/sft_action_focused_lora.yaml)
- [training/llamafactory/sft_full_trajectory_lora.yaml](/root/text2sql_RL/training/llamafactory/sft_full_trajectory_lora.yaml)

训练说明见：

- [training/llamafactory/README.md](/root/text2sql_RL/training/llamafactory/README.md)

示例：

```bash
llamafactory-cli train training/llamafactory/sft_action_focused_lora.yaml
llamafactory-cli train training/llamafactory/sft_full_trajectory_lora.yaml
```

## 9. 当前训练建议

推荐先做两个版本对照：

1. 单轮拆分版 SFT
2. 整体轨迹版 SFT

重点比较：

- 协议正确率
- SQL 可执行率
- 最终结果正确率
- 平均 turn 数
- 是否出现重复 probe

## 10. 当前状态

当前仓库适合做的事情：

- 合成多轮 SFT 数据
- 准备单轮和整体轨迹版训练集
- 用 LLaMA-Factory 做 SFT 对照实验

当前还没有正式实现的内容：

- 完整 RL 环境训练
- reward 函数系统化设计
- rollout 评测闭环自动化

## 11. 后续方向

后续建议优先级：

1. 完成单轮版与整体轨迹版 SFT 训练
2. 做 rollout 对比评测
3. 分析 reasoning / observation 显式建模的收益
4. 再决定 RL 阶段如何设计
