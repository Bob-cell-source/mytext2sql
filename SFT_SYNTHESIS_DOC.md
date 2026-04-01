# SFT 合成模块说明文档

## 1. 总体说明

这份文档说明当前新增的多轮 SFT 数据合成模块分别负责什么、输入输出是什么，以及整条数据合成链路是怎么工作的。

当前阶段的目标只有一个：

- 基于现有 72 条 gold 样本，合成多轮、可执行、最终结果正确的 SFT 训练数据

这一阶段只做 SFT 数据合成，不包含 RL 训练。

## 2. 整体流程

当前合成链路如下：

1. 从 `golden_sql_marked.json` 读取 72 条 gold 样本
2. 从 `schema.json` 读取表结构和字段描述
3. 构建 `SeedRecord`
4. 调用 teacher 模型逐轮生成动作
5. 执行 teacher 生成的 SQL
6. 将执行结果整理成 observation
7. 按最终结果是否和 gold result 一致来过滤轨迹
8. 导出多轮 SFT 数据

入口脚本：

- [scripts/synthesize_sft_trajectories.py](/root/text2sql_RL/scripts/synthesize_sft_trajectories.py)

## 3. 输入文件

### 3.1 `golden_sql_marked.json`

作用：

- 作为 seed 数据源

每条样本里当前用到的关键字段：

- `sql_id`：样本 ID
- `question`：自然语言问题
- `table_list`：相关表列表
- `knowledge`：业务知识
- `sql`：gold SQL
- `result`：gold SQL 的执行结果
- `golden_sql`：是否为正确 SQL，当前应为 `true`
- `复杂度`：难度标签

### 3.2 `schema.json`

作用：

- 提供表结构、字段类型、字段描述，用于构建 prompt 中的 schema 片段

顶层结构：

- 一个列表，列表中每一项是一张表

每张表当前使用的字段：

- `table_name`
- `table_description`
- `columns`

每个字段当前使用的内容：

- `col`
- `type`
- `description`

## 4. 输出文件

默认输出目录是：

- `output/sft_synthesis/...`

主要输出有 4 个：

- `seed_records.json`
- `sft_multiturn_full.jsonl`
- `sft_multiturn_action_focused.jsonl`
- `synthesis_report.json`

### 4.1 `seed_records.json`

作用：

- 保存标准化后的 seed 数据
- 方便后续排查问题和复用

### 4.2 `sft_multiturn_full.jsonl`

作用：

- 保存完整轨迹
- 每一条记录是一条完整的多轮轨迹，包含每一步动作和 observation

适合：

- 调试
- 人工抽样检查
- 后续分析 probe 行为

### 4.3 `sft_multiturn_action_focused.jsonl`

作用：

- 保存面向训练的 turn-level SFT 样本
- 每一条记录是一个“当前上下文 -> 下一步动作”的训练样本

每条记录主要包含：

- `seed_id`
- `turn_id`
- `prompt`
- `response`
- `action_type`

这是当前更推荐用于第一版 SFT 的输出。

### 4.4 `synthesis_report.json`

作用：

- 保存本次合成任务的汇总统计

主要包括：

- 总候选轨迹数
- 总保留轨迹数
- 总硬拒绝轨迹数
- 总排序后丢弃轨迹数
- 每个 seed 的候选数
- 每个 seed 的保留数
- 每个 seed 的硬拒绝原因
- 每个 seed 的排序后丢弃分数

## 5. 各代码文件说明

### 5.1 [agent_rl/schemas.py](/root/text2sql_RL/agent_rl/schemas.py)

作用：

- 定义整个 SFT 合成流程中使用的核心数据结构

主要定义了：

- `Observation`
- `TrajectoryTurn`
- `SeedRecord`
- `TrajectoryMeta`
- `TrajectoryRecord`

输入：

- 无直接输入

输出：

- 给其他模块复用的类型定义

### 5.2 [agent_rl/result_matcher.py](/root/text2sql_RL/agent_rl/result_matcher.py)

作用：

- 判断某条 SQL 的执行结果是否和 gold result 一致

当前判断逻辑：

- 忽略行顺序
- 忽略列名
- 保留列值顺序
- 保留重复行数量

主要函数：

- `extract_and_normalize_result`
- `build_result_fingerprint`
- `is_result_match`
- `compare_execution_to_gold`

输入：

- 预测 SQL 的执行结果
- gold SQL 的执行结果

输出：

- 规范化后的结果
- 指纹
- 是否匹配

这个模块是当前“正确性判断”的核心模块。

### 5.3 [agent_rl/seed_builder.py](/root/text2sql_RL/agent_rl/seed_builder.py)

作用：

- 从 gold 数据和 schema 元数据中构建标准化 seed

主要函数：

- `load_json`
- `build_schema_index`
- `build_schema_snippets`
- `build_seed_records`
- `dump_seed_records`

输入：

- `golden_sql_marked.json`
- `schema.json`

输出：

- `SeedRecord` 列表
- 可选导出的 `seed_records.json`

生成的 seed 里主要包含：

- `seed_id`
- `question`
- `table_list`
- `knowledge`
- `difficulty`
- `gold_sql`
- `gold_result`
- `gold_result_fingerprint`
- `schema_snippets`
- `hard_constraints`

### 5.4 [agent_rl/sql_env.py](/root/text2sql_RL/agent_rl/sql_env.py)

作用：

- 执行 SQL
- 把执行结果整理成适合 agent 使用的 observation

核心类：

- `SQLEnvironment`

主要方法：

- `execute`
- `execute_with_rows`

核心行为：

- 只允许只读查询，限制为 `SELECT` / `WITH`
- 拒绝危险 SQL 关键字
- 复用现有 [sql_exe.py](/root/text2sql_RL/sql_exe.py) 执行器
- 把原始执行结果压缩成精简 observation

输入：

- 一条 SQL
- `turns_left`
- 可选的请求 ID

输出：

- `Observation`
- 或者 `Observation + 全量结果行`

当前 observation 结构如下：

```json
{
  "status": "success|empty|error|timeout",
  "error_message": "",
  "columns": [],
  "sample_rows": [],
  "row_count": 0,
  "fingerprint": "",
  "turns_left": 0
}
```

数据库配置来源：

- `DB_HOST`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `DB_PORT`

### 5.5 [agent_rl/prompt_builder.py](/root/text2sql_RL/agent_rl/prompt_builder.py)

作用：

- 负责构建 teacher 输入 prompt
- 负责把 action 渲染成统一协议格式

主要内容：

- `SYSTEM_INSTRUCTION`
- `build_teacher_user_prompt`
- `render_assistant_action`

当前 teacher 输出协议：

如果继续诊断：

```text
<reasoning>...</reasoning>
<sql>...</sql>
```

如果提交最终答案：

```text
<reasoning>...</reasoning>
<solution>...</solution>
```

输入：

- `SeedRecord`
- 历史 turn
- `turns_left`

输出：

- teacher user prompt
- assistant action 文本

### 5.6 [agent_rl/teacher_rollout.py](/root/text2sql_RL/agent_rl/teacher_rollout.py)

作用：

- 让 teacher 模型在多轮环境里逐轮决策

主要内容：

- `ParsedAction`
- `parse_teacher_output`
- `get_teacher_client_and_model`
- `TeacherRolloutRunner`

teacher 模型配置来源：

- `OPENAI_API_KEY`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `LLM_MODEL`

兜底逻辑：

- 如果环境变量没配，会尝试回退到 [final_code.py](/root/text2sql_RL/final_code.py#L145) 里的 `API_KEY`

rollout 逻辑：

1. 根据当前状态构建 prompt
2. 调 teacher 生成下一步动作
3. 解析 `<reasoning>` + `<sql|solution>`
4. 执行 SQL
5. 记录 observation
6. 达到最终答案、错误阈值或轮数上限后停止

输入：

- 单个 `SeedRecord`

输出：

- 单条 `TrajectoryRecord`

注意：

- 最终轨迹是否有效，核心看 final solution 的执行结果是否和 gold result 匹配

### 5.7 [agent_rl/trajectory_filter.py](/root/text2sql_RL/agent_rl/trajectory_filter.py)

作用：

- 对 teacher 生成的候选轨迹做过滤和排序

主要函数：

- `evaluate_trajectory`
- `filter_and_rank_trajectories`

硬过滤条件：

- 空轨迹
- 超过最大轮数
- 没有最终 `solution`
- 最终结果不匹配
- 发生 timeout

软排序信号：

- 轨迹越短越好
- 信息增益越高越好
- probe 重复越少越好
- 执行错误越少越好

输入：

- 候选轨迹列表

输出：

- 保留轨迹列表
- 拒绝轨迹列表

### 5.8 [agent_rl/dataset_packer.py](/root/text2sql_RL/agent_rl/dataset_packer.py)

作用：

- 把筛选后的轨迹导出成训练用文件和报告文件

主要函数：

- `pack_full_trajectories`
- `pack_action_focused_samples`
- `write_synthesis_report`

输入：

- 保留轨迹
- seed 查找表
- 汇总统计

输出：

- `sft_multiturn_full.jsonl`
- `sft_multiturn_action_focused.jsonl`
- `synthesis_report.json`

### 5.9 [scripts/synthesize_sft_trajectories.py](/root/text2sql_RL/scripts/synthesize_sft_trajectories.py)

作用：

- 整个 SFT 合成流程的命令行入口

支持的参数：

- `--golden`
- `--schema`
- `--outdir`
- `--seed-limit`
- `--attempts`
- `--max-turns`
- `--keep-top-k`
- `--max-preview-rows`

执行流程：

1. 构建 seeds
2. 初始化 SQL 执行环境
3. 初始化 teacher rollout runner
4. 为每个 seed 生成多条候选轨迹
5. 过滤并排序
6. 写出完整轨迹、训练样本和报告

## 6. 关键数据结构

### 6.1 SeedRecord

```json
{
  "seed_id": "sql_1",
  "question": "...",
  "table_list": ["table_a", "table_b"],
  "knowledge": "...",
  "difficulty": "中等",
  "gold_sql": "SELECT ...",
  "gold_result": [],
  "gold_result_fingerprint": "...",
  "schema_snippets": ["...", "..."],
  "hard_constraints": "..."
}
```

### 6.2 TrajectoryRecord

```json
{
  "seed_id": "sql_1",
  "question": "...",
  "schema_snippets": ["..."],
  "hard_constraints": "...",
  "knowledge": "...",
  "turns": [
    {
      "turn_id": 1,
      "action_type": "sql",
      "reasoning": "...",
      "sql": "SELECT ...",
      "observation": {
        "status": "success",
        "error_message": "",
        "columns": ["col1"],
        "sample_rows": [{"col1": "x"}],
        "row_count": 1,
        "fingerprint": "...",
        "turns_left": 2
      }
    }
  ],
  "meta": {
    "final_status": "success",
    "final_result_match": true,
    "turn_count": 2,
    "timeout_count": 0,
    "error_count": 0,
    "repeated_probe_count": 0,
    "info_gain_score": 1.0,
    "score": 5.0,
    "reject_reason": ""
  }
}
```

## 7. 运行依赖

Python 依赖：

- `openai`
- `pymysql`

外部依赖：

- teacher 模型 API
- 已启动并已导入业务数据的 StarRocks 数据库

建议配置的环境变量：

```bash
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen3-max"

export DB_HOST="127.0.0.1"
export DB_PORT="9030"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="TGAC"
```

## 8. 运行示例

### 8.1 冒烟测试

```bash
python3 scripts/synthesize_sft_trajectories.py \
  --seed-limit 2 \
  --attempts 2 \
  --max-turns 3 \
  --keep-top-k 1 \
  --outdir output/sft_synthesis_smoke
```

### 8.2 小规模测试

```bash
python3 scripts/synthesize_sft_trajectories.py \
  --seed-limit 10 \
  --attempts 6 \
  --max-turns 3 \
  --keep-top-k 3 \
  --outdir output/sft_synthesis_test
```

## 9. 当前限制

- 当前最核心的正确性判断仍然是 final result match
- 领域规则评分目前还是启发式的，还不算很强
- SQL 执行仍然复用了现有 [sql_exe.py](/root/text2sql_RL/sql_exe.py) 的文件式执行器
- `timeout` 字段已经预留，但当前底层执行器主要返回 `success`、`empty`、`error`
- 当前模块只负责 SFT 合成，不包含训练脚本和 RL 逻辑

## 10. Agent 数据构造约束

这一节用于约束多轮 agent 数据的时序关系，避免构造出“看到了未来信息”的错误样本。

### 10.1 核心状态定义

在多轮 agent 任务里，建议始终使用下面这套定义：

- `state_t`：第 `t` 步决策前，agent 当前可见的全部信息
- `action_t`：第 `t` 步生成的动作
- `observation_t`：执行 `action_t` 之后返回的反馈

推荐的监督关系是：

- 输入：`state_t`
- 输出：`action_t`

而不是：

- 输入：`state_t + observation_t`
- 输出：`action_t`

后者会把“当前动作执行后的结果”错误地提前泄露给模型。

### 10.2 当前推荐时序

当前推荐的数据构造关系是：

- `state_t = question + schema + knowledge + history_actions[:t-1] + observation_(t-1)`
- `target = action_t`

也就是：

- 当前输入里最多只能看到上一轮 observation
- 当前输出只能是下一步 action
- 当前动作执行后的 observation 不能出现在当前输入里

### 10.3 turn 1 的特殊规则

对于第一轮动作：

- `history_actions = []`
- `latest_observation = null`

这是因为第一轮之前还没有执行过任何动作，所以不应该凭空出现 observation。

### 10.4 observation 的归属

必须明确：

- `observation_t` 只属于 `action_t`
- `observation_t` 应该作为下一条样本中 `state_(t+1)` 的一部分

不能把：

- `action_t`
- `observation_t`

一起放到同一条监督样本中作为“输入 -> 输出”的配对。

### 10.5 一致性检查规则

以后只要构造 turn-level agent 数据，建议至少检查下面 4 条：

1. 输入里是否只包含“过去已经发生的信息”
2. 输出是否真的是“下一步动作”
3. observation 是否明确属于上一轮 action
4. 抽一条样本问自己：
   “如果我是 agent，在做这一步之前，真的能看到这些信息吗？”

如果第 4 条的答案不是明确的“能”，这条样本就存在时序问题。

### 10.6 常见错误示例

错误示例：

- `turn_id = 1`
- `history_actions = []`
- 但输入里已经带了一个非空 observation

这类样本的问题是：

- observation 没有来源
- 时序不一致
- 会把模型教成“没有历史动作也会凭空拿到反馈”

另一个常见错误是：

- 输入里的 observation 已经说明了某个不确定点
- 输出动作却还是重复去查同一件事

这会让模型学会重复 probe，降低 agent 行为质量。

### 10.7 当前仓库中的对应关系

当前仓库里建议区分两类 SFT 数据：

- `train.json` / `val.json`
  作用：使用拼接后的长 prompt 作为输入，适合快速启动训练

- `train_observation_explicit.json` / `val_observation_explicit.json`
  作用：显式拆出 observation、history 和当前状态，更接近 agent 状态建模

其中显式 observation 版本必须严格遵守本节的时序约束。
