# RL 设计方案

这份文档用于定义当前项目基于 `Agent 单轮 SFT` 的后续 RL / GRPO 方案。

目标不是一步到位实现最复杂的 RL 系统，而是先把：

- 状态定义
- 动作定义
- 终止条件
- 奖励函数
- 难度驱动的 turn budget

全部收敛成一版可实现、可调试、可逐步扩展的方案。

## 1. RL 阶段的起点

RL 不从 base model 开始，而是从当前已经验证有效的：

- `Agent 单轮 SFT 模型`

开始继续优化。

这意味着：

- SFT 负责提供稳定的 agent 冷启动
- RL 负责在此基础上进一步提升：
  - 最终结果正确率
  - 多轮探索效率
- final solution 质量
  - 自纠能力

## 1.1 当前采用的 RL 形态

当前推荐主线已经切到：

- **online multi-turn GRPO**
- 通过 `rollout_func` 在线生成整条轨迹
- 从初始 seed 出发，而不是从中间 state 出发

仓库里仍然保留 single-step GRPO 作为旧实验线，用于做对照和兼容。

## 1.1.1 online 版

online 版的训练样本只是一批初始 seed：

- `question`
- `schema`
- `knowledge`
- `difficulty`
- `gold_sql`
- `gold_result`

训练时：

1. `env.reset(seed)`
2. 模型生成第 1 步 action
3. env 执行并返回 observation
4. 继续下一轮直到 `<solution>` 或终止
5. 按整条 episode 计算 reward

这条线更符合当前项目对 agent RL 的目标。

## 1.1.2 single-step 旧版

仓库里也保留了一条更早的 single-step 方案：

- `单步 state -> 单步 action`
- `reward function` 内重建当前 state
- 再调用环境做一步 `env.step(...)`

这条线的优点是简单，缺点是并不真正优化整条多轮策略。

## 1.2 当前方案与未来多步 agent RL 的区别

### 当前推荐方案：online rollout_func GRPO

训练样本形式：

- 输入：初始 seed
- 输出：在线 rollout 的整条 episode
- reward：按整条 episode 汇总

特点：

- 最贴合当前 `<reasoning>/<sql>/<solution>` 协议
- 最大化复用现有 env / reward / SQL 执行逻辑
- 不需要先把项目改造成标准 tool calling

适用阶段：

- 当前主线

### 未来方案：OpenEnv

如果后面愿意进一步按 TRL 的环境范式重构，则可以走 OpenEnv：

- 把环境交给 TRL 管理
- 更接近官方 agent 训练接口
- 但要求更贴近环境方法 / tools 规范

这时更适合：

- `TRL` 的 `OpenEnv` 集成

特点：

- 更官方
- 更规范
- 但需要更多重构

代价：

- 实现复杂很多
- 调试难度显著上升
- 如果还接入 vLLM 在线生成，才需要认真考虑：
  - `Truncated Importance Sampling`
  - `Masked Importance Sampling`
  - 训练/推理精度不一致带来的校正问题

## 1.3 当前阶段的推荐策略

当前推荐路线是：

1. 先把 **online rollout_func GRPO** 跑通
2. 先看核心任务指标是否提升
3. 如果 online 版已经稳定，再考虑是否进一步迁移到 OpenEnv

也就是说：

- 当前优先目标不是追求最完整的 RL 系统
- 而是先验证：
  - reward 是否有方向性
  - RL 是否真的优于纯 SFT

## 1.4 当前 RL 相关代码文件说明

当前仓库里和 RL / GRPO 直接相关的代码主要有下面这些。

### 核心环境与奖励

- `agent_rl/reward.py`

作用：

- 定义 difficulty 对应的 turn budget
- 定义第一版 reward 公式
- 计算：
  - `r_exec_step`
  - `r_turn`
  - `r_final_result`
  - `r_sql_ngram`
  - `r_final_exec_fail`

这是 reward 逻辑的核心入口。

- `agent_rl/rl_env.py`

作用：

- 定义 RL 环境状态
- 解析模型输出协议
- 构造当前 step 的 prompt
- 执行 `env.step(...)`
- 维护：
  - `history`
  - `remaining_turns`
  - `sql_probe_count`
  - `successful_probe_count`
  - `final_failure_reason`

这里是当前 agent 环境的核心。

- `agent_rl/grpo_runner.py`

作用：

- 用任意 policy 函数跑 rollout
- 把 rollout 结果转成可序列化结构
- 导出 TRL 风格的 step records

用途：

- 本地 smoke test
- 调试 reward / prompt / step 逻辑

### online 主线数据准备

- `scripts/prepare_online_grpo_seeds.py`

作用：

- 直接从全部 gold seeds 划分 train / val
- 可选把 synthetic / Dataflow 数据并入 train
- 验证集保持 gold-only
- 输出：
  - `output/rl_seed_pool_v2/train_rl_seeds.json`
  - `output/rl_seed_pool_v2/val_rl_seeds.json`

### online rollout

- `agent_rl/online_rollout.py`

作用：

- 从初始 seed 开始跑整条多轮 episode
- 每轮构造 prompt、生成、执行 SQL、接 observation
- 返回：
  - `prompt_ids`
  - `completion_ids`
  - `logprobs`
  - `env_reward`
  - episode 级调试信息

### online 训练入口

- `scripts/train_grpo_online_trl.py`

作用：

- 加载 base model
- 可选加载已经训练好的 SFT adapter 作为 RL 初始化
- 读取 `prepare_online_grpo_seeds.py` 导出的 seed 池
- 构建 `rollout_func`
- 用 `TRL GRPOTrainer` 训练在线多轮策略

这是当前推荐主入口。

### single-step 旧版

- `scripts/prepare_grpo_dataset.py`
- `scripts/train_grpo_trl.py`

保留用途：

- 旧实验复现
- 和 online 版做对照

### 文档

- `RL_DESIGN.md`

作用：

- 定义当前 RL 方案的状态、动作、终止条件、奖励函数
- 说明当前为什么采用“单步 GRPO”
- 记录未来升级到多步 agent RL 的方向

- `training/trl/README.md`

作用：

- 说明如何准备 GRPO 数据
- 说明如何启动第一版 `TRL + GRPO` 训练
- 给出适合当前 `14B + 96GB` 环境的推荐参数

## 2. 设计原则

第一版 RL 方案遵循以下原则：

1. 奖励函数以**最终结果正确**为核心
2. 中间步骤只做最小必要的过程奖励
3. 不引入复杂的信息增益判别器
4. 终止规则必须清晰
5. 先做小规模验证，再考虑大规模 GRPO

## 3. Observation 定义

RL 环境中每一步看到的 observation 包括：

- `question`
- `schema_snippets`
- `knowledge`
- `history`
- `remaining_turns`
- `difficulty`

### 具体字段

```json
{
  "question": "...",
  "schema_snippets": ["...", "..."],
  "knowledge": "...",
  "history": [
    {
      "turn_id": 1,
      "action": {
        "action_type": "sql",
        "reasoning": "...",
        "sql": "SELECT ..."
      },
      "observation": {
        "status": "success|empty|error|timeout",
        "error_message": "",
        "columns": ["..."],
        "sample_rows": [{...}],
        "row_count": 3,
        "turns_left": 2
      }
    }
  ],
  "remaining_turns": 2,
  "difficulty": "简单|中等|困难"
}
```

说明：

- `latest_observation` 永远表示**上一轮 action 执行后的结果**
- 第一轮时：
  - `history = []`

这种写法的好处是：

- 多轮时上下文更紧
- action 和 observation 的配对关系更清晰
- 更适合后续 reward 分析和 debug

## 4. Action 定义

Action 仍然保持当前已经验证过的协议：

- `<sql>...</sql>`
- `<solution>...</solution>`

并统一抽象为：

```json
{
  "action_type": "sql | solution",
  "reasoning": "...",
  "sql": "SELECT ..."
}
```

### 语义

- `action_type = sql`
  表示中间验证 / 探索型 SQL

- `action_type = solution`
  表示提交最终 SQL，提交后立刻结束 episode

## 5. 终止条件

Episode 在以下情况终止：

1. 模型输出 `<solution>`
2. 协议格式错误
3. 达到 `max_turns`

### 特别约束

- `solution` 一旦提交，无论对错都终止
- 协议错误直接终止，不允许继续 rollout
- 如果在最后一轮仍输出 `<sql>` 而不是 `<solution>`：
  - episode 直接结束
  - 按 final failure 处理

## 6. difficulty 到 turn budget 的映射

题目复杂度直接使用现有标签，不再额外训练复杂度分类器。

### 第一版映射

- `简单` / `easy`
  - `max_turns = 2`
  - `b_d = 0`
  - `lambda_d = 0.5`

- `中等` / `medium`
  - `max_turns = 3`
  - `b_d = 1`
  - `lambda_d = 0.3`

- `困难` / `hard`
  - `max_turns = 4`
  - `b_d = 2`
  - `lambda_d = 0.15`

### fallback

如果 difficulty 缺失：

- 默认按 `中等 / medium` 处理

## 7. 奖励函数设计

第一版 reward 采用“终局主导 + 轻量过程奖励”的设计。

### 7.1 终局奖励

#### `r_final_result`

最终 SQL 的执行结果和 gold result 直接比较：

- `1.0`：完全一致
- `0.0`：不一致

这里不做“部分匹配”。

第一版直接使用二值结果匹配，保持定义清晰。

#### `r_sql_ngram`

最终 SQL 与 golden SQL 的 n-gram 相似度，范围 `[0, 1]`。

用途：

- 只作为弱辅助奖励
- 防止在 reward 极稀疏时完全失去方向

注意：

- 它不能作为主奖励
- 因为同一个问题可能存在多个等价 SQL

### 7.2 过程奖励

#### `r_exec_step`

对于中间 `<sql>`：

- 第一次成功可执行 probe：`+0.2`
- 后续成功可执行 probe：`0`
- 执行失败：`-0.3`

第一版不判断“信息增益”，只判断能否执行。
这样可以避免模型重复刷“可执行分”。

#### `r_turn`

`r_turn` 不再按“每一步都扣”，而按 probe 次数超过免费额度后再扣：

\[
r_{turn} = -\lambda_d \cdot \max(0, n_{sql\_probe} - b_d)
\]

其中：

- `n_sql_probe`
  中间 probe 次数

- `b_d`
  对应难度允许的“免费 probe 数”

- `lambda_d`
  对应难度的惩罚强度

按 difficulty 动态取值：

- easy:
  - `b_d = 0`
  - `lambda_d = 0.5`

- medium:
  - `b_d = 1`
  - `lambda_d = 0.3`

- hard:
  - `b_d = 2`
  - `lambda_d = 0.15`

#### `r_format`

协议正确：

- `0`

协议错误：

- `-1.0`
- 并终止 episode

#### `r_final_exec_fail`

如果最终输出 `<solution>`，但 final SQL 执行失败：

- `-0.5`

这样可以区分：

- final SQL 能执行但结果不对
- final SQL 连执行都执行不了

### 7.3 第一版总 reward

建议第一版使用：

```text
R = 5.0 * r_final_result
  + 0.3 * r_sql_ngram
  + r_exec_step
  + r_turn
  + r_format
  + r_final_exec_fail
```

说明：

- `r_final_result` 是主导项
- `r_sql_ngram` 只是弱辅助
- 中间 SQL 可执行性用于稳定 rollout
- turn penalty 用于鼓励少 probe、少无效探索
- `r_final_exec_fail` 用于显式惩罚“final SQL 根本跑不通”

## 8. 为什么这样设计

### 8.1 为什么最终结果匹配是主奖励

因为最终目标不是生成“长得像”的 SQL，而是：

- SQL 执行后结果正确

所以最终结果匹配必须成为 reward 中最重要的一项。

### 8.2 为什么中间 SQL 只看可执行

因为第一版 RL 不应该过度复杂化。

如果现在引入：

- 信息增益判断
- 中间目标集合质量判断
- schema linking 分项奖励

会显著增加系统复杂度和调试成本。

而当前阶段最重要的是：

- 先让 rollout 稳定
- 先验证 RL 能不能提升 final solution

同时，只有第一次成功 probe 才给正奖励，可以避免模型通过重复输出可执行 probe 来刷 reward。

### 8.3 为什么格式错误要终止

因为协议错误意味着：

- 当前动作已经不可解释
- 后续 observation 无法可靠对齐

所以第一版直接终止最干净。

## 9. rollout 流程

第一版 RL 采用“小批 rollout -> 小批更新”的结构。

每轮：

1. 从数据集中采样一批题目
2. 用当前 policy 逐题 rollout
3. 每步执行 SQL，拿 observation
4. 如果最后一轮仍是 `<sql>`，直接按 final failure 结束
5. 到终止时计算 reward
5. 把这批 trajectories 送入 GRPO / RL 更新
6. 更新后进入下一轮 rollout

这本质上属于在线 RL，但工程实现上不需要做成复杂异步系统。

## 10. 第一版不做的事情

为了先把系统跑稳，第一版明确不做：

- 中间 SQL 信息增益奖励
- 部分结果匹配奖励
- schema linking 显式奖励
- 复杂 reward model
- 多 agent / 多分支搜索

这些可以后续作为增强项加入。

## 11. 当前最关键的评估指标

进入 RL 后，最该追踪的是：

- `pred_exec_success_rate`
- `result_match_rate`
- `pred_solution_exec_success_rate`
- `action_type_accuracy`
- `average_turns`

预期目标是：

- 先提升 final solution 的质量
- 再进一步减少无效轮次

## 12. 当前阶段结论

当前阶段建议是：

1. 用 `Agent 单轮 SFT` 模型作为 RL 初始化
2. 用本文档这版 reward 设计作为第一版 GRPO 方案
3. 先做小规模 rollout 验证
4. 验证 reward 是否能把：
   - `pred_solution_exec_success_rate`
   - `result_match_rate`
   往上推

一句话总结：

> 第一版 RL 方案应该优先围绕“最终结果正确 + 中间 SQL 可执行 + 轮数受控 + 协议稳定”展开，而不是一开始就引入过于复杂的 reward shaping。
