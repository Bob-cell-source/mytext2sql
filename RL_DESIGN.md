# RL 设计方案

这份文档用于定义当前项目第一版基于 `Agent 单轮 SFT` 的后续 RL / GRPO 方案。

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

当前仓库第一版并没有直接实现“trainer 内部完整多轮 rollout”的 RL，
而是采用了一个更稳、更容易落地的方案：

- `单步 state -> 单步 action`
- `reward function` 内重建当前 state
- 再调用环境做一步 `env.step(...)`

也就是说，当前版本更准确地说是：

- **单步 agent GRPO**

它的目标是先验证：

- reward 是否有效
- agent 单轮 policy 是否还能继续提升
- `solution` 质量是否能被 RL 拉起来

这样做的优点是：

- 更容易与 `TRL GRPOTrainer` 直接对接
- 不需要一开始就实现复杂多轮 trainer
- 更容易调试 reward 和协议解析

## 1.2 当前方案与未来多步 agent RL 的区别

### 当前方案：单步 GRPO

训练样本形式：

- 输入：`state_t`
- 输出：`action_t`
- reward：调用环境对 `action_t` 打分

特点：

- 最适合当前这版 `Agent 单轮 SFT` 冷启动
- 不需要注册 tools
- 不需要 OpenEnv
- 也不需要处理 vLLM 训练-推理精度不一致下的重要性采样修正

适用阶段：

- 第一版验证 RL 是否有增益
- 第一版验证 reward 设计是否合理

### 未来方案：多步 OpenEnv / rollout_func GRPO

如果后面要做真正的多轮 agent RL，更像这样：

- trainer 内部逐轮生成 action
- env 执行 SQL
- 返回 observation
- 再进入下一轮
- 直到 `<solution>` 或 episode 终止

这时更适合：

- `rollout_func`
- 或者 `TRL` 的 `OpenEnv` 集成

特点：

- 更接近真正的 agent 训练
- 能直接优化整条 episode 行为
- 更适合研究多轮探索与自纠

代价：

- 实现复杂很多
- 调试难度显著上升
- 如果还接入 vLLM 在线生成，才需要认真考虑：
  - `Truncated Importance Sampling`
  - `Masked Importance Sampling`
  - 训练/推理精度不一致带来的校正问题

## 1.3 当前阶段的推荐策略

当前推荐路线是：

1. 先把 **单步 GRPO** 跑通
2. 先看核心任务指标是否提升
3. 如果单步 GRPO 已经证明有效，再考虑升级成真正多步 agent RL

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

### 数据准备

- `scripts/prepare_grpo_dataset.py`

作用：

- 从 `output/llamafactory_sft_v5/train_full_trajectory.json`
  和 `val_full_trajectory.json`
  恢复出单步 RL 样本
- 把完整轨迹拆成：
  - 当前 `prompt`
  - 当前 `history`
  - 当前 `seed`
  - 当前 gold action

输出：

- `output/rl_training_inputs_v1/train_rl_single_step.json`
- `output/rl_training_inputs_v1/val_rl_single_step.json`

注意：

- 这一步产出的不是 “GRPO 生成结果”
- 而是从现有 `SFT` 轨迹整理出的 **RL 训练输入数据**

这一步的目标，是给 `TRL GRPOTrainer` 提供可直接消费的单步 RL 数据。

### 训练入口

- `scripts/train_grpo_trl.py`

作用：

- 加载 base model
- 可选加载已经训练好的 SFT adapter 作为 RL 初始化
- 读取 `prepare_grpo_dataset.py` 导出的数据
- 构建 `GRPOTrainer`
- 在 reward function 里调用当前 env 做单步打分

这是当前第一版 RL 训练主入口。

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
