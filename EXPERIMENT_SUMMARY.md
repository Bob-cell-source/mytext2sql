# 实验总结

这份文档用于记录当前阶段 `普通版 Final-SQL SFT`、`Agent 单轮 SFT`、`Agent 整体轨迹 SFT` 以及后续 `GRPO` 实验的对比结果，并给出阶段性结论。

## 1. 实验目的

当前实验的核心问题不是“模型能不能训练起来”，而是：

- 在同样经过 SFT 的前提下
- `普通 final-SQL SFT` 和 `agent 式 SFT`
- 哪一种更适合当前的 Text2SQL agent 任务

这里的对比重点是：

- 普通版：直接学习 `question + schema + knowledge -> final SQL`
- Agent 单轮版：学习 `current state -> next action`
- Agent 整体轨迹版：学习完整多轮轨迹

## 2. 结论概览

当前实验结果支持以下结论：

1. `Agent 单轮 SFT` 明显优于 `普通版 Final-SQL SFT`
2. `Agent 单轮 SFT` 也明显优于 `Agent 整体轨迹 SFT`
3. `single-step GRPO` 没有超过最优的 `Agent 单轮 SFT`
4. `online multi-turn GRPO` 比 `single-step GRPO` 更合理，也更接近目标方向
5. 但当前 `online multi-turn GRPO` 仍未超过最优的 `Agent 单轮 SFT`
6. 因此，当前阶段最值得继续推进的主线仍然是：
   `Agent 单轮 SFT -> online multi-turn RL`

## 3. 关键对比结果

### 3.1 普通版 Final-SQL Baseline

执行评测结果：

- `pred_exec_success_rate = 0.2353`
- `result_match_rate = 0.2353`

这说明：

- 模型虽然能稳定输出 SQL
- 但大部分最终 SQL 不可执行或结果不正确

### 3.2 Agent 单轮版

离线结果：

- `protocol_valid_rate = 0.8776`
- `action_type_accuracy = 0.7143`
- `action_body_exact_match_rate = 0.1429`

执行评测结果：

- `pred_exec_success_rate = 0.4694`
- `result_match_rate = 0.2857`
- `pred_sql_exec_success_rate = 0.7895`
- `pred_solution_exec_success_rate = 0.3333`

失败分布：

- `matched = 14`
- `execution_error = 12`
- `result_mismatch = 9`
- `wrong_action_type = 8`
- `missing_action = 6`

### 3.3 Agent 整体轨迹版

离线结果：

- `protocol_valid_rate = 0.6275`
- `action_type_accuracy = 0.3137`
- `action_body_exact_match_rate = 0.0784`

执行评测结果：

- `pred_exec_success_rate = 0.2745`
- `result_match_rate = 0.1569`

这说明当前整体轨迹版在数据量较小时明显更难训稳。

### 3.4 Agent 单轮版 + single-step GRPO

离线结果：

- `protocol_valid_rate = 0.8571`
- `action_type_accuracy = 0.6939`

执行评测结果：

- `pred_exec_success_rate = 0.4286`
- `result_match_rate = 0.1837`
- `pred_sql_exec_success_rate = 0.7000`
- `pred_solution_exec_success_rate = 0.3182`

这说明：

- single-step GRPO 没有带来正收益
- 它比最优 `Agent 单轮 SFT` 明显回退
- 但仍然优于最弱的整体轨迹版

### 3.5 Agent Online 多轮版 + GRPO

离线结果：

- `protocol_valid_rate = 0.8571`
- `action_type_accuracy = 0.6939`
- `action_body_exact_match_rate = 0.1429`

执行评测结果：

- `pred_exec_success_rate = 0.3878`
- `result_match_rate = 0.2245`
- `pred_sql_exec_success_rate = 0.5238`
- `pred_solution_exec_success_rate = 0.3810`

这说明：

- online GRPO 比 single-step GRPO 更合理
- 它在 `pred_solution_exec_success_rate` 上已经优于最优 SFT
- 但整体 `pred_exec_success_rate` 与 `result_match_rate` 仍未超过最优 SFT
- 当前趋势更像是：
  - `solution` 有所改善
  - `probe` 质量下降

## 4. 为什么 Agent 单轮版效果更好

当前实验中，Agent 单轮版优于普通版和整体轨迹版，主要有三个原因。

### 4.1 把任务拆成了更容易学习的局部决策

普通版要求模型一步直接输出最终 SQL。

这对当前这种小数据场景很难，因为模型需要同时学会：

- schema grounding
- SQL 构造
- 过滤条件
- 聚合逻辑
- 最终答案格式

而 Agent 单轮版学习的是：

- 当前是否应该继续探索
- 该输出 `<sql>` 还是 `<solution>`
- 下一步 SQL 应该查什么

这种监督粒度更细，因此更容易学。

### 4.2 Agent 单轮版先学会了“可执行的中间行为”

当前结果里最说明问题的一项是：

- `pred_sql_exec_success_rate = 0.7895`

这说明模型对中间 probe SQL 的学习效果明显好于对最终 solution SQL 的学习效果。

也就是说，Agent 单轮 SFT 的收益首先体现在：

- SQL 更容易写成可执行
- 探索行为更像一个真正的 SQL agent

而不是一开始就完全学会最终答案。

### 4.3 结构化协议约束降低了“直接乱猜 final SQL”的概率

Agent 单轮版强制模型遵守：

- `<reasoning>`
- `<sql>` / `<solution>`
- 当前状态 -> 下一步动作

这种结构化输出方式会减少模型直接“一步乱猜 final SQL”的倾向。

从实验结果看，这个收益首先体现在：

- `pred_exec_success_rate` 提高明显

普通版是：

- `0.2353`

Agent 单轮版是：

- `0.4694`

几乎翻倍。

## 5. 为什么整体轨迹版反而更差

当前整体轨迹版没有取得比单轮版更好的效果，主要原因是：

1. 数据量仍然偏小
2. 多轮轨迹建模比单轮决策更难
3. 长上下文下，协议稳定性更容易下降

这也体现在结果中：

- `protocol_valid_rate` 从单轮版的 `0.8776` 下降到整体轨迹版的 `0.6275`
- `action_type_accuracy` 从 `0.7143` 下降到 `0.3137`

因此，在当前数据规模下，整体轨迹版不适合作为主线方案。

## 5. 为什么当前 RL 还没有超过 best SFT

从当前结果看：

- `single-step GRPO` 明显没有对齐完整 agent 目标
- `online multi-turn GRPO` 已经比 `single-step GRPO` 更合理
- 但 reward 设计当前更偏终局结果，因此 probe 质量下降较明显

这体现在：

- 最优 SFT 的 `pred_sql_exec_success_rate = 0.7895`
- online GRPO 的 `pred_sql_exec_success_rate = 0.5238`

与此同时：

- 最优 SFT 的 `pred_solution_exec_success_rate = 0.3333`
- online GRPO 的 `pred_solution_exec_success_rate = 0.3810`

所以当前 RL 更像是在“牺牲中间 probe，换取一点 final solution 改善”。

这说明后续如果继续做 RL，更值得优化的是：

- probe 质量
- action type 决策
- reward 对中间探索行为的约束

## 6. 这个对比实验是否有效

当前实验是有效的，原因是：

- 三个版本都经过了 SFT
- 使用了统一的任务背景与相近的训练框架
- 最终用统一的 SQL 执行评测来比较

因此，这个实验可以支持一个阶段性结论：

> 在当前数据规模和实验设置下，Agent 单轮式 SFT 相比普通 final-SQL SFT 是有效的，并且能够带来更高的 SQL 可执行率和更高的最终结果匹配率。

同时当前 RL 实验支持：

> online multi-turn GRPO 比 single-step GRPO 更符合 agent 训练目标，但当前整体效果仍未超过最佳的 Agent 单轮 SFT 基线。

## 7. 当前还不能过度解读的地方

虽然 Agent 单轮版已经优于普通版，但当前结果仍然说明模型距离成熟系统还有明显差距。

主要问题：

- `pred_solution_exec_success_rate = 0.3333`
  说明最终 solution 仍然偏弱

- `result_match_rate = 0.2857`
  说明整体正确率还不高

- `execution_error = 12`
  说明仍有较多 SQL 根本无法执行

- `wrong_action_type = 8`
  说明什么时候该 probe、什么时候该 solution 仍不稳定

所以当前更准确的定位是：

- `Agent 单轮 SFT` 是一个有效的冷启动方案
- 但还不是最终成品
- `online multi-turn GRPO` 是一个方向正确的后续方案
- 但当前还不能说 RL 已经带来整体正收益

## 8. 对后续 RL 的意义

当前结果支持：

- 后续 RL 应该基于 `Agent 单轮 SFT` 继续做
- 而且应优先走 `online multi-turn GRPO` 路线，而不是回到 `single-step GRPO`

原因是：

- 它已经优于普通 baseline
- 它已经具备一定的 agent 行为能力
- 它的中间 SQL 可执行性明显更强

这与 SQL-TRAIL 的思路是一致的：

- SFT 负责把模型训成一个“能按 agent 方式工作”的初始策略
- RL 再去进一步提升：
  - 最终正确率
  - 探索效率
  - 自纠能力

## 9. 当前推荐主线

建议当前项目的优先级如下：

1. `Agent 单轮 SFT`
2. `基于 Agent 单轮 SFT 的 online multi-turn RL`
3. `普通版 baseline` 作为对照
4. `single-step GRPO` 保留为负例对照
5. `整体轨迹版` 保留为补充实验，不作为主线

## 10. 一句话总结

当前实验结果表明：

> 在当前小数据场景下，Agent 单轮式 SFT 相比普通 final-SQL SFT 更有效。它的优势主要体现在更高的 SQL 可执行率和更好的最终结果匹配率，说明 agent 化监督能够帮助模型更好地学习可执行的中间行为与结构化决策过程。这为后续基于该模型继续进行 RL 优化提供了合理起点。
