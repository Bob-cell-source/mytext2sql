# Dataflow 测评集导出说明

这份说明用于把 Dataflow `text2sql_pipeline_refine.py` 产出的 step 缓存文件，整理成当前项目统一的测试集格式。

## 1. 哪个 step 适合导出

常见选择有两个：

- `step5`
  适合做最朴素的 `question -> SQL` 测评集。
  到这一步已经完成了问题生成和 question-SQL 对应性过滤，数据相对干净。

- `step9`
  更适合做当前项目的正式测试集。
  到这一步除了 `question` 和 `SQL`，还会带上：
  - `prompt`
  - `cot_reasoning`
  - `sql_component_difficulty`

如果你现在主要想做模型评测，优先建议从 `step9` 导出。

## 2. 新增导出脚本

脚本位置：

- [export_dataflow_evalset.py](/root/text2sql_RL/scripts/export_dataflow_evalset.py)

默认输入：

- `run_dataflow/api_pipelines/cache_tgac_refine/dataflow_cache_step_step9.jsonl`

## 3. 导出后的统一格式

每条样本默认会被整理成：

```json
{
  "sample_id": "df_eval_0001",
  "db_id": "TGAC",
  "question": "...",
  "gold_sql": "...",
  "evidence": "...",
  "knowledge": "...",
  "difficulty": "hard",
  "source": "dataflow_refine_step9",
  "prompt": "...",
  "cot_reasoning": "...",
  "metadata": {
    "sql_id": "",
    "question_type": "Colloquial",
    "sql_variation_type": "Query Structure Modifications",
    "table_list": [],
    "dataflow_sql_component_difficulty": "hard",
    "dataflow_original_difficulty": ""
  }
}
```

说明：

- `gold_sql`
  是后续评测时要对照的目标 SQL。

- `difficulty`
  默认优先取 `sql_component_difficulty`，如果没有再回退到 Dataflow 原始 `difficulty`。

- `prompt` / `cot_reasoning`
  默认会保留，方便你后续做更复杂的评测或分析。

## 4. 最常用的导出命令

### 从 step9 导出为 JSON

```bash
python3 scripts/export_dataflow_evalset.py \
  --input run_dataflow/api_pipelines/cache_tgac_refine/dataflow_cache_step_step9.jsonl \
  --output output/dataflow_evalsets/tgac_step9_eval.json \
  --source-name dataflow_refine_step9
```

### 从 step9 导出为 JSONL

```bash
python3 scripts/export_dataflow_evalset.py \
  --input run_dataflow/api_pipelines/cache_tgac_refine/dataflow_cache_step_step9.jsonl \
  --output output/dataflow_evalsets/tgac_step9_eval.jsonl \
  --source-name dataflow_refine_step9
```

### 从 step5 导出一个更轻量的测评集

```bash
python3 scripts/export_dataflow_evalset.py \
  --input run_dataflow/api_pipelines/cache_tgac_refine/dataflow_cache_step_step5.jsonl \
  --output output/dataflow_evalsets/tgac_step5_eval.json \
  --source-name dataflow_refine_step5 \
  --no-prompt \
  --no-cot
```

## 5. 常用参数

- `--no-prompt`
  不导出 `prompt`

- `--no-cot`
  不导出 `cot_reasoning`

- `--keep-original-fields`
  把 Dataflow 原始整行记录额外保存在 `original_record`

- `--no-dedupe`
  默认会按 `db_id + question + gold_sql` 去重；如果你不想去重，可以关闭

- `--id-prefix`
  控制导出样本的 `sample_id` 前缀

## 6. 导出后建议怎么用

建议你把导出的文件作为项目自己的标准测试集，而不是直接在后续代码里反复读取 Dataflow 的 step 缓存文件。

原因：

- Dataflow step 文件属于中间缓存，字段比较偏 pipeline 内部使用
- 统一导出后，后续评测脚本可以只依赖固定字段：
  - `sample_id`
  - `db_id`
  - `question`
  - `gold_sql`
  - `difficulty`

## 7. 当前建议

如果你现在是第一次整理测试集，建议直接用 `step9`：

```bash
python3 scripts/export_dataflow_evalset.py \
  --input run_dataflow/api_pipelines/cache_tgac_refine/dataflow_cache_step_step9.jsonl \
  --output output/dataflow_evalsets/tgac_step9_eval.json \
  --source-name dataflow_refine_step9
```

这样后面你就可以基于这个导出文件，继续做：

- SQL 执行正确率评测
- difficulty 分层统计
- prompt / cot 质量分析
