# Dataflow 测试集生成说明

这份文档说明如何把当前项目中的 gold SQL 数据导出为 Dataflow 可用 seed，并用于生成新的测试问题。

当前建议优先使用：

- [run_dataflow/api_pipelines/text2sql_pipeline_refine.py](/root/text2sql_RL/run_dataflow/api_pipelines/text2sql_pipeline_refine.py)

不建议一开始就把：

- [run_dataflow/api_pipelines/text2vecsql_pipeline_gen.py](/root/text2sql_RL/run_dataflow/api_pipelines/text2vecsql_pipeline_gen.py)

作为主测试集来源。

原因：

- `text2sql_pipeline_refine.py` 更适合从你已有的业务问题 / SQL 出发做扩增
- `text2vecsql_pipeline_gen.py` 更像从 schema 中“从零造题”，更适合作为补充数据源

## 1. 两个 pipeline 的原理区别

### 1.1 `text2sql_pipeline_refine.py`

作用：

- 从现有 question + SQL 种子出发扩增数据

主要流程：

1. 过滤掉不能执行的原始 SQL
2. 生成 SQL 变体
3. 再过滤不能执行的 SQL 变体
4. 基于 SQL 生成新问题
5. 做 question-SQL 对应性过滤
6. 生成 prompt
7. 生成 CoT
8. 对 CoT 做投票
9. 做组件难度分类
10. 做执行难度分类

适合你现在的原因：

- 你已经有高质量 gold SQL
- 你更需要“围绕真实业务题做扩增”
- 这种方式更容易生成自然且贴近业务分布的测试题

### 1.2 `text2vecsql_pipeline_gen.py`

作用：

- 从数据库结构和列信息出发，先生成 SQL，再反向生成问题

主要流程：

1. 先按列/表结构生成 SQL
2. 过滤掉执行失败的 SQL
3. 再基于 SQL 生成问题
4. 生成 prompt
5. 做难度分类

适合的场景：

- 你想从 schema 中额外挖掘一些题
- 想补充一些“不在原始题分布中”的测试题

不适合作为第一优先级的原因：

- 更容易生成语法上成立但业务感较弱的题

## 2. 推荐工作流

建议你按这个顺序做：

1. 先把当前 [golden_sql_marked.json](/root/text2sql_RL/golden_sql_marked.json) 导出成 Dataflow seed
2. 先跑 refine pipeline
3. 从 refine 的结果里抽新问题，人工筛一遍
4. 再决定是否用 vecsql pipeline 做补充测试集

## 3. 导出 Dataflow seed

已经提供脚本：

- [scripts/export_dataflow_text2sql_seed.py](/root/text2sql_RL/scripts/export_dataflow_text2sql_seed.py)

示例：

```bash
python3 scripts/export_dataflow_text2sql_seed.py \
  --input golden_sql_marked.json \
  --output run_dataflow/example_data/Text2SQLPipeline/tgac_pipeline_refine.jsonl \
  --db-id tgac
```

导出后的每条记录格式是：

```json
{
  "db_id": "tgac",
  "question": "...",
  "SQL": "SELECT ...",
  "sql_id": "sql_1",
  "table_list": ["..."],
  "knowledge": "...",
  "difficulty": "中等"
}
```

其中 Dataflow refine pipeline 核心需要的字段是：

- `db_id`
- `question`
- `SQL`

其余字段只是为了后续你自己做回溯和筛选更方便。

## 4. 如何修改 `text2sql_pipeline_refine.py`

你需要改 3 个地方。

### 4.1 改 seed 输入文件

当前默认：

- `../example_data/Text2SQLPipeline/pipeline_refine.jsonl`

你应该改成：

- `../example_data/Text2SQLPipeline/tgac_pipeline_refine.jsonl`

对应代码位置：

- `self.storage = FileStorage(...)`

### 4.2 改数据库连接

当前脚本默认用的是 SQLite 示例数据库：

```python
database_manager = DatabaseManager(
    db_type="sqlite",
    config={"root_path": self.db_root_path}
)
```

如果你要接自己的业务库，建议改成 MySQL：

```python
database_manager = DatabaseManager(
    db_type="mysql",
    config={
        "host": "127.0.0.1",
        "user": "root",
        "password": "your_password",
        "database": "TGAC"
    }
)
```

如果底层兼容 StarRocks 的 MySQL 协议，也可以直接用这个方式。

### 4.3 改缓存目录

当前缓存目录是：

- `./cache`

建议改成单独目录，例如：

- `./cache_tgac_refine`

避免和示例缓存混在一起。

## 5. API 配置

根据 [run_dataflow/api_pipelines/README.md](/root/text2sql_RL/run_dataflow/api_pipelines/README.md)，需要先配置 API key。

如果你沿用 OpenAI 风格接口，要先导出对应 key。

Dataflow README 里示例是：

```bash
export DF_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

但你具体还要看你使用的 serving 配置。

当前 refine pipeline 里默认写的是：

```python
self.llm_serving = APILLMServing_request(
    api_url="https://api.openai.com/v1/chat/completions",
    model_name="gpt-4o",
    max_workers=100
)
```

你如果要接自己的兼容接口，需要改：

- `api_url`
- `model_name`

## 6. 运行方式

在改完 seed 文件路径和数据库连接后，可以直接运行：

```bash
python3 run_dataflow/api_pipelines/text2sql_pipeline_refine.py
```

运行后，Dataflow 会把中间步骤写到你配置的 cache 目录中。

## 7. 你应该怎么使用生成结果

不建议把 Dataflow 输出直接全部当成最终测试集。

建议这样处理：

1. 从 cache 最后阶段抽取：
   - 新问题
   - 对应 SQL
   - prompt
   - 难度标签
2. 做人工抽样筛选
3. 去重
4. 标记来源为 `dataflow_refine`
5. 单独保存成你的测试集 JSON

推荐保留的字段：

```json
{
  "sample_id": "df_test_001",
  "source": "dataflow_refine",
  "db_id": "tgac",
  "question": "...",
  "sql": "...",
  "difficulty": "medium",
  "table_list": ["..."],
  "knowledge": "..."
}
```

## 8. 质量控制建议

生成后至少检查：

- 是否和现有 gold 问题过于相似
- 是否只是简单改日期
- 是否业务口径自然
- SQL 是否真的可执行
- 难度分布是否均衡

建议你把这批 Dataflow 生成题只作为：

- 测试集
- 泛化评测集

不要直接混到训练集中。

## 9. 当前建议

对你这个项目，建议优先路线：

1. 用 [scripts/export_dataflow_text2sql_seed.py](/root/text2sql_RL/scripts/export_dataflow_text2sql_seed.py) 导出 TGAC seed
2. 改 `text2sql_pipeline_refine.py`
3. 跑 refine pipeline
4. 把输出整理成测试集
5. 后续如果需要，再尝试 `text2vecsql_pipeline_gen.py` 做补充测试题
