import json
from pathlib import Path
from typing import Any, Dict, List

from agent_rl.result_matcher import build_result_fingerprint
from agent_rl.schemas import SeedRecord


DEFAULT_HARD_CONSTRAINTS = "\n".join(
    [
        "1. 只允许单条 MySQL/StarRocks 查询语句，禁止非查询语句。",
        "2. 禁止 Hive/Spark 方言，如 LATERAL VIEW、explode、to_date 等。",
        "3. 只能使用给定 schema 中的表和字段，禁止编造字段。",
        "4. _di 表必须带时间范围，_df 表必须使用单日快照。",
        "5. 严格遵守 ID 体系与映射规则，禁止将全局 ID 和游戏 ID 直接错连。",
        "6. probe SQL 以确认值域、口径、连接路径为目标，尽量简洁高效。",
    ]
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_schema_index(schema_json: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for table in schema_json:
        columns = []
        for col in table.get("columns", []):
            columns.append(
                {
                    "name": col.get("col", ""),
                    "type": col.get("type", ""),
                    "description": col.get("description", ""),
                }
            )
        index[table.get("table_name", "")] = {
            "description": table.get("table_description", ""),
            "columns": columns,
        }
    return index


def build_schema_snippets(table_list: List[str], schema_index: Dict[str, Dict[str, Any]]) -> List[str]:
    snippets: List[str] = []
    for table_name in table_list:
        meta = schema_index.get(table_name, {})
        lines = [f"Table: {table_name}", f"Description: {meta.get('description', '')}", "Columns:"]
        for col in meta.get("columns", []):
            lines.append(f"- {col['name']} ({col['type']}): {col['description']}")
        snippets.append("\n".join(lines))
    return snippets


def build_seed_records(
    golden_path: str,
    schema_path: str,
    hard_constraints: str = DEFAULT_HARD_CONSTRAINTS,
) -> List[SeedRecord]:
    golden_data = load_json(golden_path)
    schema_data = load_json(schema_path)
    schema_index = build_schema_index(schema_data)

    records: List[SeedRecord] = []
    for item in golden_data:
        if item.get("golden_sql") is not True:
            continue
        gold_result = item.get("result", []) or []
        records.append(
            {
                "seed_id": item["sql_id"],
                "question": item.get("question", "").strip(),
                "table_list": item.get("table_list", []) or [],
                "knowledge": item.get("knowledge", "").strip(),
                "difficulty": item.get("复杂度", ""),
                "gold_sql": item.get("sql", "").strip(),
                "gold_result": gold_result,
                "gold_result_fingerprint": build_result_fingerprint(gold_result),
                "schema_snippets": build_schema_snippets(item.get("table_list", []) or [], schema_index),
                "hard_constraints": hard_constraints,
            }
        )
    return records


def dump_seed_records(records: List[SeedRecord], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

