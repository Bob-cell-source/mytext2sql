import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT = "run_dataflow/api_pipelines/cache_tgac_refine/dataflow_cache_step_step9.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def build_sample_id(row: Dict[str, Any], index: int, prefix: str) -> str:
    sql_id = normalize_value(row.get("sql_id"))
    if sql_id:
        return f"{prefix}_{sql_id}"
    return f"{prefix}_{index:04d}"


def convert_row(
    row: Dict[str, Any],
    index: int,
    source_name: str,
    id_prefix: str,
    include_prompt: bool,
    include_cot: bool,
    keep_original_fields: bool,
) -> Dict[str, Any]:
    sample: Dict[str, Any] = {
        "sample_id": build_sample_id(row, index, id_prefix),
        "db_id": normalize_value(row.get("db_id")),
        "question": normalize_value(row.get("question")),
        "gold_sql": normalize_value(row.get("SQL")),
        "evidence": normalize_value(row.get("evidence")),
        "knowledge": normalize_value(row.get("knowledge")),
        "difficulty": normalize_value(row.get("sql_component_difficulty") or row.get("difficulty")),
        "source": source_name,
        "metadata": {
            "sql_id": normalize_value(row.get("sql_id")),
            "question_type": normalize_value(row.get("question_type")),
            "sql_variation_type": normalize_value(row.get("sql_variation_type")),
            "table_list": row.get("table_list") if row.get("table_list") is not None else [],
            "dataflow_sql_component_difficulty": normalize_value(row.get("sql_component_difficulty")),
            "dataflow_original_difficulty": normalize_value(row.get("difficulty")),
        },
    }

    if include_prompt:
        sample["prompt"] = normalize_value(row.get("prompt"))
    if include_cot:
        sample["cot_reasoning"] = normalize_value(row.get("cot_reasoning"))
    if keep_original_fields:
        sample["original_record"] = row
    return sample


def dedupe_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    removed = 0
    for row in rows:
        key = (
            row.get("db_id", ""),
            row.get("question", ""),
            row.get("gold_sql", ""),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def write_output(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Dataflow refine cache into a unified evaluation dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input Dataflow cache jsonl file, typically step5 or step9.")
    parser.add_argument("--output", required=True, help="Output dataset path (.json or .jsonl).")
    parser.add_argument("--source-name", default="dataflow_refine_step9", help="Source tag written into exported samples.")
    parser.add_argument("--id-prefix", default="df_eval", help="Prefix for generated sample_id.")
    parser.add_argument("--no-prompt", action="store_true", help="Do not export Dataflow prompt field.")
    parser.add_argument("--no-cot", action="store_true", help="Do not export cot_reasoning field.")
    parser.add_argument("--keep-original-fields", action="store_true", help="Keep the original Dataflow row under original_record.")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication by db_id + question + gold_sql.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_rows = read_jsonl(input_path)
    exported_rows = [
        convert_row(
            row=row,
            index=index,
            source_name=args.source_name,
            id_prefix=args.id_prefix,
            include_prompt=not args.no_prompt,
            include_cot=not args.no_cot,
            keep_original_fields=args.keep_original_fields,
        )
        for index, row in enumerate(raw_rows, start=1)
    ]

    removed_duplicates = 0
    if not args.no_dedupe:
        exported_rows, removed_duplicates = dedupe_rows(exported_rows)

    write_output(output_path, exported_rows)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "source_name": args.source_name,
        "raw_count": len(raw_rows),
        "exported_count": len(exported_rows),
        "removed_duplicates": removed_duplicates,
        "include_prompt": not args.no_prompt,
        "include_cot": not args.no_cot,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
