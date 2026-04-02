#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export project gold SQL data to Dataflow Text2SQL seed format.")
    parser.add_argument(
        "--input",
        default="golden_sql_marked.json",
        help="Input gold SQL json file.",
    )
    parser.add_argument(
        "--output",
        default="run_dataflow/example_data/Text2SQLPipeline/tgac_pipeline_refine.jsonl",
        help="Output jsonl seed file for Dataflow refine pipeline.",
    )
    parser.add_argument(
        "--db-id",
        default="tgac",
        help="db_id field used by Dataflow.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only export first N records when > 0.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_json(input_path)
    exported = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in rows:
            if item.get("golden_sql") is not True:
                continue
            record = {
                "db_id": args.db_id,
                "question": (item.get("question") or "").strip(),
                "SQL": (item.get("sql") or "").strip(),
                "sql_id": item.get("sql_id", ""),
                "table_list": item.get("table_list", []),
                "knowledge": item.get("knowledge", ""),
                "difficulty": item.get("复杂度", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
            if args.limit > 0 and exported >= args.limit:
                break

    print(f"Exported {exported} seed records to {output_path}")


if __name__ == "__main__":
    main()
