import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.result_matcher import is_result_match
from agent_rl.sql_env import SQLEnvironment


def is_exec_success(observation: Dict[str, Any]) -> bool:
    return observation.get("status") in {"success", "empty"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute predicted final SQLs from a baseline vLLM report and compare against gold SQL results."
    )
    parser.add_argument("--report-path", required=True, help="Path to baseline vLLM inference report JSON.")
    parser.add_argument("--output-path", required=True, help="Where to write execution evaluation JSON.")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only the first N predictions if > 0.")
    parser.add_argument("--work-dir", default="./tmp_final_sql_eval", help="Temporary directory for SQL execution helper.")
    args = parser.parse_args()

    report_path = Path(args.report_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = report.get("predictions", [])
    if args.max_samples > 0:
        predictions = predictions[:args.max_samples]

    sql_env = SQLEnvironment(work_dir=args.work_dir)

    total = len(predictions)
    pred_non_empty = 0
    pred_exec_success = 0
    gold_exec_success = 0
    result_match_count = 0

    records: List[Dict[str, Any]] = []

    for item in predictions:
        pred_sql = (item.get("pred_output") or "").strip()
        gold_sql = (item.get("gold_output") or "").strip()

        if pred_sql:
            pred_non_empty += 1

        pred_observation, pred_rows = (
            sql_env.execute_with_rows(pred_sql, turns_left=0, sql_id=f"{item['id']}_pred")
            if pred_sql
            else (
                {
                    "status": "error",
                    "error_message": "No predicted SQL generated.",
                    "columns": [],
                    "sample_rows": [],
                    "row_count": 0,
                    "fingerprint": "",
                    "turns_left": 0,
                },
                [],
            )
        )
        gold_observation, gold_rows = (
            sql_env.execute_with_rows(gold_sql, turns_left=0, sql_id=f"{item['id']}_gold")
            if gold_sql
            else (
                {
                    "status": "error",
                    "error_message": "No gold SQL provided.",
                    "columns": [],
                    "sample_rows": [],
                    "row_count": 0,
                    "fingerprint": "",
                    "turns_left": 0,
                },
                [],
            )
        )

        pred_success = is_exec_success(pred_observation)
        gold_success = is_exec_success(gold_observation)
        matched = pred_success and gold_success and is_result_match(pred_rows, gold_rows)

        if pred_success:
            pred_exec_success += 1
        if gold_success:
            gold_exec_success += 1
        if matched:
            result_match_count += 1

        records.append(
            {
                "id": item.get("id"),
                "seed_id": item.get("seed_id"),
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "pred_observation": pred_observation,
                "gold_observation": gold_observation,
                "pred_exec_success": pred_success,
                "gold_exec_success": gold_success,
                "result_match": matched,
            }
        )

    metrics = {
        "total": total,
        "pred_non_empty_rate": round(pred_non_empty / total, 4) if total else 0.0,
        "pred_exec_success_rate": round(pred_exec_success / total, 4) if total else 0.0,
        "gold_exec_success_rate": round(gold_exec_success / total, 4) if total else 0.0,
        "result_match_rate": round(result_match_count / total, 4) if total else 0.0,
    }

    output = {
        "report_path": str(report_path),
        "metrics": metrics,
        "records": records,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved final-SQL execution evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
