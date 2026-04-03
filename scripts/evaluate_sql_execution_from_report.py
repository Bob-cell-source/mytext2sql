import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.result_matcher import is_result_match
from agent_rl.sql_env import SQLEnvironment


SQL_PATTERN = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.DOTALL | re.IGNORECASE)
SOLUTION_PATTERN = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.DOTALL | re.IGNORECASE)


def extract_action(text: str) -> Tuple[Optional[str], str]:
    sql_match = SQL_PATTERN.search(text or "")
    if sql_match:
        return "sql", sql_match.group(1).strip()

    solution_match = SOLUTION_PATTERN.search(text or "")
    if solution_match:
        return "solution", solution_match.group(1).strip()

    return None, ""


def is_exec_success(observation: Dict[str, Any]) -> bool:
    return observation.get("status") in {"success", "empty"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute predicted SQLs from a vLLM report and compare against gold SQL results.")
    parser.add_argument("--report-path", required=True, help="Path to vLLM inference report JSON.")
    parser.add_argument("--output-path", required=True, help="Where to write execution evaluation JSON.")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only the first N predictions if > 0.")
    parser.add_argument("--work-dir", default="./tmp_sql_eval", help="Temporary directory for SQL execution helper.")
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
    pred_action_present = 0
    pred_exec_success = 0
    gold_action_present = 0
    gold_exec_success = 0
    result_match_count = 0
    action_type_match_count = 0

    pred_sql_count = 0
    pred_sql_exec_success = 0
    pred_solution_count = 0
    pred_solution_exec_success = 0

    records: List[Dict[str, Any]] = []

    for item in predictions:
        pred_action_type, pred_sql = extract_action(item.get("pred_output", ""))
        gold_action_type, gold_sql = extract_action(item.get("gold_output", ""))

        if pred_action_type:
            pred_action_present += 1
        if gold_action_type:
            gold_action_present += 1
        if pred_action_type == gold_action_type and pred_action_type is not None:
            action_type_match_count += 1

        pred_observation, pred_rows = (
            sql_env.execute_with_rows(pred_sql, turns_left=0, sql_id=f"{item['id']}_pred")
            if pred_sql else
            ({
                "status": "error",
                "error_message": "No predicted SQL extracted.",
                "columns": [],
                "sample_rows": [],
                "row_count": 0,
                "fingerprint": "",
                "turns_left": 0,
            }, [])
        )

        gold_observation, gold_rows = (
            sql_env.execute_with_rows(gold_sql, turns_left=0, sql_id=f"{item['id']}_gold")
            if gold_sql else
            ({
                "status": "error",
                "error_message": "No gold SQL extracted.",
                "columns": [],
                "sample_rows": [],
                "row_count": 0,
                "fingerprint": "",
                "turns_left": 0,
            }, [])
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

        if pred_action_type == "sql":
            pred_sql_count += 1
            if pred_success:
                pred_sql_exec_success += 1
        elif pred_action_type == "solution":
            pred_solution_count += 1
            if pred_success:
                pred_solution_exec_success += 1

        records.append(
            {
                "id": item.get("id"),
                "seed_id": item.get("seed_id"),
                "turn_id": item.get("turn_id"),
                "dataset_kind": item.get("dataset_kind"),
                "pred_action_type": pred_action_type,
                "gold_action_type": gold_action_type,
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
        "pred_action_present_rate": round(pred_action_present / total, 4) if total else 0.0,
        "gold_action_present_rate": round(gold_action_present / total, 4) if total else 0.0,
        "action_type_accuracy": round(action_type_match_count / total, 4) if total else 0.0,
        "pred_exec_success_rate": round(pred_exec_success / total, 4) if total else 0.0,
        "gold_exec_success_rate": round(gold_exec_success / total, 4) if total else 0.0,
        "result_match_rate": round(result_match_count / total, 4) if total else 0.0,
        "pred_sql_exec_success_rate": round(pred_sql_exec_success / pred_sql_count, 4) if pred_sql_count else None,
        "pred_solution_exec_success_rate": round(pred_solution_exec_success / pred_solution_count, 4) if pred_solution_count else None,
    }

    output = {
        "report_path": str(report_path),
        "metrics": metrics,
        "records": records,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved SQL execution evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
