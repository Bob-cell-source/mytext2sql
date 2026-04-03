#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_if_exists(path_str: str) -> Optional[Dict[str, Any]]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_experiment(
    label: str,
    text_report: Optional[Dict[str, Any]],
    exec_report: Optional[Dict[str, Any]],
    failure_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"experiment": label}

    text_metrics = (text_report or {}).get("metrics", {})
    exec_metrics = (exec_report or {}).get("metrics", {})

    row.update(
        {
            "text_total": text_metrics.get("total"),
            "exact_match_rate": text_metrics.get("exact_match_rate"),
            "protocol_valid_rate": text_metrics.get("protocol_valid_rate"),
            "reasoning_present_rate": text_metrics.get("reasoning_present_rate"),
            "action_type_accuracy_text": text_metrics.get("action_type_accuracy"),
            "action_body_exact_match_rate": text_metrics.get("action_body_exact_match_rate"),
            "sql_action_body_exact_match_rate": text_metrics.get("sql_action_body_exact_match_rate"),
            "solution_action_body_exact_match_rate": text_metrics.get("solution_action_body_exact_match_rate"),
            "non_empty_rate": text_metrics.get("non_empty_rate"),
            "sql_exact_match_rate": text_metrics.get("sql_exact_match_rate"),
        }
    )

    row.update(
        {
            "exec_total": exec_metrics.get("total"),
            "pred_action_present_rate": exec_metrics.get("pred_action_present_rate"),
            "gold_action_present_rate": exec_metrics.get("gold_action_present_rate"),
            "action_type_accuracy_exec": exec_metrics.get("action_type_accuracy"),
            "pred_exec_success_rate": exec_metrics.get("pred_exec_success_rate"),
            "gold_exec_success_rate": exec_metrics.get("gold_exec_success_rate"),
            "result_match_rate": exec_metrics.get("result_match_rate"),
            "pred_sql_exec_success_rate": exec_metrics.get("pred_sql_exec_success_rate"),
            "pred_solution_exec_success_rate": exec_metrics.get("pred_solution_exec_success_rate"),
            "pred_non_empty_rate": exec_metrics.get("pred_non_empty_rate"),
        }
    )

    if failure_report:
        failure_breakdown = {
            item["value"]: item["count"]
            for item in failure_report.get("failure_breakdown", [])
            if isinstance(item, dict) and "value" in item and "count" in item
        }
        row.update(
            {
                "failure_matched": failure_breakdown.get("matched"),
                "failure_execution_error": failure_breakdown.get("execution_error"),
                "failure_result_mismatch": failure_breakdown.get("result_mismatch"),
                "failure_wrong_action_type": failure_breakdown.get("wrong_action_type"),
                "failure_missing_action": failure_breakdown.get("missing_action"),
            }
        )

    return row


def parse_experiment(spec: str) -> Dict[str, str]:
    parts = spec.split("::")
    if len(parts) < 2:
        raise ValueError(
            "Each --experiment must be in the form "
            "'label::text_report::exec_report::failure_report'. "
            "exec_report and failure_report can be left empty."
        )
    while len(parts) < 4:
        parts.append("")
    return {
        "label": parts[0],
        "text_report": parts[1],
        "exec_report": parts[2],
        "failure_report": parts[3],
    }


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    columns = [
        "experiment",
        "exact_match_rate",
        "protocol_valid_rate",
        "action_type_accuracy_text",
        "pred_exec_success_rate",
        "result_match_rate",
        "pred_sql_exec_success_rate",
        "pred_solution_exec_success_rate",
        "failure_execution_error",
        "failure_result_mismatch",
        "failure_wrong_action_type",
        "failure_missing_action",
    ]

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                value = ""
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple experiment report JSON files.")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help=(
            "Experiment spec in the form "
            "'label::text_report::exec_report::failure_report'. "
            "For baseline without failure report, leave the last field empty."
        ),
    )
    parser.add_argument("--output-json", required=True, help="Path to save merged comparison JSON.")
    parser.add_argument("--output-md", default="", help="Optional path to save Markdown comparison table.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for spec in args.experiment:
        parsed = parse_experiment(spec)
        row = flatten_experiment(
            label=parsed["label"],
            text_report=load_json_if_exists(parsed["text_report"]),
            exec_report=load_json_if_exists(parsed["exec_report"]),
            failure_report=load_json_if_exists(parsed["failure_report"]),
        )
        rows.append(row)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(to_markdown(rows), encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved comparison JSON to: {output_json}")
    if args.output_md:
        print(f"Saved comparison Markdown to: {args.output_md}")


if __name__ == "__main__":
    main()
