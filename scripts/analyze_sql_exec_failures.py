import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def shorten(text: str, max_len: int = 160) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def top_counter(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    return [{"value": key, "count": value} for key, value in counter.most_common(limit)]


def classify_failure(record: Dict[str, Any]) -> str:
    pred_action_type = record.get("pred_action_type")
    gold_action_type = record.get("gold_action_type")
    pred_exec_success = record.get("pred_exec_success", False)
    result_match = record.get("result_match", False)

    if not pred_action_type:
        return "missing_action"
    if pred_action_type != gold_action_type:
        return "wrong_action_type"
    if not pred_exec_success:
        return "execution_error"
    if not result_match:
        return "result_mismatch"
    return "matched"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failure modes from SQL execution evaluation report.")
    parser.add_argument("--input", required=True, help="Path to vllm_action_exec_eval.json or vllm_full_traj_exec_eval.json")
    parser.add_argument("--output", required=True, help="Path to save the analysis JSON")
    parser.add_argument("--top-k", type=int, default=20, help="Top K error messages / seed ids to keep")
    parser.add_argument("--sample-k", type=int, default=20, help="Number of representative failure samples to keep")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = json.loads(input_path.read_text(encoding="utf-8"))
    records = report.get("records", [])

    failure_counter: Counter = Counter()
    by_pred_action_type: Counter = Counter()
    by_gold_action_type: Counter = Counter()
    error_messages: Counter = Counter()
    seed_failures: Counter = Counter()
    mismatch_by_action_type: Counter = Counter()
    execution_error_by_action_type: Counter = Counter()

    representative_samples: List[Dict[str, Any]] = []

    for record in records:
        failure_type = classify_failure(record)
        failure_counter[failure_type] += 1
        by_pred_action_type[str(record.get("pred_action_type"))] += 1
        by_gold_action_type[str(record.get("gold_action_type"))] += 1

        if failure_type != "matched":
            seed_failures[str(record.get("seed_id"))] += 1
            if len(representative_samples) < args.sample_k:
                representative_samples.append(
                    {
                        "id": record.get("id"),
                        "seed_id": record.get("seed_id"),
                        "turn_id": record.get("turn_id"),
                        "failure_type": failure_type,
                        "pred_action_type": record.get("pred_action_type"),
                        "gold_action_type": record.get("gold_action_type"),
                        "pred_error": shorten(record.get("pred_observation", {}).get("error_message", "")),
                        "gold_error": shorten(record.get("gold_observation", {}).get("error_message", "")),
                        "pred_sql": record.get("pred_sql", ""),
                        "gold_sql": record.get("gold_sql", ""),
                    }
                )

        if failure_type == "execution_error":
            execution_error_by_action_type[str(record.get("pred_action_type"))] += 1
            error_messages[shorten(record.get("pred_observation", {}).get("error_message", ""))] += 1

        if failure_type == "result_mismatch":
            mismatch_by_action_type[str(record.get("pred_action_type"))] += 1

    output = {
        "input": str(input_path),
        "metrics": report.get("metrics", {}),
        "record_count": len(records),
        "failure_breakdown": top_counter(failure_counter, args.top_k),
        "pred_action_type_distribution": top_counter(by_pred_action_type, args.top_k),
        "gold_action_type_distribution": top_counter(by_gold_action_type, args.top_k),
        "top_execution_errors": top_counter(error_messages, args.top_k),
        "top_failed_seed_ids": top_counter(seed_failures, args.top_k),
        "result_mismatch_by_action_type": top_counter(mismatch_by_action_type, args.top_k),
        "execution_error_by_action_type": top_counter(execution_error_by_action_type, args.top_k),
        "representative_failure_samples": representative_samples,
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["failure_breakdown"], ensure_ascii=False, indent=2))
    print(f"Saved failure analysis to: {output_path}")


if __name__ == "__main__":
    main()
