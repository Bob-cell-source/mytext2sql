#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


SOLUTION_START = "<solution>"
SOLUTION_END = "</solution>"
DEFAULT_INSTRUCTION = "你是一个 StarRocks Text2SQL 助手。请根据给定上下文直接输出最终 SQL。不要输出推理，不要输出额外解释，不要输出除 SQL 之外的任何内容。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a final-SQL baseline dataset from full-trajectory data.")
    parser.add_argument(
        "--train-input",
        default="output/llamafactory_sft_v5/train_full_trajectory.json",
        help="Input train full trajectory JSON.",
    )
    parser.add_argument(
        "--val-input",
        default="output/llamafactory_sft_v5/val_full_trajectory.json",
        help="Input val full trajectory JSON.",
    )
    parser.add_argument(
        "--outdir",
        default="output/llamafactory_baselines_v1",
        help="Output directory for baseline train/val files.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction string for the baseline dataset.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_solution_sql(text: str) -> str:
    if SOLUTION_START in text and SOLUTION_END in text:
        return text.split(SOLUTION_START, 1)[1].split(SOLUTION_END, 1)[0].strip()
    return text.strip()


def build_baseline_samples(rows: List[Dict], instruction: str) -> List[Dict]:
    samples: List[Dict] = []
    for row in rows:
        messages = row.get("messages", [])
        if len(messages) < 2:
            continue

        initial_user = messages[0].get("content", "").strip()
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_messages:
            continue

        final_solution = extract_solution_sql(assistant_messages[-1].get("content", ""))
        if not final_solution:
            continue

        samples.append(
            {
                "id": f"{row['id']}_final_sql",
                "seed_id": row.get("seed_id", ""),
                "instruction": instruction,
                "input": initial_user,
                "output": final_solution,
                "task_type": "final_sql_baseline",
            }
        )
    return samples


def write_json(path: Path, payload: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_rows = load_json(Path(args.train_input))
    val_rows = load_json(Path(args.val_input))

    train_samples = build_baseline_samples(train_rows, args.instruction)
    val_samples = build_baseline_samples(val_rows, args.instruction)

    write_json(outdir / "train_final_sql.json", train_samples)
    write_json(outdir / "val_final_sql.json", val_samples)

    dataset_info = {
        "text2sql_final_sql_baseline_train": {
            "file_name": "train_final_sql.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "text2sql_final_sql_baseline_val": {
            "file_name": "val_final_sql.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    (outdir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "train_input": args.train_input,
        "val_input": args.val_input,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "instruction": args.instruction,
    }
    (outdir / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
