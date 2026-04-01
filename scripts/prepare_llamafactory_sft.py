#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.IGNORECASE | re.DOTALL)
SQL_RE = re.compile(r"<sql>.*?</sql>", re.IGNORECASE | re.DOTALL)
SOLUTION_RE = re.compile(r"<solution>.*?</solution>", re.IGNORECASE | re.DOTALL)

DEFAULT_INSTRUCTION = "你是一个 StarRocks Text2SQL agent。请根据给定上下文输出下一步动作，严格使用 <reasoning> 与 <sql> 或 <solution> 协议。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT data for LLaMA-Factory.")
    parser.add_argument(
        "--input",
        default="output/sft_synthesis_full_v2/sft_multiturn_action_focused.jsonl",
        help="Input action-focused JSONL file.",
    )
    parser.add_argument(
        "--full-input",
        default="output/sft_synthesis_full_v2/sft_multiturn_full.jsonl",
        help="Input full-trajectory JSONL file for observation-explicit export.",
    )
    parser.add_argument(
        "--outdir",
        default="output/llamafactory_sft",
        help="Directory for cleaned train/val data.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation seed ratio.")
    parser.add_argument("--min-val-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_observation_block(observation: Dict | None) -> Dict | None:
    if not observation:
        return None
    compact = {
        "status": observation.get("status", ""),
        "error_message": observation.get("error_message", ""),
        "columns": observation.get("columns", []),
        "sample_rows": (observation.get("sample_rows", []) or [])[:3],
        "row_count": observation.get("row_count", 0),
        "turns_left": observation.get("turns_left", 0),
    }
    return compact


def infer_resolved_uncertainty(sql_text: str) -> str:
    sql_lower = sql_text.lower()
    if " join " in sql_lower:
        return "join_path"
    if "dtstatdate" in sql_lower or "iregdate" in sql_lower or "ilastactdate" in sql_lower or "kp_imp_date" in sql_lower:
        return "date_scope"
    if "distinct" in sql_lower or "count(" in sql_lower:
        return "target_set"
    if " in (" in sql_lower or "group by" in sql_lower:
        return "value_domain"
    return "none"


def infer_probe_quality(turn: Dict, previous_turn: Dict | None = None) -> str:
    if not turn:
        return "none"
    observation = turn.get("observation", {})
    status = observation.get("status", "")
    sql_text = turn.get("sql", "")

    if status in ("error", "timeout"):
        return "failed"
    if status == "empty":
        return "weak"

    if previous_turn and previous_turn.get("observation", {}).get("fingerprint") == observation.get("fingerprint"):
        return "weak"

    resolved = infer_resolved_uncertainty(sql_text)
    if resolved != "none":
        return "useful"
    if observation.get("row_count", 0) <= 5 and observation.get("columns"):
        return "useful"
    return "weak"


def build_observation_explicit_samples(full_rows: List[Dict], instruction: str) -> List[Dict]:
    samples: List[Dict] = []
    for traj_idx, traj in enumerate(full_rows):
        history_actions: List[Dict] = []
        previous_turn = None
        for turn in traj.get("turns", []):
            latest_observation = render_observation_block(previous_turn.get("observation")) if previous_turn else None
            sample = {
                "id": f"{traj['seed_id']}_traj_{traj_idx}_turn_{turn['turn_id']}_explicit",
                "seed_id": traj["seed_id"],
                "turn_id": turn["turn_id"],
                "instruction": instruction,
                "input": json.dumps(
                    {
                        "question": traj.get("question", ""),
                        "schema_snippets": traj.get("schema_snippets", []),
                        "knowledge": traj.get("knowledge", ""),
                        "hard_constraints": traj.get("hard_constraints", ""),
                        "history_actions": history_actions,
                        "latest_observation": latest_observation,
                        "latest_probe_quality": infer_probe_quality(previous_turn) if previous_turn else "none",
                        "latest_resolved_uncertainty": infer_resolved_uncertainty(previous_turn.get("sql", "")) if previous_turn else "none",
                        "turns_left": previous_turn.get("observation", {}).get("turns_left", 0) if previous_turn else traj.get("meta", {}).get("turn_count", 0),
                    },
                    ensure_ascii=False,
                ),
                "output": render_response(turn),
                "action_type": turn.get("action_type", ""),
            }
            samples.append(sample)
            history_actions.append(
                {
                    "turn_id": turn.get("turn_id"),
                    "action_type": turn.get("action_type"),
                    "reasoning": turn.get("reasoning", ""),
                    "sql": turn.get("sql", ""),
                }
            )
            previous_turn = turn
    return samples


def valid_response(text: str) -> Tuple[bool, str]:
    has_reasoning = bool(REASONING_RE.search(text))
    has_sql = bool(SQL_RE.search(text))
    has_solution = bool(SOLUTION_RE.search(text))

    if not has_reasoning:
        return False, "missing_reasoning"
    if has_sql and has_solution:
        return False, "both_sql_and_solution"
    if not has_sql and not has_solution:
        return False, "missing_sql_or_solution"
    return True, ""


def build_initial_user_message(traj: Dict) -> str:
    parts = [
        "你将解决一个多轮 Text2SQL agent 任务。请在后续轮次中根据 observation 决定下一步动作。",
        f"Question:\n{traj.get('question', '')}",
        "Schema Snippets:\n" + "\n\n".join(traj.get("schema_snippets", [])),
        f"Hard Constraints:\n{traj.get('hard_constraints', '')}",
    ]
    knowledge = (traj.get("knowledge") or "").strip()
    if knowledge:
        parts.append(f"Knowledge:\n{knowledge}")
    return "\n\n".join(parts)


def build_observation_user_message(turn: Dict) -> str:
    observation = render_observation_block(turn.get("observation"))
    return "<observation>\n" + json.dumps(observation, ensure_ascii=False, indent=2) + "\n</observation>"


def normalize_record(record: Dict, instruction: str) -> Dict:
    return {
        "id": f"{record['seed_id']}_turn_{record['turn_id']}",
        "seed_id": record["seed_id"],
        "turn_id": record["turn_id"],
        "instruction": instruction,
        "input": record["prompt"].strip(),
        "output": record["response"].strip(),
        "action_type": record.get("action_type", ""),
    }


def render_response(turn: Dict) -> str:
    action_type = turn.get("action_type", "sql")
    tag = "sql" if action_type == "sql" else "solution"
    return f"<reasoning>{turn.get('reasoning', '').strip()}</reasoning>\n<{tag}>{turn.get('sql', '').strip()}</{tag}>"


def build_full_trajectory_conversations(full_rows: List[Dict], system_prompt: str) -> List[Dict]:
    conversations = []
    for traj_idx, traj in enumerate(full_rows):
        turns = traj.get("turns", [])
        if not turns:
            continue

        messages = [{"role": "user", "content": build_initial_user_message(traj)}]
        for turn_idx, turn in enumerate(turns):
            messages.append({"role": "assistant", "content": render_response(turn)})
            if turn_idx < len(turns) - 1:
                messages.append({"role": "user", "content": build_observation_user_message(turn)})

        conversations.append(
            {
                "id": f"{traj['seed_id']}_traj_{traj_idx}_full",
                "seed_id": traj["seed_id"],
                "messages": messages,
                "system": system_prompt,
            }
        )
    return conversations


def stable_seed_bucket(seed_id: str, seed: int) -> int:
    payload = f"{seed}:{seed_id}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16)


def split_seed_ids(seed_ids: List[str], val_ratio: float, min_val_seeds: int, seed: int) -> Tuple[set[str], set[str]]:
    ordered = sorted(seed_ids, key=lambda x: stable_seed_bucket(x, seed))
    val_count = max(min_val_seeds, int(round(len(ordered) * val_ratio)))
    val_count = min(val_count, max(1, len(ordered) - 1)) if len(ordered) > 1 else len(ordered)
    val_ids = set(ordered[:val_count])
    train_ids = set(ordered[val_count:])
    return train_ids, val_ids


def write_json(path: Path, payload: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    full_input_path = Path(args.full_input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_jsonl(input_path)
    full_rows = load_jsonl(full_input_path) if full_input_path.exists() else []

    cleaned_rows: List[Dict] = []
    dropped_counts: Dict[str, int] = {}
    dedupe_keys = set()

    for row in raw_rows:
        prompt = row.get("prompt", "").strip()
        response = row.get("response", "").strip()
        seed_id = row.get("seed_id", "").strip()

        if not seed_id:
            dropped_counts["missing_seed_id"] = dropped_counts.get("missing_seed_id", 0) + 1
            continue
        if not prompt:
            dropped_counts["empty_prompt"] = dropped_counts.get("empty_prompt", 0) + 1
            continue
        if not response:
            dropped_counts["empty_response"] = dropped_counts.get("empty_response", 0) + 1
            continue

        ok, reason = valid_response(response)
        if not ok:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1
            continue

        dedupe_key = (seed_id, row.get("turn_id"), prompt, response)
        if dedupe_key in dedupe_keys:
            dropped_counts["duplicate"] = dropped_counts.get("duplicate", 0) + 1
            continue
        dedupe_keys.add(dedupe_key)

        cleaned_rows.append(normalize_record(row, args.instruction))

    explicit_rows = build_observation_explicit_samples(full_rows, args.instruction)
    explicit_cleaned_rows: List[Dict] = []
    explicit_dedupe_keys = set()
    explicit_dropped_duplicate = 0
    for row in explicit_rows:
        dedupe_key = (row["seed_id"], row["turn_id"], row["input"], row["output"])
        if dedupe_key in explicit_dedupe_keys:
            explicit_dropped_duplicate += 1
            continue
        explicit_dedupe_keys.add(dedupe_key)
        explicit_cleaned_rows.append(row)

    seed_ids = sorted({row["seed_id"] for row in cleaned_rows})
    train_seed_ids, val_seed_ids = split_seed_ids(
        seed_ids=seed_ids,
        val_ratio=args.val_ratio,
        min_val_seeds=args.min_val_seeds,
        seed=args.seed,
    )

    train_rows = [row for row in cleaned_rows if row["seed_id"] in train_seed_ids]
    val_rows = [row for row in cleaned_rows if row["seed_id"] in val_seed_ids]

    write_json(outdir / "train.json", train_rows)
    write_json(outdir / "val.json", val_rows)

    explicit_train_rows = [row for row in explicit_cleaned_rows if row["seed_id"] in train_seed_ids]
    explicit_val_rows = [row for row in explicit_cleaned_rows if row["seed_id"] in val_seed_ids]
    write_json(outdir / "train_observation_explicit.json", explicit_train_rows)
    write_json(outdir / "val_observation_explicit.json", explicit_val_rows)

    full_conversations = build_full_trajectory_conversations(full_rows, args.instruction)
    full_conversation_cleaned = []
    full_dedupe = set()
    full_dropped_duplicate = 0
    for row in full_conversations:
        dedupe_key = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        if dedupe_key in full_dedupe:
            full_dropped_duplicate += 1
            continue
        full_dedupe.add(dedupe_key)
        full_conversation_cleaned.append(row)

    full_train_rows = [row for row in full_conversation_cleaned if row["seed_id"] in train_seed_ids]
    full_val_rows = [row for row in full_conversation_cleaned if row["seed_id"] in val_seed_ids]
    write_json(outdir / "train_full_trajectory.json", full_train_rows)
    write_json(outdir / "val_full_trajectory.json", full_val_rows)

    dataset_info = {
        "text2sql_agent_train": {
            "file_name": "train.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "text2sql_agent_val": {
            "file_name": "val.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "text2sql_agent_train_observation_explicit": {
            "file_name": "train_observation_explicit.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "text2sql_agent_val_observation_explicit": {
            "file_name": "val_observation_explicit.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "text2sql_agent_train_full_trajectory": {
            "file_name": "train_full_trajectory.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "system": "system"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        },
        "text2sql_agent_val_full_trajectory": {
            "file_name": "val_full_trajectory.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "system": "system"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        },
    }
    write_json(outdir / "dataset_info.json", dataset_info)  # type: ignore[arg-type]

    report = {
        "input_path": str(input_path),
        "full_input_path": str(full_input_path),
        "raw_sample_count": len(raw_rows),
        "cleaned_sample_count": len(cleaned_rows),
        "observation_explicit_raw_sample_count": len(explicit_rows),
        "observation_explicit_sample_count": len(explicit_cleaned_rows),
        "full_trajectory_raw_sample_count": len(full_conversations),
        "full_trajectory_sample_count": len(full_conversation_cleaned),
        "dropped_counts": dropped_counts,
        "observation_explicit_dropped_counts": {
            "duplicate": explicit_dropped_duplicate,
        },
        "full_trajectory_dropped_counts": {
            "duplicate": full_dropped_duplicate,
        },
        "unique_seed_count": len(seed_ids),
        "train_seed_count": len(train_seed_ids),
        "val_seed_count": len(val_seed_ids),
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "train_observation_explicit_sample_count": len(explicit_train_rows),
        "val_observation_explicit_sample_count": len(explicit_val_rows),
        "train_full_trajectory_sample_count": len(full_train_rows),
        "val_full_trajectory_sample_count": len(full_val_rows),
        "train_seed_ids": sorted(train_seed_ids),
        "val_seed_ids": sorted(val_seed_ids),
    }
    (outdir / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
