#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.rl_env import RLHistoryItem, Text2SQLRLEnv, parse_agent_output
from agent_rl.seed_builder import build_seed_records


OBSERVATION_RE = re.compile(r"<observation>\s*(.*?)\s*</observation>", re.IGNORECASE | re.DOTALL)


class NoopSQLEnv:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare single-step GRPO dataset from full-trajectory SFT data.")
    parser.add_argument(
        "--train-full-trajectory",
        default="output/llamafactory_sft_v5/train_full_trajectory.json",
        help="Full-trajectory train JSON exported for LLaMA-Factory.",
    )
    parser.add_argument(
        "--val-full-trajectory",
        default="output/llamafactory_sft_v5/val_full_trajectory.json",
        help="Full-trajectory val JSON exported for LLaMA-Factory.",
    )
    parser.add_argument(
        "--golden-path",
        default="golden_sql_marked.json",
        help="Path to golden_sql_marked.json.",
    )
    parser.add_argument(
        "--schema-path",
        default="schema.json",
        help="Path to schema.json.",
    )
    parser.add_argument(
        "--outdir",
        default="output/rl_training_inputs_v1",
        help="Output directory for RL training input dataset prepared from SFT trajectories.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_plain_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "系统说明：\n"
        + system_prompt.strip()
        + "\n\n用户输入：\n"
        + user_prompt.strip()
        + "\n\n请直接输出下一步动作，严格使用 <reasoning> 与 <sql> 或 <solution> 协议。"
    )


def parse_observation_message(content: str) -> Dict[str, Any]:
    match = OBSERVATION_RE.search(content or "")
    if not match:
        raise ValueError("Observation message does not contain <observation> block.")
    payload = json.loads(match.group(1).strip())
    return {
        "status": payload.get("status", "error"),
        "error_message": payload.get("error_message", ""),
        "columns": payload.get("columns", []) or [],
        "sample_rows": payload.get("sample_rows", []) or [],
        "row_count": int(payload.get("row_count", 0) or 0),
        "fingerprint": payload.get("fingerprint", ""),
        "turns_left": int(payload.get("turns_left", 0) or 0),
    }


def build_step_records(
    trajectories: List[Dict[str, Any]],
    seed_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    env = Text2SQLRLEnv(sql_env=NoopSQLEnv())
    records: List[Dict[str, Any]] = []

    for traj in trajectories:
        seed_id = traj.get("seed_id", "")
        seed = seed_map.get(seed_id)
        if not seed:
            continue

        state = env.reset(seed)
        messages = traj.get("messages", [])
        if not messages:
            continue

        msg_index = 1
        while msg_index < len(messages):
            message = messages[msg_index]
            if message.get("role") != "assistant":
                msg_index += 1
                continue

            gold_output = (message.get("content") or "").strip()
            gold_action = parse_agent_output(gold_output)
            system_prompt = env.build_system_prompt()
            user_prompt = env.build_user_prompt(state)
            history_payload = env.serialize_history(state.history)

            records.append(
                {
                    "id": f"{seed_id}_turn_{state.turn_index}",
                    "seed_id": seed_id,
                    "turn_id": state.turn_index,
                    "prompt": render_plain_prompt(system_prompt, user_prompt),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "difficulty": state.difficulty,
                    "max_turns": state.max_turns,
                    "remaining_turns": state.remaining_turns,
                    "history_json": json.dumps(history_payload, ensure_ascii=False),
                    "seed_json": json.dumps(seed, ensure_ascii=False),
                    "gold_output": gold_output,
                    "gold_action_type": gold_action.action_type,
                    "gold_sql": gold_action.sql,
                    "gold_result_json": json.dumps(seed.get("gold_result", []), ensure_ascii=False),
                    "reference_reasoning": gold_action.reasoning,
                }
            )

            next_index = msg_index + 1
            if next_index >= len(messages) or messages[next_index].get("role") != "user":
                break

            observation = parse_observation_message(messages[next_index].get("content", ""))
            state.history.append(
                RLHistoryItem(
                    turn_id=state.turn_index,
                    action={
                        "action_type": gold_action.action_type,
                        "reasoning": gold_action.reasoning,
                        "sql": gold_action.sql,
                    },
                    observation=observation,
                )
            )
            if gold_action.action_type == "sql":
                state.sql_probe_count += 1
                if observation["status"] in {"success", "empty"}:
                    state.successful_probe_count += 1
            msg_index += 2

    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_trajectories = load_json(Path(args.train_full_trajectory))
    val_trajectories = load_json(Path(args.val_full_trajectory))
    seeds = build_seed_records(args.golden_path, args.schema_path)
    seed_map = {seed["seed_id"]: seed for seed in seeds}

    train_records = build_step_records(train_trajectories, seed_map)
    val_records = build_step_records(val_trajectories, seed_map)

    outdir = Path(args.outdir)
    write_json(outdir / "train_rl_single_step.json", train_records)
    write_json(outdir / "val_rl_single_step.json", val_records)
    write_json(
        outdir / "prepare_report.json",
        {
            "train_trajectory_count": len(train_trajectories),
            "val_trajectory_count": len(val_trajectories),
            "train_step_count": len(train_records),
            "val_step_count": len(val_records),
            "unique_train_seeds": len({item["seed_id"] for item in train_records}),
            "unique_val_seeds": len({item["seed_id"] for item in val_records}),
        },
    )

    print(f"Saved RL train dataset to: {outdir / 'train_rl_single_step.json'}")
    print(f"Saved RL val dataset to: {outdir / 'val_rl_single_step.json'}")


if __name__ == "__main__":
    main()
