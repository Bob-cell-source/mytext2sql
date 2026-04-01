import json
from pathlib import Path
from typing import Any, Dict, List

from agent_rl.prompt_builder import build_teacher_user_prompt, render_assistant_action
from agent_rl.schemas import SeedRecord, TrajectoryRecord


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pack_full_trajectories(trajectories: List[TrajectoryRecord], output_path: str) -> None:
    _write_jsonl(Path(output_path), trajectories)


def pack_action_focused_samples(
    seeds_by_id: Dict[str, SeedRecord],
    trajectories: List[TrajectoryRecord],
    output_path: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    for trajectory in trajectories:
        seed = seeds_by_id[trajectory["seed_id"]]
        history = []
        for turn in trajectory["turns"]:
            prompt = build_teacher_user_prompt(seed, history, turns_left=turn["observation"]["turns_left"])
            response = render_assistant_action(turn["action_type"], turn["reasoning"], turn["sql"])
            rows.append(
                {
                    "seed_id": trajectory["seed_id"],
                    "turn_id": turn["turn_id"],
                    "prompt": prompt,
                    "response": response,
                    "action_type": turn["action_type"],
                }
            )
            history.append(turn)
    _write_jsonl(Path(output_path), rows)


def write_synthesis_report(report: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
