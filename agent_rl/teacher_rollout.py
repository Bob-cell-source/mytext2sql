import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional,Dict

from openai import OpenAI

from agent_rl.prompt_builder import SYSTEM_INSTRUCTION, build_teacher_user_prompt, render_assistant_action
from agent_rl.result_matcher import is_result_match
from agent_rl.schemas import Observation, SeedRecord, TrajectoryMeta, TrajectoryRecord, TrajectoryTurn
from agent_rl.sql_env import SQLEnvironment


REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
SQL_RE = re.compile(r"<sql>(.*?)</sql>", re.IGNORECASE | re.DOTALL)
SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedAction:
    action_type: str
    reasoning: str
    sql: str


def _fallback_api_key() -> str:
    try:
        from final_code import API_KEY  # pylint: disable=import-outside-toplevel

        return API_KEY
    except Exception:
        return ""


def get_teacher_client_and_model() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or _fallback_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or DASHSCOPE_API_KEY for teacher rollout.")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv("LLM_MODEL", "qwen3-max")
    return OpenAI(api_key=api_key, base_url=base_url), model_name


def parse_teacher_output(text: str) -> ParsedAction:
    reasoning_match = REASONING_RE.search(text)
    sql_match = SQL_RE.search(text)
    solution_match = SOLUTION_RE.search(text)

    if sql_match and solution_match:
        raise ValueError("Teacher output contains both <sql> and <solution>.")
    if not reasoning_match:
        raise ValueError("Teacher output is missing <reasoning>.")
    if not sql_match and not solution_match:
        raise ValueError("Teacher output is missing <sql> or <solution>.")

    action_type = "sql" if sql_match else "solution"
    sql = (sql_match or solution_match).group(1).strip()
    return ParsedAction(action_type=action_type, reasoning=reasoning_match.group(1).strip(), sql=sql)


class TeacherRolloutRunner:
    def __init__(
        self,
        env: SQLEnvironment,
        max_turns: int = 3,
        max_errors: int = 2,
        max_timeouts: int = 1,
        temperature: float = 0.2,
    ):
        self.env = env
        self.max_turns = max_turns
        self.max_errors = max_errors
        self.max_timeouts = max_timeouts
        self.temperature = temperature
        self.client, self.model = get_teacher_client_and_model()

    def _call_teacher(self, seed: SeedRecord, turns: List[TrajectoryTurn], turns_left: int) -> ParsedAction:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": build_teacher_user_prompt(seed, turns, turns_left)},
            ],
            temperature=self.temperature,
            stream=False,
        )
        content = response.choices[0].message.content.strip()
        return parse_teacher_output(content)

    def rollout_once(self, seed: SeedRecord) -> TrajectoryRecord:
        turns: List[TrajectoryTurn] = []
        error_count = 0
        timeout_count = 0
        reject_reason = ""

        for turn_idx in range(1, self.max_turns + 1):
            turns_left = self.max_turns - turn_idx
            try:
                action = self._call_teacher(seed, turns, turns_left)
            except Exception as exc:
                reject_reason = f"protocol_error: {exc}"
                break

            if action.action_type == "solution":
                observation, final_rows = self.env.execute_with_rows(
                    action.sql, turns_left=turns_left, sql_id=f"{seed['seed_id']}_t{turn_idx}"
                )
            else:
                observation = self.env.execute(action.sql, turns_left=turns_left, sql_id=f"{seed['seed_id']}_t{turn_idx}")
                final_rows = []
            turn: TrajectoryTurn = {
                "turn_id": turn_idx,
                "action_type": action.action_type,
                "reasoning": action.reasoning,
                "sql": action.sql,
                "observation": observation,
            }
            turns.append(turn)

            if observation["status"] == "error":
                error_count += 1
            if observation["status"] == "timeout":
                timeout_count += 1
            if error_count > self.max_errors:
                reject_reason = "too_many_errors"
                break
            if timeout_count > self.max_timeouts:
                reject_reason = "too_many_timeouts"
                break
            if action.action_type == "solution":
                break

        final_status = turns[-1]["observation"]["status"] if turns else "error"
        final_result_match = False
        if turns and turns[-1]["action_type"] == "solution":
            final_result_match = is_result_match(final_rows, seed["gold_result"])

        meta: TrajectoryMeta = {
            "final_status": final_status,
            "final_result_match": final_result_match,
            "turn_count": len(turns),
            "timeout_count": timeout_count,
            "error_count": error_count,
            "repeated_probe_count": 0,
            "info_gain_score": 0.0,
            "score": 0.0,
            "reject_reason": reject_reason,
        }
        return {
            "seed_id": seed["seed_id"],
            "question": seed["question"],
            "schema_snippets": seed["schema_snippets"],
            "hard_constraints": seed["hard_constraints"],
            "knowledge": seed["knowledge"],
            "turns": turns,
            "meta": meta,
        }

    def rollout_many(self, seed: SeedRecord, attempts: int) -> List[TrajectoryRecord]:
        return [self.rollout_once(seed) for _ in range(attempts)]


def render_training_pair(seed: SeedRecord, turns: List[TrajectoryTurn], current_action: ParsedAction, turns_left: int) -> Dict[str, str]:
    return {
        "prompt": build_teacher_user_prompt(seed, turns, turns_left),
        "response": render_assistant_action(current_action.action_type, current_action.reasoning, current_action.sql),
    }
