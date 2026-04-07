import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_rl.prompt_builder import SYSTEM_INSTRUCTION
from agent_rl.reward import DEFAULT_REWARD_CONFIG, RewardConfig, get_difficulty_budget, normalize_difficulty, compute_step_exec_reward, compute_terminal_rewards
from agent_rl.schemas import Observation, SeedRecord

if TYPE_CHECKING:
    from agent_rl.sql_env import SQLEnvironment


REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
SQL_RE = re.compile(r"<sql>(.*?)</sql>", re.IGNORECASE | re.DOTALL)
SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.IGNORECASE | re.DOTALL)


@dataclass
class RLAction:
    action_type: str
    reasoning: str
    sql: str
    raw_output: str


@dataclass
class RLHistoryItem:
    turn_id: int
    action: Dict[str, Any]
    observation: Observation


@dataclass
class RLEpisodeState:
    seed: SeedRecord
    difficulty: str
    max_turns: int
    history: List[RLHistoryItem] = field(default_factory=list)
    done: bool = False
    final_failure_reason: str = ""
    successful_probe_count: int = 0
    sql_probe_count: int = 0

    @property
    def turn_index(self) -> int:
        return len(self.history) + 1

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - len(self.history))

    @property
    def latest_observation(self) -> Optional[Observation]:
        if not self.history:
            return None
        return self.history[-1].observation


class ProtocolError(ValueError):
    pass


def parse_agent_output(text: str) -> RLAction:
    reasoning_match = REASONING_RE.search(text or "")
    sql_match = SQL_RE.search(text or "")
    solution_match = SOLUTION_RE.search(text or "")

    if not reasoning_match:
        raise ProtocolError("Missing <reasoning> tag.")
    if bool(sql_match) == bool(solution_match):
        raise ProtocolError("Output must contain exactly one of <sql> or <solution>.")

    action_type = "sql" if sql_match else "solution"
    sql_text = (sql_match or solution_match).group(1).strip()
    if not sql_text:
        raise ProtocolError("SQL body is empty.")

    return RLAction(
        action_type=action_type,
        reasoning=reasoning_match.group(1).strip(),
        sql=sql_text,
        raw_output=(text or "").strip(),
    )


class Text2SQLRLEnv:
    def __init__(
        self,
        sql_env: Optional["SQLEnvironment"] = None,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    ):
        if sql_env is None:
            from agent_rl.sql_env import SQLEnvironment

            sql_env = SQLEnvironment()
        self.sql_env = sql_env
        self.reward_config = reward_config

    def reset(self, seed: SeedRecord) -> RLEpisodeState:
        difficulty = normalize_difficulty(seed.get("difficulty", ""))
        budget = get_difficulty_budget(difficulty)
        return RLEpisodeState(
            seed=seed,
            difficulty=difficulty,
            max_turns=budget.max_turns,
        )

    def restore_state(
        self,
        seed: SeedRecord,
        history_payload: List[Dict[str, Any]] | None = None,
        difficulty: str | None = None,
    ) -> RLEpisodeState:
        difficulty_value = normalize_difficulty(difficulty or seed.get("difficulty", ""))
        budget = get_difficulty_budget(difficulty_value)
        state = RLEpisodeState(
            seed=seed,
            difficulty=difficulty_value,
            max_turns=budget.max_turns,
        )

        for raw_item in history_payload or []:
            action = raw_item.get("action", {}) or {}
            observation = raw_item.get("observation", {}) or {}
            history_item = RLHistoryItem(
                turn_id=int(raw_item.get("turn_id", len(state.history) + 1)),
                action={
                    "action_type": action.get("action_type", ""),
                    "reasoning": action.get("reasoning", ""),
                    "sql": action.get("sql", ""),
                },
                observation={
                    "status": observation.get("status", "error"),
                    "error_message": observation.get("error_message", ""),
                    "columns": observation.get("columns", []) or [],
                    "sample_rows": observation.get("sample_rows", []) or [],
                    "row_count": int(observation.get("row_count", 0) or 0),
                    "fingerprint": observation.get("fingerprint", ""),
                    "turns_left": int(observation.get("turns_left", 0) or 0),
                },
            )
            state.history.append(history_item)

            if history_item.action["action_type"] == "sql":
                state.sql_probe_count += 1
                if history_item.observation["status"] in {"success", "empty"}:
                    state.successful_probe_count += 1

        return state

    @staticmethod
    def serialize_history(history: List[RLHistoryItem]) -> List[Dict[str, Any]]:
        return [
            {
                "turn_id": item.turn_id,
                "action": {
                    "action_type": item.action["action_type"],
                    "reasoning": item.action["reasoning"],
                    "sql": item.action["sql"],
                },
                "observation": {
                    "status": item.observation["status"],
                    "error_message": item.observation["error_message"],
                    "columns": item.observation["columns"],
                    "sample_rows": item.observation["sample_rows"],
                    "row_count": item.observation["row_count"],
                    "fingerprint": item.observation["fingerprint"],
                    "turns_left": item.observation["turns_left"],
                },
            }
            for item in history
        ]

    def build_system_prompt(self) -> str:
        return SYSTEM_INSTRUCTION

    def build_user_prompt(self, state: RLEpisodeState) -> str:
        parts = [
            "任务目标：用尽量少但有效的验证/探索型 SQL 找到正确的最终 SQL。",
            f"Question:\n{state.seed['question']}",
            "Schema Snippets:\n" + "\n\n".join(state.seed["schema_snippets"]),
            f"Hard Constraints:\n{state.seed['hard_constraints']}",
        ]
        if state.seed["knowledge"]:
            parts.append(f"Knowledge:\n{state.seed['knowledge']}")
        parts.append(f"Difficulty: {state.difficulty}")
        parts.append(f"Turns Left: {state.remaining_turns}")
        parts.append("History:\n" + self._render_history(state.history))
        parts.append(
            "中间 <sql> 仅用于验证或探索当前不确定点。"
            " 若当前信息已经足够，请立即输出 <solution>。"
            " 如果这是最后一轮，你必须输出 <solution>。"
        )
        return "\n\n".join(parts)

    @staticmethod
    def _render_history(history: List[RLHistoryItem]) -> str:
        if not history:
            return "暂无历史。"
        blocks: List[str] = []
        for item in history:
            action = item.action
            tag = "sql" if action["action_type"] == "sql" else "solution"
            compact_observation = {
                "status": item.observation["status"],
                "error_message": item.observation["error_message"],
                "columns": item.observation["columns"],
                "sample_rows": item.observation["sample_rows"][:3],
                "row_count": item.observation["row_count"],
                "turns_left": item.observation["turns_left"],
            }
            blocks.append(
                "\n".join(
                    [
                        f"Turn {item.turn_id}",
                        f"<reasoning>{action['reasoning']}</reasoning>",
                        f"<{tag}>{action['sql']}</{tag}>",
                        "<observation>",
                        json.dumps(compact_observation, ensure_ascii=False, separators=(",", ":")),
                        "</observation>",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def step(self, state: RLEpisodeState, model_output: str) -> Dict[str, Any]:
        if state.done:
            raise RuntimeError("Episode is already done.")

        reward_breakdown: Dict[str, float] = {
            "r_format": 0.0,
            "r_exec_step": 0.0,
            "r_final_result": 0.0,
            "r_sql_ngram": 0.0,
            "r_turn": 0.0,
            "r_final_exec_fail": 0.0,
        }

        try:
            action = parse_agent_output(model_output)
        except ProtocolError as exc:
            state.done = True
            state.final_failure_reason = f"protocol_error: {exc}"
            reward_breakdown["r_format"] = self.reward_config.format_error_penalty
            return {
                "state": state,
                "done": True,
                "action": None,
                "observation": None,
                "reward_breakdown": reward_breakdown,
                "reward": sum(reward_breakdown.values()),
                "final_rows": [],
            }

        turn_id = state.turn_index
        turns_left_after_step = max(0, state.max_turns - turn_id)

        if turn_id == state.max_turns and action.action_type == "sql":
            observation, rows = self.sql_env.execute_with_rows(
                action.sql,
                turns_left=turns_left_after_step,
                sql_id=f"{state.seed['seed_id']}_t{turn_id}",
            )
            history_item = RLHistoryItem(
                turn_id=turn_id,
                action={"action_type": action.action_type, "reasoning": action.reasoning, "sql": action.sql},
                observation=observation,
            )
            state.history.append(history_item)
            state.sql_probe_count += 1
            reward_breakdown["r_exec_step"] = compute_step_exec_reward(
                action_type="sql",
                observation_status=observation["status"],
                successful_probe_count_before_step=state.successful_probe_count,
                config=self.reward_config,
            )
            if observation["status"] in {"success", "empty"}:
                state.successful_probe_count += 1
            state.done = True
            state.final_failure_reason = "last_turn_still_probe"
            terminal = compute_terminal_rewards(
                final_action_type="sql",
                final_exec_success=False,
                pred_rows=[],
                gold_rows=state.seed["gold_result"],
                pred_sql=action.sql,
                gold_sql=state.seed["gold_sql"],
                sql_probe_count=state.sql_probe_count,
                difficulty=state.difficulty,
                config=self.reward_config,
            )
            reward_breakdown.update(terminal)
            return {
                "state": state,
                "done": True,
                "action": action,
                "observation": observation,
                "reward_breakdown": reward_breakdown,
                "reward": sum(reward_breakdown.values()),
                "final_rows": [],
            }

        observation, rows = self.sql_env.execute_with_rows(
            action.sql,
            turns_left=turns_left_after_step,
            sql_id=f"{state.seed['seed_id']}_t{turn_id}",
        )
        history_item = RLHistoryItem(
            turn_id=turn_id,
            action={"action_type": action.action_type, "reasoning": action.reasoning, "sql": action.sql},
            observation=observation,
        )
        state.history.append(history_item)

        if action.action_type == "sql":
            reward_breakdown["r_exec_step"] = compute_step_exec_reward(
                action_type="sql",
                observation_status=observation["status"],
                successful_probe_count_before_step=state.successful_probe_count,
                config=self.reward_config,
            )
            state.sql_probe_count += 1
            if observation["status"] in {"success", "empty"}:
                state.successful_probe_count += 1

        done = action.action_type == "solution" or turn_id >= state.max_turns
        state.done = done

        if done:
            final_exec_success = observation["status"] in {"success", "empty"} if action.action_type == "solution" else False
            terminal = compute_terminal_rewards(
                final_action_type=action.action_type,
                final_exec_success=final_exec_success,
                pred_rows=rows if action.action_type == "solution" else [],
                gold_rows=state.seed["gold_result"],
                pred_sql=action.sql,
                gold_sql=state.seed["gold_sql"],
                sql_probe_count=state.sql_probe_count,
                difficulty=state.difficulty,
                config=self.reward_config,
            )
            reward_breakdown.update(terminal)
            if action.action_type == "solution":
                if not final_exec_success:
                    state.final_failure_reason = "final_exec_fail"
                elif reward_breakdown["r_final_result"] <= 0:
                    state.final_failure_reason = "final_result_mismatch"
                else:
                    state.final_failure_reason = ""

        return {
            "state": state,
            "done": done,
            "action": action,
            "observation": observation,
            "reward_breakdown": reward_breakdown,
            "reward": sum(reward_breakdown.values()),
            "final_rows": rows if action.action_type == "solution" else [],
        }
