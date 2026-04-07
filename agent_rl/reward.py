import re
from dataclasses import dataclass
from typing import Dict, Optional

from agent_rl.result_matcher import is_result_match


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|<=|>=|!=|=|<|>|\(|\)|,|'[^']*'|\"[^\"]*\"|\d+")


@dataclass(frozen=True)
class DifficultyBudget:
    max_turns: int
    free_probes: int
    turn_penalty_lambda: float


@dataclass(frozen=True)
class RewardConfig:
    final_result_weight: float = 5.0
    sql_ngram_weight: float = 0.3
    first_successful_probe_reward: float = 0.2
    failed_probe_penalty: float = -0.3
    format_error_penalty: float = -1.0
    final_exec_fail_penalty: float = -0.5


DEFAULT_REWARD_CONFIG = RewardConfig()

DIFFICULTY_BUDGETS: Dict[str, DifficultyBudget] = {
    "easy": DifficultyBudget(max_turns=2, free_probes=0, turn_penalty_lambda=0.5),
    "medium": DifficultyBudget(max_turns=3, free_probes=1, turn_penalty_lambda=0.3),
    "hard": DifficultyBudget(max_turns=4, free_probes=2, turn_penalty_lambda=0.15),
}

DIFFICULTY_ALIASES = {
    "简单": "easy",
    "easy": "easy",
    "中等": "medium",
    "中": "medium",
    "medium": "medium",
    "困难": "hard",
    "复杂": "hard",
    "hard": "hard",
    "extra": "hard",
}


def normalize_difficulty(value: str) -> str:
    difficulty = (value or "").strip().lower()
    return DIFFICULTY_ALIASES.get(difficulty, "medium")


def get_difficulty_budget(value: str) -> DifficultyBudget:
    return DIFFICULTY_BUDGETS[normalize_difficulty(value)]


def _tokenize_sql(sql_text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(sql_text or "")]


def sql_ngram_similarity(pred_sql: str, gold_sql: str, n: int = 3) -> float:
    pred_tokens = _tokenize_sql(pred_sql)
    gold_tokens = _tokenize_sql(gold_sql)
    if not pred_tokens or not gold_tokens:
        return 0.0

    def ngrams(tokens: list[str]) -> set[tuple[str, ...]]:
        if len(tokens) < n:
            return {tuple(tokens)} if tokens else set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    pred_ngrams = ngrams(pred_tokens)
    gold_ngrams = ngrams(gold_tokens)
    if not pred_ngrams or not gold_ngrams:
        return 0.0
    intersection = len(pred_ngrams & gold_ngrams)
    union = len(pred_ngrams | gold_ngrams)
    return intersection / union if union else 0.0


def compute_step_exec_reward(
    *,
    action_type: str,
    observation_status: str,
    successful_probe_count_before_step: int,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if action_type != "sql":
        return 0.0

    if observation_status in {"success", "empty"}:
        if successful_probe_count_before_step == 0:
            return config.first_successful_probe_reward
        return 0.0
    return config.failed_probe_penalty


def compute_turn_reward(sql_probe_count: int, difficulty: str) -> float:
    budget = get_difficulty_budget(difficulty)
    overflow = max(0, sql_probe_count - budget.free_probes)
    return -budget.turn_penalty_lambda * overflow


def compute_terminal_rewards(
    *,
    final_action_type: Optional[str],
    final_exec_success: bool,
    pred_rows: list[dict],
    gold_rows: list[dict],
    pred_sql: str,
    gold_sql: str,
    sql_probe_count: int,
    difficulty: str,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> Dict[str, float]:
    final_result = 1.0 if final_exec_success and is_result_match(pred_rows, gold_rows) else 0.0
    sql_ngram = sql_ngram_similarity(pred_sql, gold_sql)
    turn_reward = compute_turn_reward(sql_probe_count, difficulty)
    final_exec_fail = config.final_exec_fail_penalty if final_action_type == "solution" and not final_exec_success else 0.0

    return {
        "r_final_result": config.final_result_weight * final_result,
        "r_sql_ngram": config.sql_ngram_weight * sql_ngram,
        "r_turn": turn_reward,
        "r_final_exec_fail": final_exec_fail,
    }
