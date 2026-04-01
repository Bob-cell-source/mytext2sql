from typing import Any, Dict, List, Literal, TypedDict


ActionType = Literal["sql", "solution"]
ObservationStatus = Literal["success", "empty", "error", "timeout"]


class Observation(TypedDict):
    status: ObservationStatus
    error_message: str
    columns: List[str]
    sample_rows: List[Dict[str, Any]]
    row_count: int
    fingerprint: str
    turns_left: int


class TrajectoryTurn(TypedDict):
    turn_id: int
    action_type: ActionType
    reasoning: str
    sql: str
    observation: Observation


class SeedRecord(TypedDict):
    seed_id: str
    question: str
    table_list: List[str]
    knowledge: str
    difficulty: str
    gold_sql: str
    gold_result: List[Dict[str, Any]]
    gold_result_fingerprint: str
    schema_snippets: List[str]
    hard_constraints: str


class TrajectoryMeta(TypedDict):
    final_status: str
    final_result_match: bool
    turn_count: int
    timeout_count: int
    error_count: int
    repeated_probe_count: int
    info_gain_score: float
    score: float
    reject_reason: str


class TrajectoryRecord(TypedDict):
    seed_id: str
    question: str
    schema_snippets: List[str]
    hard_constraints: str
    knowledge: str
    turns: List[TrajectoryTurn]
    meta: TrajectoryMeta

