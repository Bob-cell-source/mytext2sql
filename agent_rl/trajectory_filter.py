from typing import List, Tuple

from agent_rl.schemas import TrajectoryRecord


def _count_repeated_probe_fingerprints(trajectory: TrajectoryRecord) -> int:
    seen = set()
    repeated = 0
    for turn in trajectory["turns"]:
        if turn["action_type"] != "sql":
            continue
        fp = turn["observation"]["fingerprint"]
        if fp and fp in seen:
            repeated += 1
        seen.add(fp)
    return repeated


def _compute_info_gain_score(trajectory: TrajectoryRecord) -> float:
    unique_probe_results = {
        turn["observation"]["fingerprint"]
        for turn in trajectory["turns"]
        if turn["action_type"] == "sql" and turn["observation"]["status"] in ("success", "empty")
    }
    return float(len([fp for fp in unique_probe_results if fp]))


def _shortness_score(turn_count: int) -> float:
    if turn_count <= 1:
        return 1.5
    if turn_count == 2:
        return 1.0
    if turn_count == 3:
        return 0.5
    return 0.0


def evaluate_trajectory(trajectory: TrajectoryRecord, max_turns: int) -> Tuple[bool, str, float]:
    turns = trajectory["turns"]
    meta = trajectory["meta"]

    if not turns:
        return False, "empty_trajectory", 0.0
    if len(turns) > max_turns:
        return False, "too_many_turns", 0.0
    if turns[-1]["action_type"] != "solution":
        return False, "missing_solution", 0.0
    if not meta["final_result_match"]:
        return False, "final_result_mismatch", 0.0
    for turn in turns:
        if turn["observation"]["status"] == "timeout":
            return False, "timeout", 0.0

    repeated_probe_count = _count_repeated_probe_fingerprints(trajectory)
    info_gain_score = _compute_info_gain_score(trajectory)
    domain_compliance_score = 1.0 if meta["error_count"] == 0 else 0.4
    safety_score = 1.0 if meta["timeout_count"] == 0 else 0.0
    non_redundancy_score = max(0.0, 1.0 - repeated_probe_count * 0.5)

    score = (
        3.0
        + 1.0 * _shortness_score(len(turns))
        + 1.5 * info_gain_score
        + 1.0 * non_redundancy_score
        + 1.0 * domain_compliance_score
        + 0.5 * safety_score
    )

    meta["repeated_probe_count"] = repeated_probe_count
    meta["info_gain_score"] = info_gain_score
    meta["score"] = score
    meta["reject_reason"] = ""
    return True, "", score


def filter_and_rank_trajectories(
    trajectories: List[TrajectoryRecord],
    max_turns: int,
    keep_top_k: int,
) -> Tuple[List[TrajectoryRecord], List[TrajectoryRecord], List[TrajectoryRecord]]:
    eligible: List[TrajectoryRecord] = []
    rejected: List[TrajectoryRecord] = []

    for trajectory in trajectories:
        ok, reason, _ = evaluate_trajectory(trajectory, max_turns=max_turns)
        if ok:
            eligible.append(trajectory)
        else:
            trajectory["meta"]["reject_reason"] = reason
            rejected.append(trajectory)

    eligible.sort(key=lambda x: x["meta"]["score"], reverse=True)
    kept = eligible[:keep_top_k]
    dropped_after_ranking = eligible[keep_top_k:]
    for trajectory in dropped_after_ranking:
        trajectory["meta"]["reject_reason"] = "dropped_after_ranking"
    return kept, rejected, dropped_after_ranking
