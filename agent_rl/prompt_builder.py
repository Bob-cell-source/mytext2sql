import json
from typing import List

from agent_rl.schemas import SeedRecord, TrajectoryTurn


SYSTEM_INSTRUCTION = """你是一个 StarRocks Text2SQL agent。
你必须严格按协议输出，并同时遵守业务与安全约束。

协议要求：
1. 如果需要继续验证或探索，输出 <reasoning>...</reasoning> 和 <sql>...</sql>
2. 如果信息已经足够，输出 <reasoning>...</reasoning> 和 <solution>...</solution>
3. reasoning 必须简短，只写一句话
4. 第一轮 reasoning 先指出计划使用的核心表和核心字段
5. sql 或 solution 内只能是一条单独的查询语句
6. 不要输出协议之外的任何文字

硬约束：
1. 只能使用给定 schema 中出现的表和字段
2. 只允许只读查询，不允许 DML 或 DDL
3. 中间 <sql> 仅用于验证或探索不确定点，例如值域、连接路径、时间口径、目标集合定义
4. 优先生成低成本、小结果集、信息密度高的验证 SQL
5. 日期格式必须与字段口径一致，若字段使用 YYYYMMDD 风格，不要擅自改写成 YYYY-MM-DD
6. 单次空结果只说明当前查询未命中，不足以直接断定表无数据或条件无效，应先检查日期格式、过滤条件和字段口径
7. 没有必要时直接输出 <solution>，不要为了多轮而多轮"""


def _render_history(turns: List[TrajectoryTurn]) -> str:
    if not turns:
        return "暂无历史。"
    parts: List[str] = []
    for turn in turns:
        obs = turn["observation"]
        action_tag = "sql" if turn["action_type"] == "sql" else "solution"
        compact_observation = {
            "status": obs["status"],
            "error_message": obs["error_message"],
            "columns": obs["columns"],
            "sample_rows": obs["sample_rows"][:3],
            "row_count": obs["row_count"],
            "turns_left": obs["turns_left"],
        }
        parts.append(
            "\n".join(
                [
                    f"Turn {turn['turn_id']}",
                    f"<reasoning>{turn['reasoning']}</reasoning>",
                    f"<{action_tag}>{turn['sql']}</{action_tag}>",
                    "<observation>",
                    json.dumps(compact_observation, ensure_ascii=False, separators=(",", ":")),
                    "</observation>",
                ]
            )
        )
    return "\n\n".join(parts)


def build_teacher_user_prompt(seed: SeedRecord, turns: List[TrajectoryTurn], turns_left: int) -> str:
    parts = [
        "任务目标：用尽量少但有效的验证/探索型 SQL 找到正确的最终 SQL。",
        f"Question:\n{seed['question']}",
        "Schema Snippets:\n" + "\n\n".join(seed["schema_snippets"]),
        f"Hard Constraints:\n{seed['hard_constraints']}",
    ]
    if seed["knowledge"]:
        parts.append(f"Knowledge:\n{seed['knowledge']}")
    parts.append(f"Turns Left: {turns_left}")
    parts.append("History:\n" + _render_history(turns))
    parts.append(
        "中间 <sql> 仅用于验证或探索当前不确定点，例如值域、连接路径、时间口径或目标集合定义。"
        " 若遇到空结果，先检查日期格式、过滤条件和字段口径是否正确，不要直接下“表无数据”或“无法筛选”的结论。"
        " 若当前信息已经足够，请立即输出 <solution>。"
    )
    return "\n\n".join(parts)


def render_assistant_action(action_type: str, reasoning: str, sql: str) -> str:
    tag = "sql" if action_type == "sql" else "solution"
    return f"<reasoning>{reasoning}</reasoning>\n<{tag}>{sql}</{tag}>"
