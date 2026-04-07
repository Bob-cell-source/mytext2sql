#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.result_matcher import build_result_fingerprint
from agent_rl.rl_env import Text2SQLRLEnv
from agent_rl.seed_builder import build_schema_index, build_schema_snippets, build_seed_records, load_json


class NoopSQLEnv:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare online multi-turn GRPO seed pools.")
    parser.add_argument("--golden-path", default="golden_sql_marked.json", help="Path to golden_sql_marked.json.")
    parser.add_argument("--schema-path", default="schema.json", help="Path to schema.json.")
    parser.add_argument(
        "--synthetic-path",
        default="",
        help="Optional synthetic/evalset JSON or JSONL to augment RL train pool. Validation split remains gold-only.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for gold seed split.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for gold train/val split.")
    parser.add_argument(
        "--max-gold-train-seeds",
        type=int,
        default=0,
        help="Optional limit for gold train seeds after split.",
    )
    parser.add_argument(
        "--max-gold-val-seeds",
        type=int,
        default=0,
        help="Optional limit for gold val seeds after split.",
    )
    parser.add_argument(
        "--max-synthetic-train-seeds",
        type=int,
        default=0,
        help="Optional limit for synthetic train seeds.",
    )
    parser.add_argument(
        "--synthetic-work-dir",
        default="./tmp_online_seed_sql",
        help="Temporary directory when executing synthetic gold SQL to obtain gold_result.",
    )
    parser.add_argument(
        "--outdir",
        default="output/rl_seed_pool_v2",
        help="Output directory for online GRPO seed pools.",
    )
    return parser.parse_args()


def render_initial_plain_prompt(seed_id: str, system_prompt: str, user_prompt: str) -> str:
    return (
        f"[seed_id={seed_id}]\n"
        "系统说明：\n"
        + system_prompt.strip()
        + "\n\n用户输入：\n"
        + user_prompt.strip()
    )


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_limit(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit and limit > 0:
        return rows[:limit]
    return rows


def split_gold_seeds(
    gold_seeds: List[Dict[str, Any]],
    *,
    val_ratio: float,
    split_seed: int,
    max_train: int,
    max_val: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(gold_seeds)
    rng = random.Random(split_seed)
    rng.shuffle(rows)
    if not rows or val_ratio <= 0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(rows) * val_ratio)))
    val_rows = maybe_limit(rows[:val_count], max_val)
    train_rows = maybe_limit(rows[val_count:], max_train)
    return train_rows, val_rows


def build_seed_pool(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    env = Text2SQLRLEnv(sql_env=NoopSQLEnv())
    records: List[Dict[str, Any]] = []
    for seed in seed_rows:
        state = env.reset(seed)
        system_prompt = env.build_system_prompt()
        user_prompt = env.build_user_prompt(state)
        records.append(
            {
                "id": seed["seed_id"],
                "seed_id": seed["seed_id"],
                "prompt": render_initial_plain_prompt(seed["seed_id"], system_prompt, user_prompt),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "difficulty": state.difficulty,
                "max_turns": state.max_turns,
                "seed_json": json.dumps(seed, ensure_ascii=False),
                "gold_sql": seed.get("gold_sql", ""),
                "gold_result_json": json.dumps(seed.get("gold_result", []), ensure_ascii=False),
                "source": seed.get("source", "gold"),
            }
        )
    return records


def _extract_table_list(row: Dict[str, Any]) -> List[str]:
    metadata = row.get("metadata", {}) or {}
    table_list = metadata.get("table_list")
    if isinstance(table_list, list):
        return table_list
    if isinstance(row.get("table_list"), list):
        return row["table_list"]
    return []


def build_synthetic_seed_records(
    *,
    synthetic_rows: List[Dict[str, Any]],
    schema_path: str,
    work_dir: str,
) -> List[Dict[str, Any]]:
    schema_json = load_json(schema_path)
    schema_index = build_schema_index(schema_json)
    sql_env = Text2SQLRLEnv(sql_env=None).sql_env
    sql_env.work_dir = Path(work_dir)
    sql_env.work_dir.mkdir(parents=True, exist_ok=True)

    synthetic_seeds: List[Dict[str, Any]] = []
    for idx, row in enumerate(synthetic_rows, start=1):
        question = (row.get("question") or "").strip()
        gold_sql = (row.get("gold_sql") or row.get("SQL") or "").strip()
        if not question or not gold_sql:
            continue
        table_list = _extract_table_list(row)
        knowledge = (row.get("knowledge") or row.get("evidence") or "").strip()
        difficulty = (row.get("difficulty") or "").strip()
        sample_id = (row.get("sample_id") or row.get("sql_id") or f"synthetic_{idx:04d}").strip()

        if row.get("gold_result") is not None:
            gold_result = row.get("gold_result") or []
        else:
            observation, gold_result = Text2SQLRLEnv(sql_env=sql_env).sql_env.execute_with_rows(
                gold_sql,
                turns_left=0,
                sql_id=f"{sample_id}_gold",
            )
            if observation.get("status") not in {"success", "empty"}:
                continue

        synthetic_seeds.append(
            {
                "seed_id": f"syn_{sample_id}",
                "question": question,
                "table_list": table_list,
                "knowledge": knowledge,
                "difficulty": difficulty,
                "gold_sql": gold_sql,
                "gold_result": gold_result,
                "gold_result_fingerprint": build_result_fingerprint(gold_result),
                "schema_snippets": build_schema_snippets(table_list, schema_index),
                "hard_constraints": "\n".join(
                    [
                        "1. 只允许单条 MySQL/StarRocks 查询语句，禁止非查询语句。",
                        "2. 禁止 Hive/Spark 方言，如 LATERAL VIEW、explode、to_date 等。",
                        "3. 只能使用给定 schema 中的表和字段，禁止编造字段。",
                        "4. _di 表必须带时间范围，_df 表必须使用单日快照。",
                        "5. 严格遵守 ID 体系与映射规则，禁止将全局 ID 和游戏 ID 直接错连。",
                        "6. probe SQL 以确认值域、口径、连接路径为目标，尽量简洁高效。",
                    ]
                ),
                "source": row.get("source", "synthetic"),
            }
        )
    return synthetic_seeds


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    gold_seeds = build_seed_records(args.golden_path, args.schema_path)
    gold_train, gold_val = split_gold_seeds(
        gold_seeds,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        max_train=args.max_gold_train_seeds,
        max_val=args.max_gold_val_seeds,
    )

    synthetic_seed_count = 0
    train_seed_rows = list(gold_train)

    if args.synthetic_path:
        synthetic_rows = load_json_or_jsonl(Path(args.synthetic_path))
        synthetic_rows = maybe_limit(synthetic_rows, args.max_synthetic_train_seeds)
        synthetic_seeds = build_synthetic_seed_records(
            synthetic_rows=synthetic_rows,
            schema_path=args.schema_path,
            work_dir=args.synthetic_work_dir,
        )
        synthetic_seed_count = len(synthetic_seeds)
        train_seed_rows.extend(synthetic_seeds)

    outdir = Path(args.outdir)
    train_records = build_seed_pool(train_seed_rows)
    val_records = build_seed_pool(gold_val)

    write_json(outdir / "train_rl_seeds.json", train_records)
    write_json(outdir / "val_rl_seeds.json", val_records)
    write_json(
        outdir / "prepare_report.json",
        {
            "gold_seed_count": len(gold_seeds),
            "gold_train_seed_count": len(gold_train),
            "gold_val_seed_count": len(gold_val),
            "synthetic_train_seed_count": synthetic_seed_count,
            "train_seed_count": len(train_records),
            "val_seed_count": len(val_records),
            "val_is_gold_only": True,
            "sources_in_train": sorted({item["source"] for item in train_records}),
            "sources_in_val": sorted({item["source"] for item in val_records}),
        },
    )

    print(f"Saved online RL train seed pool to: {outdir / 'train_rl_seeds.json'}")
    print(f"Saved online RL val seed pool to: {outdir / 'val_rl_seeds.json'}")


if __name__ == "__main__":
    main()
