#!/usr/bin/env python3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize multi-turn SFT trajectories for Text2SQL.")
    parser.add_argument("--golden", default="golden_sql_marked.json")
    parser.add_argument("--schema", default="schema.json")
    parser.add_argument("--outdir", default="output/sft_synthesis")
    parser.add_argument("--seed-limit", type=int, default=0, help="Only process the first N seeds when > 0.")
    parser.add_argument("--attempts", type=int, default=6, help="Teacher rollouts per seed.")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--max-preview-rows", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="Number of seeds to process concurrently.")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _load_processed_seed_ids(checkpoint_dir: Path) -> set[str]:
    if not checkpoint_dir.exists():
        return set()
    processed = set()
    for path in checkpoint_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed_id = payload.get("seed_id")
            if seed_id:
                processed.add(seed_id)
        except Exception:
            continue
    return processed


def _process_seed_job(
    seed: dict,
    attempts: int,
    max_turns: int,
    keep_top_k: int,
    max_preview_rows: int,
) -> Tuple[dict, List[dict]]:
    from agent_rl.sql_env import SQLEnvironment
    from agent_rl.teacher_rollout import TeacherRolloutRunner
    from agent_rl.trajectory_filter import filter_and_rank_trajectories

    env = SQLEnvironment(max_preview_rows=max_preview_rows)
    runner = TeacherRolloutRunner(env=env, max_turns=max_turns)

    candidates = runner.rollout_many(seed, attempts=attempts)
    kept, rejected, dropped_after_ranking = filter_and_rank_trajectories(
        candidates,
        max_turns=max_turns,
        keep_top_k=keep_top_k,
    )
    summary = {
        "seed_id": seed["seed_id"],
        "candidate_count": len(candidates),
        "kept_count": len(kept),
        "hard_rejected_count": len(rejected),
        "dropped_after_ranking_count": len(dropped_after_ranking),
        "kept_scores": [item["meta"]["score"] for item in kept],
        "hard_reject_reasons": [item["meta"]["reject_reason"] for item in rejected],
        "dropped_after_ranking_scores": [item["meta"]["score"] for item in dropped_after_ranking],
    }
    checkpoint_payload = {
        "seed_id": seed["seed_id"],
        "summary": summary,
        "kept_trajectories": kept,
    }
    return checkpoint_payload, kept


def _rebuild_outputs_from_checkpoints(outdir: Path, seeds_by_id: Dict[str, dict], report_header: dict) -> None:
    from agent_rl.dataset_packer import pack_action_focused_samples, pack_full_trajectories, write_synthesis_report

    checkpoint_dir = outdir / "per_seed_results"
    checkpoint_payloads = []
    for path in checkpoint_dir.glob("*.json"):
        try:
            checkpoint_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    order_index = {seed_id: idx for idx, seed_id in enumerate(seeds_by_id.keys())}
    checkpoint_payloads.sort(key=lambda item: order_index.get(item.get("seed_id", ""), 10**9))

    kept_all: List[dict] = []
    per_seed = []
    total_candidates = 0
    total_kept = 0
    total_hard_rejected = 0
    total_dropped_after_ranking = 0

    for payload in checkpoint_payloads:
        summary = payload.get("summary", {})
        kept = payload.get("kept_trajectories", [])
        kept_all.extend(kept)
        per_seed.append(summary)
        total_candidates += summary.get("candidate_count", 0)
        total_kept += summary.get("kept_count", 0)
        total_hard_rejected += summary.get("hard_rejected_count", 0)
        total_dropped_after_ranking += summary.get("dropped_after_ranking_count", 0)

    report = dict(report_header)
    report.update(
        {
            "completed_seed_count": len(checkpoint_payloads),
            "per_seed": per_seed,
            "total_candidates": total_candidates,
            "total_kept": total_kept,
            "total_hard_rejected": total_hard_rejected,
            "total_dropped_after_ranking": total_dropped_after_ranking,
        }
    )

    pack_full_trajectories(kept_all, str(outdir / "sft_multiturn_full.jsonl"))
    pack_action_focused_samples(seeds_by_id, kept_all, str(outdir / "sft_multiturn_action_focused.jsonl"))
    write_synthesis_report(report, str(outdir / "synthesis_report.json"))


def main() -> None:
    from agent_rl.seed_builder import build_seed_records, dump_seed_records
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / "per_seed_results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = build_seed_records(args.golden, args.schema)
    if args.seed_limit > 0:
        seeds = seeds[: args.seed_limit]
    dump_seed_records(seeds, str(outdir / "seed_records.json"))
    seeds_by_id: Dict[str, dict] = {seed["seed_id"]: seed for seed in seeds}
    report_header = {
        "seed_count": len(seeds),
        "attempts_per_seed": args.attempts,
        "max_turns": args.max_turns,
        "workers": args.workers,
    }

    processed_seed_ids = _load_processed_seed_ids(checkpoint_dir)
    pending_seeds = [seed for seed in seeds if seed["seed_id"] not in processed_seed_ids]

    if args.workers <= 1:
        for idx, seed in enumerate(pending_seeds, start=1):
            checkpoint_payload, _ = _process_seed_job(
                seed=seed,
                attempts=args.attempts,
                max_turns=args.max_turns,
                keep_top_k=args.keep_top_k,
                max_preview_rows=args.max_preview_rows,
            )
            _atomic_write_json(checkpoint_dir / f"{seed['seed_id']}.json", checkpoint_payload)
            if idx % 1 == 0:
                _rebuild_outputs_from_checkpoints(outdir, seeds_by_id, report_header)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _process_seed_job,
                    seed,
                    args.attempts,
                    args.max_turns,
                    args.keep_top_k,
                    args.max_preview_rows,
                ): seed["seed_id"]
                for seed in pending_seeds
            }
            for future in as_completed(future_map):
                seed_id = future_map[future]
                checkpoint_payload, _ = future.result()
                _atomic_write_json(checkpoint_dir / f"{seed_id}.json", checkpoint_payload)
                _rebuild_outputs_from_checkpoints(outdir, seeds_by_id, report_header)

    _rebuild_outputs_from_checkpoints(outdir, seeds_by_id, report_header)


if __name__ == "__main__":
    main()
