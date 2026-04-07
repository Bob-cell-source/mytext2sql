#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_reward_series(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float], str]:
    candidate_keys = [
        "reward",
        "mean_reward",
        "objective",
        "rl_reward",
        "env_reward",
        "rewards/mean",
    ]
    for key in candidate_keys:
        xs: List[int] = []
        ys: List[float] = []
        for row in rows:
            logs = row.get("trainer_logs", {}) or {}
            if key in logs and isinstance(logs[key], (int, float)):
                xs.append(int(row.get("global_step", 0)))
                ys.append(float(logs[key]))
        if ys:
            return xs, ys, key

    xs = []
    ys = []
    for row in rows:
        summary = row.get("raw_rollout_summary", {}) or {}
        if "avg_env_reward" in summary:
            xs.append(int(row.get("global_step", 0)))
            ys.append(float(summary["avg_env_reward"]))
    return xs, ys, "avg_env_reward"


def save_markdown_summary(path: Path, *, xs: List[int], ys: List[float], label: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"指标: {label}",
                f"点数: {len(xs)}",
                f"起点: step={xs[0] if xs else 0}, value={ys[0] if ys else 0}",
                f"终点: step={xs[-1] if xs else 0}, value={ys[-1] if ys else 0}",
                f"最大值: {max(ys) if ys else 0}",
                f"最小值: {min(ys) if ys else 0}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRPO reward curve from training_metrics.jsonl.")
    parser.add_argument("--input", required=True, help="Path to training_metrics.jsonl.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    xs, ys, label = pick_reward_series(rows)
    if not ys:
        raise ValueError("No reward-like series found in metrics log.")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib 未安装，无法绘制曲线图。请先安装 matplotlib。") from exc

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.title(f"GRPO Reward Curve ({label})")
    plt.xlabel("Global Step")
    plt.ylabel(label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    summary_path = output_path.with_suffix(".txt")
    save_markdown_summary(summary_path, xs=xs, ys=ys, label=label)
    print(f"Saved reward curve to: {output_path}")
    print(f"Saved reward summary to: {summary_path}")


if __name__ == "__main__":
    main()
