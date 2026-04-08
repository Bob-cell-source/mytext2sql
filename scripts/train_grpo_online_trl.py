#!/usr/bin/env python3
import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.online_rollout import extract_seed_id_from_prompt, rollout_many_for_seed
from agent_rl.rl_env import Text2SQLRLEnv


DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _bytes_to_gib(value: int) -> float:
    return round(float(value) / (1024 ** 3), 4)


def _estimate_param_bytes(model: Any, *, trainable_only: Optional[bool] = None) -> int:
    total = 0
    for param in model.parameters():
        if trainable_only is True and not param.requires_grad:
            continue
        if trainable_only is False and param.requires_grad:
            continue
        total += param.numel() * param.element_size()
    return total


def _estimate_grad_bytes(model: Any) -> int:
    total = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.numel() * param.grad.element_size()
    return total


def _estimate_optimizer_state_bytes(optimizer: Any) -> int:
    if optimizer is None:
        return 0
    total = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
    return total


def _build_cuda_memory_summary(model: Any, optimizer: Any = None) -> Dict[str, float] | Dict[str, str]:
    if not torch.cuda.is_available():
        return {"状态": "CUDA不可用"}

    device = None
    if hasattr(model, "device"):
        device = model.device
    else:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda:0")

    if device is None or device.type != "cuda":
        return {"状态": "模型当前不在CUDA设备上"}

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    max_allocated = torch.cuda.max_memory_allocated(device_index)
    max_reserved = torch.cuda.max_memory_reserved(device_index)
    trainable_param_bytes = _estimate_param_bytes(model, trainable_only=True)
    frozen_param_bytes = _estimate_param_bytes(model, trainable_only=False)
    grad_bytes = _estimate_grad_bytes(model)
    optimizer_state_bytes = _estimate_optimizer_state_bytes(optimizer)
    approx_activation_and_temp_bytes = max(
        0,
        allocated - trainable_param_bytes - frozen_param_bytes - grad_bytes - optimizer_state_bytes,
    )
    return {
        "当前已分配显存(GiB)": _bytes_to_gib(allocated),
        "当前已保留显存(GiB)": _bytes_to_gib(reserved),
        "峰值已分配显存(GiB)": _bytes_to_gib(max_allocated),
        "峰值已保留显存(GiB)": _bytes_to_gib(max_reserved),
        "当前可用显存(GiB)": _bytes_to_gib(free_bytes),
        "显存总量(GiB)": _bytes_to_gib(total_bytes),
        "模型可训练参数显存(GiB)": _bytes_to_gib(trainable_param_bytes),
        "模型冻结参数显存(GiB)": _bytes_to_gib(frozen_param_bytes),
        "当前梯度显存(GiB)": _bytes_to_gib(grad_bytes),
        "当前优化器状态显存(GiB)": _bytes_to_gib(optimizer_state_bytes),
        "估算激活与临时显存(GiB)": _bytes_to_gib(approx_activation_and_temp_bytes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train online multi-turn Text2SQL GRPO with TRL rollout_func.")
    parser.add_argument("--base-model-path", required=True, help="Base model path or model id.")
    parser.add_argument(
        "--sft-adapter-path",
        default="",
        help="Optional SFT LoRA adapter path used as RL init when base model is not already merged.",
    )
    parser.add_argument("--train-dataset-path", required=True, help="Path to prepared online GRPO train seeds JSON.")
    parser.add_argument("--eval-dataset-path", default="", help="Optional online GRPO eval seeds JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_MAP.keys()), help="Model dtype.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true", help="Use tokenizer chat template during online rollout when available.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=0, help="0 means disable eval during training.")
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default="", help="Optional checkpoint path.")
    parser.add_argument("--report-to", default="none", help="Training report backend, e.g. none or wandb.")
    parser.add_argument(
        "--timing-log-path",
        default="",
        help="Optional timing log JSONL path. Defaults to <output_dir>/timing_metrics.jsonl.",
    )
    parser.add_argument(
        "--debug-sample-log-path",
        default="",
        help="Optional JSONL path for dumping a few online rollout samples. Defaults to <output_dir>/debug_online_rollout_samples.jsonl.",
    )
    parser.add_argument(
        "--debug-sample-limit-per-log",
        type=int,
        default=2,
        help="How many rollout samples to dump per rollout_func call.",
    )
    parser.add_argument(
        "--use-rl-lora",
        action="store_true",
        help="Wrap the loaded model with a fresh RL LoRA adapter. Recommended when base-model-path points to a merged SFT model.",
    )
    parser.add_argument("--rl-lora-r", type=int, default=64, help="Rank for fresh RL LoRA adapter.")
    parser.add_argument("--rl-lora-alpha", type=int, default=128, help="Alpha for fresh RL LoRA adapter.")
    parser.add_argument("--rl-lora-dropout", type=float, default=0.05, help="Dropout for fresh RL LoRA adapter.")
    parser.add_argument(
        "--rl-lora-target-modules",
        default="all-linear",
        help="Target modules for fresh RL LoRA. Use all-linear or comma-separated module names.",
    )
    return parser.parse_args()


def load_json_array(path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_truncate(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit and limit > 0:
        return rows[:limit]
    return rows


def parse_target_modules(value: str) -> str | List[str]:
    raw = (value or "").strip()
    if not raw:
        return "all-linear"
    if raw == "all-linear":
        return raw
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )
    if args.sft_adapter_path and args.use_rl_lora:
        raise ValueError(
            "当前脚本暂不支持在未 merge 的 SFT adapter 上再叠一层新的 RL LoRA。"
            "如果你已经 merge 了 SFT 模型，请只传 --base-model-path merged_model 并加 --use-rl-lora。"
        )
    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    elif args.use_rl_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rl_lora_r,
            lora_alpha=args.rl_lora_alpha,
            lora_dropout=args.rl_lora_dropout,
            target_modules=parse_target_modules(args.rl_lora_target_modules),
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


class OnlineTimingTracker:
    def __init__(self) -> None:
        self.reset_interval()

    def reset_interval(self) -> None:
        self.rollout_calls = 0
        self.episodes = 0
        self.rollout_wall_seconds = 0.0
        self.generation_seconds = 0.0
        self.env_step_seconds = 0.0
        self.sql_exec_seconds = 0.0
        self.train_step_wall_seconds = 0.0
        self.train_step_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.max_prompt_tokens = 0
        self.max_completion_tokens = 0
        self.turn_count_total = 0
        self.env_reward_total = 0.0
        self.final_result_match_count = 0
        self.group_reward_std_total = 0.0
        self.group_count = 0
        self.zero_std_group_count = 0

    def add_rollout_call(
        self,
        *,
        rollout_wall: float,
        generation_seconds: float,
        env_step_seconds: float,
        sql_exec_seconds: float,
        episodes: int,
        prompt_tokens: int,
        completion_tokens: int,
        max_prompt_tokens: int,
        max_completion_tokens: int,
        turn_count_total: int,
        env_reward_total: float,
        final_result_match_count: int,
        group_reward_std_total: float,
        group_count: int,
        zero_std_group_count: int,
    ) -> None:
        self.rollout_calls += 1
        self.episodes += episodes
        self.rollout_wall_seconds += rollout_wall
        self.generation_seconds += generation_seconds
        self.env_step_seconds += env_step_seconds
        self.sql_exec_seconds += sql_exec_seconds
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, max_prompt_tokens)
        self.max_completion_tokens = max(self.max_completion_tokens, max_completion_tokens)
        self.turn_count_total += turn_count_total
        self.env_reward_total += env_reward_total
        self.final_result_match_count += final_result_match_count
        self.group_reward_std_total += group_reward_std_total
        self.group_count += group_count
        self.zero_std_group_count += zero_std_group_count

    def add_train_step(self, wall_seconds: float) -> None:
        self.train_step_wall_seconds += wall_seconds
        self.train_step_count += 1

    def build_summary(self) -> Dict[str, float]:
        avg_train_step_wall = self.train_step_wall_seconds / self.train_step_count if self.train_step_count else 0.0
        avg_rollout_call_wall = self.rollout_wall_seconds / self.rollout_calls if self.rollout_calls else 0.0
        avg_episode_generation = self.generation_seconds / self.episodes if self.episodes else 0.0
        avg_episode_env = self.env_step_seconds / self.episodes if self.episodes else 0.0
        avg_episode_sql = self.sql_exec_seconds / self.episodes if self.episodes else 0.0
        avg_prompt_tokens = self.prompt_tokens / self.episodes if self.episodes else 0.0
        avg_completion_tokens = self.completion_tokens / self.episodes if self.episodes else 0.0
        avg_turn_count = self.turn_count_total / self.episodes if self.episodes else 0.0
        avg_env_reward = self.env_reward_total / self.episodes if self.episodes else 0.0
        final_result_match_rate = self.final_result_match_count / self.episodes if self.episodes else 0.0
        avg_group_reward_std = self.group_reward_std_total / self.group_count if self.group_count else 0.0
        zero_std_group_rate = self.zero_std_group_count / self.group_count if self.group_count else 0.0
        return {
            "train_steps": self.train_step_count,
            "rollout_calls": self.rollout_calls,
            "episodes": self.episodes,
            "avg_train_step_wall_seconds": round(avg_train_step_wall, 4),
            "avg_rollout_call_wall_seconds": round(avg_rollout_call_wall, 4),
            "avg_episode_generation_seconds": round(avg_episode_generation, 4),
            "avg_episode_env_seconds": round(avg_episode_env, 4),
            "avg_episode_sql_exec_seconds": round(avg_episode_sql, 4),
            "avg_prompt_tokens": round(avg_prompt_tokens, 2),
            "avg_completion_tokens": round(avg_completion_tokens, 2),
            "max_prompt_tokens": int(self.max_prompt_tokens),
            "max_completion_tokens": int(self.max_completion_tokens),
            "avg_turn_count": round(avg_turn_count, 2),
            "avg_env_reward": round(avg_env_reward, 4),
            "final_result_match_rate": round(final_result_match_rate, 4),
            "avg_group_reward_std": round(avg_group_reward_std, 4),
            "zero_std_group_rate": round(zero_std_group_rate, 4),
        }


class TimingCallback(TrainerCallback):
    def __init__(self, tracker: OnlineTimingTracker, log_path: str, metrics_log_path: str, model: Any):
        self.tracker = tracker
        self._step_started_at = None
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_log_path = Path(metrics_log_path)
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_started_at = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_started_at is not None:
            self.tracker.add_train_step(time.perf_counter() - self._step_started_at)
            self._step_started_at = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        summary = self.tracker.build_summary()
        memory_summary = _build_cuda_memory_summary(self.model, kwargs.get("optimizer"))
        chinese_summary = {
            "训练步数": summary["train_steps"],
            "rollout调用次数": summary["rollout_calls"],
            "episode数量": summary["episodes"],
            "平均每个训练步总耗时(秒)": summary["avg_train_step_wall_seconds"],
            "平均每次rollout耗时(秒)": summary["avg_rollout_call_wall_seconds"],
            "平均每个episode生成耗时(秒)": summary["avg_episode_generation_seconds"],
            "平均每个episode环境交互耗时(秒)": summary["avg_episode_env_seconds"],
            "平均每个episode SQL执行耗时(秒)": summary["avg_episode_sql_exec_seconds"],
            "平均输入token数": summary["avg_prompt_tokens"],
            "平均输出token数": summary["avg_completion_tokens"],
            "最大输入token数": summary["max_prompt_tokens"],
            "最大输出token数": summary["max_completion_tokens"],
            "平均轮数": summary["avg_turn_count"],
            "平均episode奖励": summary["avg_env_reward"],
            "最终结果命中率": summary["final_result_match_rate"],
            "组内reward标准差均值": summary["avg_group_reward_std"],
            "组内reward零方差比例": summary["zero_std_group_rate"],
        }
        if "状态" not in memory_summary:
            chinese_summary.update(memory_summary)
        else:
            chinese_summary["显存统计状态"] = memory_summary["状态"]
        print("[时间统计]", json.dumps(chinese_summary, ensure_ascii=False))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "global_step": int(getattr(state, "global_step", 0)),
                        "timing_summary": chinese_summary,
                        "raw_summary": summary,
                        "memory_summary": memory_summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        with self.metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "global_step": int(getattr(state, "global_step", 0)),
                        "trainer_logs": logs or {},
                        "rollout_summary": chinese_summary,
                        "raw_rollout_summary": summary,
                        "memory_summary": memory_summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self.tracker.reset_interval()


class OnlineEpisodeRewardFunc:
    def __init__(self) -> None:
        self.__name__ = "online_episode_reward"

    def __call__(self, prompts: List[str], completions: List[Any], **kwargs: Any) -> List[float]:
        rewards = kwargs.get("env_reward", [])
        return [float(value) for value in rewards]


def build_training_args(args: argparse.Namespace) -> GRPOConfig:
    report_to = [] if args.report_to == "none" else [args.report_to]
    eval_strategy = "no" if not args.eval_dataset_path or args.eval_steps <= 0 else "steps"
    common_kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "bf16": args.dtype == "bf16",
        "fp16": args.dtype == "fp16",
        "seed": args.seed,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": report_to,
        "remove_unused_columns": False,
    }
    if eval_strategy != "no":
        common_kwargs["eval_steps"] = args.eval_steps
    sig = inspect.signature(GRPOConfig.__init__)
    supported = set(sig.parameters.keys())

    if "eval_strategy" in supported:
        common_kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in supported:
        common_kwargs["evaluation_strategy"] = eval_strategy

    filtered_kwargs = {key: value for key, value in common_kwargs.items() if key in supported}
    dropped_keys = sorted(set(common_kwargs.keys()) - set(filtered_kwargs.keys()))
    if dropped_keys:
        print(f"[GRPOConfig兼容] 当前 TRL 版本不支持这些参数，已忽略: {', '.join(dropped_keys)}")
    return GRPOConfig(**filtered_kwargs)


def build_rollout_func(
    *,
    seed_by_id: Dict[str, Dict[str, Any]],
    use_chat_template: bool,
    tracker: OnlineTimingTracker,
    debug_sample_log_path: str,
    debug_sample_limit_per_log: int,
):
    debug_log_path = Path(debug_sample_log_path)
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _rollout_func(prompts: List[str], trainer: Any) -> Dict[str, List[Any]]:
        env = Text2SQLRLEnv()
        batch_size = len(prompts)
        prompt_ids_batch: List[List[int] | None] = [None] * batch_size
        completion_ids_batch: List[List[int] | None] = [None] * batch_size
        logprobs_batch: List[List[float] | None] = [None] * batch_size
        env_reward_batch: List[float | None] = [None] * batch_size
        turn_count_batch: List[int | None] = [None] * batch_size
        final_failure_batch: List[str | None] = [None] * batch_size
        final_result_match_batch: List[bool | None] = [None] * batch_size
        episode_json_batch: List[str | None] = [None] * batch_size
        started = time.perf_counter()
        generation_seconds = 0.0
        env_step_seconds = 0.0
        sql_exec_seconds = 0.0
        prompt_tokens_total = 0
        completion_tokens_total = 0
        max_prompt_tokens = 0
        max_completion_tokens = 0
        turn_count_total = 0
        env_reward_total = 0.0
        final_result_match_count = 0
        group_reward_std_total = 0.0
        group_count = 0
        zero_std_group_count = 0
        debug_samples: List[Dict[str, Any]] = []

        grouped_indices: Dict[str, List[int]] = {}
        for idx, prompt in enumerate(prompts):
            seed_id = extract_seed_id_from_prompt(prompt)
            grouped_indices.setdefault(seed_id, []).append(idx)

        for seed_id, indices in grouped_indices.items():
            seed = seed_by_id[seed_id]
            group_rewards: List[float] = []
            episodes = rollout_many_for_seed(
                trainer=trainer,
                env=env,
                seed=seed,
                num_generations=len(indices),
                use_chat_template=use_chat_template,
            )
            for out_idx, episode in zip(indices, episodes):
                prompt_ids_batch[out_idx] = episode["prompt_ids"]
                completion_ids_batch[out_idx] = episode["completion_ids"]
                logprobs_batch[out_idx] = episode["logprobs"]
                env_reward_batch[out_idx] = float(episode["env_reward"])
                turn_count_batch[out_idx] = int(episode["turn_count"])
                final_failure_batch[out_idx] = episode["final_failure_reason"]
                final_result_match_batch[out_idx] = bool(episode["final_result_match"])
                episode_json_batch[out_idx] = episode["episode_json"]
                generation_seconds += float(episode["generation_seconds"])
                env_step_seconds += float(episode["env_step_seconds"])
                sql_exec_seconds += float(episode["sql_exec_seconds"])
                prompt_token_count = len(episode["prompt_ids"])
                completion_token_count = len(episode["completion_ids"])
                prompt_tokens_total += prompt_token_count
                completion_tokens_total += completion_token_count
                max_prompt_tokens = max(max_prompt_tokens, prompt_token_count)
                max_completion_tokens = max(max_completion_tokens, completion_token_count)
                turn_count_total += int(episode["turn_count"])
                env_reward_total += float(episode["env_reward"])
                final_result_match_count += 1 if episode["final_result_match"] else 0
                group_rewards.append(float(episode["env_reward"]))
                if len(debug_samples) < debug_sample_limit_per_log:
                    try:
                        episode_obj = json.loads(episode["episode_json"])
                    except Exception:
                        episode_obj = {}
                    steps = episode_obj.get("steps", []) if isinstance(episode_obj, dict) else []
                    last_step = steps[-1] if steps else {}
                    debug_samples.append(
                        {
                            "seed_id": seed_id,
                            "env_reward": float(episode["env_reward"]),
                            "turn_count": int(episode["turn_count"]),
                            "final_failure_reason": episode["final_failure_reason"],
                            "final_result_match": bool(episode["final_result_match"]),
                            "prompt_token_count": prompt_token_count,
                            "completion_token_count": completion_token_count,
                            "last_prompt": last_step.get("prompt", ""),
                            "last_completion": last_step.get("completion", ""),
                            "last_action_type": last_step.get("action_type"),
                            "last_reward_breakdown": last_step.get("reward_breakdown", {}),
                        }
                    )

            if group_rewards:
                group_count += 1
                mean_reward = sum(group_rewards) / len(group_rewards)
                variance = sum((value - mean_reward) ** 2 for value in group_rewards) / len(group_rewards)
                std = variance ** 0.5
                group_reward_std_total += std
                if std <= 1e-8:
                    zero_std_group_count += 1

        tracker.add_rollout_call(
            rollout_wall=time.perf_counter() - started,
            generation_seconds=generation_seconds,
            env_step_seconds=env_step_seconds,
            sql_exec_seconds=sql_exec_seconds,
            episodes=len([x for x in env_reward_batch if x is not None]),
            prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            turn_count_total=turn_count_total,
            env_reward_total=env_reward_total,
            final_result_match_count=final_result_match_count,
            group_reward_std_total=group_reward_std_total,
            group_count=group_count,
            zero_std_group_count=zero_std_group_count,
        )
        if debug_samples:
            with debug_log_path.open("a", encoding="utf-8") as f:
                for sample in debug_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        if any(item is None for item in prompt_ids_batch + completion_ids_batch + logprobs_batch):
            raise RuntimeError("rollout_func 生成结果数量与输入 prompts 数量不一致。")
        return {
            "prompt_ids": prompt_ids_batch,
            "completion_ids": completion_ids_batch,
            "logprobs": logprobs_batch,
            "env_reward": env_reward_batch,
            "turn_count": turn_count_batch,
            "final_failure_reason": final_failure_batch,
            "final_result_match": final_result_match_batch,
            "episode_json": episode_json_batch,
        }

    return _rollout_func


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args: GRPOConfig,
    use_chat_template: bool,
    seed_by_id: Dict[str, Dict[str, Any]],
    timing_log_path: str,
    metrics_log_path: str,
    debug_sample_log_path: str,
    debug_sample_limit_per_log: int,
):
    tracker = OnlineTimingTracker()
    reward_func = OnlineEpisodeRewardFunc()
    rollout_func = build_rollout_func(
        seed_by_id=seed_by_id,
        use_chat_template=use_chat_template,
        tracker=tracker,
        debug_sample_log_path=debug_sample_log_path,
        debug_sample_limit_per_log=debug_sample_limit_per_log,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "reward_funcs": [reward_func],
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "rollout_func": rollout_func,
    }
    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "rollout_func" in str(exc):
            raise RuntimeError(
                "当前安装的 TRL 版本不支持 rollout_func。请升级到支持 rollout_func/OpenEnv 的 TRL 版本。"
            ) from exc
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = GRPOTrainer(**trainer_kwargs)
    trainer.add_callback(TimingCallback(tracker, timing_log_path, metrics_log_path, model))
    return trainer


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = maybe_truncate(load_json_array(args.train_dataset_path), args.max_train_samples)
    eval_rows = maybe_truncate(load_json_array(args.eval_dataset_path), args.max_eval_samples) if args.eval_dataset_path else []

    seed_by_id: Dict[str, Dict[str, Any]] = {}
    for row in train_rows + eval_rows:
        seed_by_id[row["seed_id"]] = json.loads(row["seed_json"])

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    model, tokenizer = load_model_and_tokenizer(args)
    training_args = build_training_args(args)
    timing_log_path = args.timing_log_path or str(output_dir / "timing_metrics.jsonl")
    metrics_log_path = str(output_dir / "training_metrics.jsonl")
    debug_sample_log_path = args.debug_sample_log_path or str(output_dir / "debug_online_rollout_samples.jsonl")
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        use_chat_template=args.use_chat_template,
        seed_by_id=seed_by_id,
        timing_log_path=timing_log_path,
        metrics_log_path=metrics_log_path,
        debug_sample_log_path=debug_sample_log_path,
        debug_sample_limit_per_log=args.debug_sample_limit_per_log,
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Saved online GRPO model to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
