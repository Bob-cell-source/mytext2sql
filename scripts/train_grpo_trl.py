#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.rl_env import Text2SQLRLEnv


DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-step Text2SQL GRPO with TRL.")
    parser.add_argument("--base-model-path", required=True, help="Base model path or model id.")
    parser.add_argument("--sft-adapter-path", default="", help="Optional SFT LoRA adapter path used as RL init.")
    parser.add_argument("--train-dataset-path", required=True, help="Path to prepared GRPO train JSON.")
    parser.add_argument("--eval-dataset-path", default="", help="Optional GRPO eval JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Model dtype.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true", help="Render prompts with tokenizer chat template.")
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
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default="", help="Optional checkpoint path.")
    parser.add_argument("--report-to", default="none", help="Training report backend, e.g. none or wandb.")
    return parser.parse_args()


def load_json_array(path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_truncate(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit and limit > 0:
        return rows[:limit]
    return rows


def normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion and isinstance(completion["content"], str):
            return completion["content"]
        if "text" in completion and isinstance(completion["text"], str):
            return completion["text"]
    if isinstance(completion, list):
        text_parts: List[str] = []
        for item in completion:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
        return "".join(text_parts)
    return str(completion or "")


def maybe_apply_chat_template(dataset: Dataset, tokenizer: Any, use_chat_template: bool) -> Dataset:
    if not use_chat_template:
        return dataset
    if not hasattr(tokenizer, "apply_chat_template"):
        print("Tokenizer does not expose apply_chat_template, fallback to plain prompt.")
        return dataset

    def _map_record(record: Dict[str, Any]) -> Dict[str, Any]:
        messages = []
        if record.get("system_prompt"):
            messages.append({"role": "system", "content": record["system_prompt"]})
        messages.append({"role": "user", "content": record["user_prompt"]})
        record["prompt"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return record

    return dataset.map(_map_record, desc="Applying chat template")


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )
    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


class SingleStepRewardFunc:
    def __init__(self) -> None:
        self.env = Text2SQLRLEnv()

    def __call__(self, prompts: List[str], completions: List[Any], **kwargs: Any) -> List[float]:
        rewards: List[float] = []
        seed_payloads = kwargs.get("seed_json", [])
        history_payloads = kwargs.get("history_json", [])
        difficulties = kwargs.get("difficulty", [])

        for idx, completion in enumerate(completions):
            completion_text = normalize_completion_text(completion)
            try:
                seed = json.loads(seed_payloads[idx])
                history = json.loads(history_payloads[idx])
                difficulty = difficulties[idx] if idx < len(difficulties) else seed.get("difficulty", "")
                state = self.env.restore_state(seed, history, difficulty=difficulty)
                step_result = self.env.step(state, completion_text)
                rewards.append(float(step_result["reward"]))
            except Exception as exc:
                print(f"[reward_func] failed on sample {idx}: {exc}")
                rewards.append(-2.0)

        return rewards


def build_training_args(args: argparse.Namespace) -> GRPOConfig:
    report_to = [] if args.report_to == "none" else [args.report_to]
    eval_strategy = "no" if not args.eval_dataset_path or args.eval_steps <= 0 else "steps"
    common_kwargs = {
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
    try:
        return GRPOConfig(eval_strategy=eval_strategy, **common_kwargs)
    except TypeError:
        return GRPOConfig(evaluation_strategy=eval_strategy, **common_kwargs)


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args: GRPOConfig,
):
    reward_func = SingleStepRewardFunc()
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "reward_funcs": [reward_func],
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
    }
    try:
        return GRPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        return GRPOTrainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = maybe_truncate(load_json_array(args.train_dataset_path), args.max_train_samples)
    eval_rows = maybe_truncate(load_json_array(args.eval_dataset_path), args.max_eval_samples) if args.eval_dataset_path else []

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = maybe_apply_chat_template(train_dataset, tokenizer, args.use_chat_template)
    if eval_dataset is not None:
        eval_dataset = maybe_apply_chat_template(eval_dataset, tokenizer, args.use_chat_template)

    training_args = build_training_args(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )

    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Saved GRPO model to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
