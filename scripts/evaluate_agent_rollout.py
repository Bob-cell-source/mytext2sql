#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.grpo_runner import GRPORolloutRunner
from agent_rl.rl_env import RLEpisodeState, Text2SQLRLEnv
from agent_rl.seed_builder import build_seed_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end rollout evaluation on golden Text2SQL seeds.")
    parser.add_argument("--model-name-or-path", required=True, help="Merged model path or base model path.")
    parser.add_argument("--adapter-path", default="", help="Optional adapter path.")
    parser.add_argument("--golden-path", default="golden_sql_marked.json")
    parser.add_argument("--schema-path", default="schema.json")
    parser.add_argument("--output-path", required=True, help="Where to save rollout evaluation report JSON.")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only first N seeds.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(model_name_or_path: str, adapter_path: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def apply_chat_template_or_fallback(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    model_name_or_path: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_path_lower = (model_name_or_path or "").lower()
    if "qwen" in model_path_lower:
        parts: List[str] = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    return f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_policy_fn(model: Any, tokenizer: Any, args: argparse.Namespace):
    def _policy(system_prompt: str, user_prompt: str, state: RLEpisodeState) -> str:
        prompt_text = apply_chat_template_or_fallback(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name_or_path=args.model_name_or_path,
        )
        return generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    return _policy


def build_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {"total": 0}

    matched = 0
    done_count = 0
    total_turns = 0
    failure_breakdown: Dict[str, int] = {}

    for episode in episodes:
        steps = episode.get("steps", [])
        done_count += 1 if episode.get("done") else 0
        total_turns += len(steps)
        if steps and float(steps[-1].get("reward_breakdown", {}).get("r_final_result", 0.0) or 0.0) > 0:
            matched += 1
        reason = episode.get("final_failure_reason", "")
        if reason:
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1

    return {
        "total": total,
        "done_rate": round(done_count / total, 4),
        "result_match_rate": round(matched / total, 4),
        "avg_turns": round(total_turns / total, 4),
        "failure_breakdown": failure_breakdown,
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = build_seed_records(args.golden_path, args.schema_path)
    if args.max_samples > 0:
        seeds = seeds[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.adapter_path, args.trust_remote_code)
    env = Text2SQLRLEnv()
    runner = GRPORolloutRunner(env)
    policy_fn = build_policy_fn(model, tokenizer, args)

    episodes = runner.rollout_many(seeds, policy_fn)
    episodes_payload = runner.to_serializable(episodes)

    report = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "golden_path": args.golden_path,
        "schema_path": args.schema_path,
        "metrics": build_metrics(episodes_payload),
        "episodes": episodes_payload,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved rollout evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
