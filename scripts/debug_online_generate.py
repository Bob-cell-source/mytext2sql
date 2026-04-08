#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_rl.online_rollout import render_prompt_text
from agent_rl.rl_env import Text2SQLRLEnv


DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one online-rollout generation for a single seed.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--seed-dataset-path", required=True, help="Path to train_rl_seeds.json or val_rl_seeds.json.")
    parser.add_argument("--seed-id", required=True, help="Seed id to inspect.")
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_MAP.keys()))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--output-path", default="", help="Optional JSON output path.")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.seed_dataset_path).read_text(encoding="utf-8"))
    target = None
    for row in rows:
        if row.get("seed_id") == args.seed_id:
            target = row
            break
    if target is None:
        raise ValueError(f"未在 {args.seed_dataset_path} 中找到 seed_id={args.seed_id}")

    seed = json.loads(target["seed_json"])
    env = Text2SQLRLEnv()
    state = env.reset(seed)
    system_prompt = env.build_system_prompt()
    user_prompt = env.build_user_prompt(state)

    model, tokenizer = load_model_and_tokenizer(args)
    prompt_text = render_prompt_text(
        seed_id=seed["seed_id"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = int(attention_mask[0].sum().item()) if attention_mask is not None else input_ids.shape[1]
    completion_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    payload = {
        "seed_id": seed["seed_id"],
        "prompt_token_count": prompt_len,
        "completion_token_count": len(completion_ids),
        "prompt_text": prompt_text,
        "completion_text": completion_text,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
