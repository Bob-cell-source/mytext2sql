import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REASONING_PATTERN = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
SQL_PATTERN = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.DOTALL | re.IGNORECASE)
SOLUTION_PATTERN = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.DOTALL | re.IGNORECASE)


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_sql_text(text: str) -> str:
    normalized = normalize_text(text)
    return normalized.rstrip(";")


def extract_reasoning(text: str) -> str:
    match = REASONING_PATTERN.search(text or "")
    return match.group(1).strip() if match else ""


def extract_action(text: str) -> Tuple[Optional[str], str]:
    sql_match = SQL_PATTERN.search(text or "")
    if sql_match:
        return "sql", sql_match.group(1).strip()

    solution_match = SOLUTION_PATTERN.search(text or "")
    if solution_match:
        return "solution", solution_match.group(1).strip()

    return None, ""


def has_valid_protocol(text: str) -> bool:
    has_reasoning = bool(REASONING_PATTERN.search(text or ""))
    action_type, action_text = extract_action(text or "")
    return has_reasoning and bool(action_type) and bool(action_text)


@dataclass
class EvalExample:
    example_id: str
    seed_id: str
    prompt_messages: List[Dict[str, str]]
    gold_output: str
    dataset_kind: str
    turn_id: Optional[int] = None


def render_single_turn_user_content(record: Dict[str, Any]) -> str:
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def build_eval_examples(dataset_path: Path) -> Tuple[List[EvalExample], str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset is empty or invalid: {dataset_path}")

    first = data[0]
    if "messages" in first:
        examples: List[EvalExample] = []
        for record in data:
            messages = record["messages"]
            assistant_turn_idx = 0
            for idx, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                prompt_messages = messages[:idx]
                examples.append(
                    EvalExample(
                        example_id=f"{record['id']}_assistant_{assistant_turn_idx + 1}",
                        seed_id=record.get("seed_id", ""),
                        prompt_messages=prompt_messages,
                        gold_output=message.get("content", ""),
                        dataset_kind="full_trajectory",
                        turn_id=assistant_turn_idx + 1,
                    )
                )
                assistant_turn_idx += 1
        return examples, "full_trajectory"

    if "instruction" in first and "output" in first:
        examples = [
            EvalExample(
                example_id=record["id"],
                seed_id=record.get("seed_id", ""),
                prompt_messages=[{"role": "user", "content": render_single_turn_user_content(record)}],
                gold_output=record.get("output", ""),
                dataset_kind="single_turn",
                turn_id=record.get("turn_id"),
            )
            for record in data
        ]
        return examples, "single_turn"

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def apply_chat_template_or_fallback(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str],
) -> str:
    full_messages: List[Dict[str, str]] = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts: List[str] = []
    for message in full_messages:
        role = message["role"].upper()
        parts.append(f"{role}:\n{message['content']}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def load_model_and_tokenizer(model_name_or_path: str, adapter_path: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
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

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def compute_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    exact_match = 0
    protocol_valid = 0
    reasoning_present = 0
    action_type_match = 0
    action_body_exact_match = 0
    sql_action_body_exact_match = 0
    solution_action_body_exact_match = 0

    for item in predictions:
        gold = item["gold_output"]
        pred = item["pred_output"]

        if normalize_text(pred) == normalize_text(gold):
            exact_match += 1

        if has_valid_protocol(pred):
            protocol_valid += 1

        if extract_reasoning(pred):
            reasoning_present += 1

        gold_action_type, gold_action_body = extract_action(gold)
        pred_action_type, pred_action_body = extract_action(pred)

        if gold_action_type == pred_action_type and gold_action_type is not None:
            action_type_match += 1

        if normalize_sql_text(pred_action_body) == normalize_sql_text(gold_action_body) and gold_action_body:
            action_body_exact_match += 1
            if gold_action_type == "sql":
                sql_action_body_exact_match += 1
            elif gold_action_type == "solution":
                solution_action_body_exact_match += 1

    return {
        "total": total,
        "exact_match_rate": round(exact_match / total, 4),
        "protocol_valid_rate": round(protocol_valid / total, 4),
        "reasoning_present_rate": round(reasoning_present / total, 4),
        "action_type_accuracy": round(action_type_match / total, 4),
        "action_body_exact_match_rate": round(action_body_exact_match / total, 4),
        "sql_action_body_exact_match_rate": round(sql_action_body_exact_match / total, 4),
        "solution_action_body_exact_match_rate": round(solution_action_body_exact_match / total, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA-Factory SFT outputs for Text2SQL agent data.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path or name.")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path. Optional.")
    parser.add_argument("--dataset-path", required=True, help="Path to val.json or val_full_trajectory.json.")
    parser.add_argument("--output-path", required=True, help="Where to save prediction report JSON.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt for single-turn evaluation.")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only the first N examples if > 0.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples, dataset_kind = build_eval_examples(dataset_path)
    if args.max_samples > 0:
        examples = examples[:args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.adapter_path)

    predictions: List[Dict[str, Any]] = []
    for example in examples:
        prompt_text = apply_chat_template_or_fallback(
            tokenizer=tokenizer,
            messages=example.prompt_messages,
            system_prompt=args.system_prompt,
        )
        pred_output = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        predictions.append(
            {
                "id": example.example_id,
                "seed_id": example.seed_id,
                "dataset_kind": example.dataset_kind,
                "turn_id": example.turn_id,
                "prompt_text": prompt_text,
                "gold_output": example.gold_output,
                "pred_output": pred_output,
            }
        )

    metrics = compute_metrics(predictions)
    report = {
        "dataset_path": str(dataset_path),
        "dataset_kind": dataset_kind,
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": args.adapter_path,
        "max_samples": len(predictions),
        "metrics": metrics,
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
