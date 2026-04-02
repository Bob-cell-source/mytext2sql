import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


REASONING_PATTERN = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
SQL_PATTERN = re.compile(r"<sql>\s*(.*?)\s*</sql>", re.DOTALL | re.IGNORECASE)
SOLUTION_PATTERN = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.DOTALL | re.IGNORECASE)


@dataclass
class EvalExample:
    example_id: str
    seed_id: str
    turn_id: Optional[int]
    dataset_kind: str
    prompt_messages: List[Dict[str, str]]
    gold_output: str


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_sql_text(text: str) -> str:
    return normalize_text(text).rstrip(";")


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
    if not extract_reasoning(text):
        return False
    action_type, action_body = extract_action(text)
    return bool(action_type and action_body)


def render_single_turn_user_content(record: Dict[str, Any]) -> str:
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def build_eval_examples(dataset_path: Path) -> Tuple[List[EvalExample], str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    first = data[0]
    if "messages" in first:
        examples: List[EvalExample] = []
        for record in data:
            assistant_turn_idx = 0
            for idx, message in enumerate(record["messages"]):
                if message.get("role") != "assistant":
                    continue
                examples.append(
                    EvalExample(
                        example_id=f"{record['id']}_assistant_{assistant_turn_idx + 1}",
                        seed_id=record.get("seed_id", ""),
                        turn_id=assistant_turn_idx + 1,
                        dataset_kind="full_trajectory",
                        prompt_messages=record["messages"][:idx],
                        gold_output=message.get("content", ""),
                    )
                )
                assistant_turn_idx += 1
        return examples, "full_trajectory"

    if "instruction" in first and "output" in first:
        examples = [
            EvalExample(
                example_id=record["id"],
                seed_id=record.get("seed_id", ""),
                turn_id=record.get("turn_id"),
                dataset_kind="single_turn",
                prompt_messages=[{"role": "user", "content": render_single_turn_user_content(record)}],
                gold_output=record.get("output", ""),
            )
            for record in data
        ]
        return examples, "single_turn"

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def apply_chat_template_or_fallback(
    tokenizer,
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
        parts.append(f"{message['role'].upper()}:\n{message['content']}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def compute_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    exact_match = 0
    protocol_valid = 0
    reasoning_present = 0
    action_type_match = 0
    action_body_exact_match = 0

    sql_total = 0
    sql_match = 0
    solution_total = 0
    solution_match = 0

    for item in predictions:
        gold = item["gold_output"]
        pred = item["pred_output"]

        if normalize_text(pred) == normalize_text(gold):
            exact_match += 1
        if has_valid_protocol(pred):
            protocol_valid += 1
        if extract_reasoning(pred):
            reasoning_present += 1

        gold_type, gold_body = extract_action(gold)
        pred_type, pred_body = extract_action(pred)

        if gold_type == pred_type and gold_type is not None:
            action_type_match += 1

        body_match = normalize_sql_text(gold_body) == normalize_sql_text(pred_body) and bool(gold_body)
        if body_match:
            action_body_exact_match += 1

        if gold_type == "sql":
            sql_total += 1
            if body_match:
                sql_match += 1
        elif gold_type == "solution":
            solution_total += 1
            if body_match:
                solution_match += 1

    return {
        "total": total,
        "exact_match_rate": round(exact_match / total, 4),
        "protocol_valid_rate": round(protocol_valid / total, 4),
        "reasoning_present_rate": round(reasoning_present / total, 4),
        "action_type_accuracy": round(action_type_match / total, 4),
        "action_body_exact_match_rate": round(action_body_exact_match / total, 4),
        "sql_action_body_exact_match_rate": round(sql_match / sql_total, 4) if sql_total else None,
        "solution_action_body_exact_match_rate": round(solution_match / solution_total, 4) if solution_total else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference on Text2SQL SFT datasets with vLLM.")
    parser.add_argument("--model-path", required=True, help="Merged model path for vLLM.")
    parser.add_argument("--dataset-path", required=True, help="Path to val.json or val_full_trajectory.json.")
    parser.add_argument("--output-path", required=True, help="Path to save prediction report JSON.")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt for single-turn evaluation.")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only the first N examples if > 0.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples, dataset_kind = build_eval_examples(dataset_path)
    if args.max_samples > 0:
        examples = examples[:args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = [
        apply_chat_template_or_fallback(
            tokenizer=tokenizer,
            messages=example.prompt_messages,
            system_prompt=args.system_prompt,
        )
        for example in examples
    ]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    predictions: List[Dict[str, Any]] = []
    for example, prompt_text, output in zip(examples, prompts, outputs):
        generated_text = output.outputs[0].text.strip() if output.outputs else ""
        predictions.append(
            {
                "id": example.example_id,
                "seed_id": example.seed_id,
                "turn_id": example.turn_id,
                "dataset_kind": example.dataset_kind,
                "prompt_text": prompt_text,
                "gold_output": example.gold_output,
                "pred_output": generated_text,
            }
        )

    report = {
        "dataset_path": str(dataset_path),
        "dataset_kind": dataset_kind,
        "model_path": args.model_path,
        "max_samples": len(predictions),
        "metrics": compute_metrics(predictions),
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved vLLM inference report to: {output_path}")


if __name__ == "__main__":
    main()
