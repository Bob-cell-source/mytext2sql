import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def normalize_sql(text: str) -> str:
    return " ".join((text or "").strip().rstrip(";").split())


def try_load_tokenizer(model_path: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception as exc:
        print(
            "Warning: failed to load tokenizer normally. Falling back to manual prompt rendering.\n"
            f"Tokenizer error: {type(exc).__name__}: {exc}"
        )
        return None


def apply_chat_template_or_fallback(tokenizer, instruction: str, input_text: str, model_path: str) -> str:
    user_content = f"{instruction}\n\n{input_text}".strip()
    messages = [{"role": "user", "content": user_content}]

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if "qwen" in model_path.lower():
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    return f"USER:\n{user_content}\n\nASSISTANT:\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for final-SQL baseline with vLLM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
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

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    tokenizer = try_load_tokenizer(args.model_path, args.trust_remote_code)
    prompts = [
        apply_chat_template_or_fallback(
            tokenizer=tokenizer,
            instruction=row.get("instruction", ""),
            input_text=row.get("input", ""),
            model_path=args.model_path,
        )
        for row in rows
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
    exact_match = 0
    non_empty = 0

    for row, prompt_text, output in zip(rows, prompts, outputs):
        pred_output = output.outputs[0].text.strip() if output.outputs else ""
        if pred_output:
            non_empty += 1
        if normalize_sql(pred_output) == normalize_sql(row.get("output", "")):
            exact_match += 1
        predictions.append(
            {
                "id": row.get("id"),
                "seed_id": row.get("seed_id"),
                "prompt_text": prompt_text,
                "gold_output": row.get("output", ""),
                "pred_output": pred_output,
                "task_type": "final_sql_baseline",
            }
        )

    total = len(predictions)
    report = {
        "dataset_path": str(dataset_path),
        "dataset_kind": "final_sql_baseline",
        "model_path": args.model_path,
        "max_samples": total,
        "metrics": {
            "total": total,
            "non_empty_rate": round(non_empty / total, 4) if total else 0.0,
            "sql_exact_match_rate": round(exact_match / total, 4) if total else 0.0,
        },
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved baseline vLLM inference report to: {output_path}")


if __name__ == "__main__":
    main()
