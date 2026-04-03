import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def sanitize_tokenizer_config(model_dir: Path) -> None:
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return

    data = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    extra_special_tokens = data.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        data["_original_extra_special_tokens_list"] = extra_special_tokens
        data["extra_special_tokens"] = {}
        tokenizer_config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Sanitized tokenizer_config.json at: {tokenizer_config_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model-path", required=True, help="Base model path or HF model name.")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter directory.")
    parser.add_argument("--output-path", required=True, help="Merged model output directory.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Model load dtype. Use bf16 on modern GPUs by default.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code when loading model/tokenizer.")
    parser.add_argument("--safe-serialization", action="store_true", help="Save merged model with safetensors.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_dtype = DTYPE_MAP[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(
        output_path,
        safe_serialization=args.safe_serialization,
    )
    tokenizer.save_pretrained(output_path)
    sanitize_tokenizer_config(output_path)

    print(f"Merged model saved to: {output_path}")


if __name__ == "__main__":
    main()
