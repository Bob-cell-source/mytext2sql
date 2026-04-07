import json
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List

import torch

from agent_rl.grpo_runner import RolloutEpisode, GRPORolloutRunner
from agent_rl.rl_env import Text2SQLRLEnv
from agent_rl.schemas import SeedRecord


SEED_ID_RE = re.compile(r"^\[seed_id=(.*?)\]\s*$", re.MULTILINE)


def extract_seed_id_from_prompt(prompt_text: str) -> str:
    match = SEED_ID_RE.search(prompt_text or "")
    if not match:
        raise ValueError("Prompt does not contain [seed_id=...] header.")
    return match.group(1).strip()


def render_initial_plain_prompt(seed_id: str, system_prompt: str, user_prompt: str) -> str:
    return (
        f"[seed_id={seed_id}]\n"
        "系统说明：\n"
        + system_prompt.strip()
        + "\n\n用户输入：\n"
        + user_prompt.strip()
    )


def render_prompt_text(
    *,
    seed_id: str,
    system_prompt: str,
    user_prompt: str,
    tokenizer: Any,
    use_chat_template: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[seed_id={seed_id}]\n{user_prompt}"},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return render_initial_plain_prompt(seed_id, system_prompt, user_prompt)


def _get_tokenizer(trainer: Any) -> Any:
    tokenizer = getattr(trainer, "processing_class", None)
    if tokenizer is None:
        tokenizer = getattr(trainer, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Cannot find tokenizer/processing_class on trainer.")
    return tokenizer


def _get_model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _compute_completion_logprobs(model: Any, input_ids: torch.Tensor, prompt_len: int) -> List[float]:
    if input_ids.shape[1] <= prompt_len:
        return []
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        start = max(prompt_len - 1, 0)
        return token_log_probs[0, start:].detach().cpu().tolist()


def generate_completion_with_model(trainer: Any, prompt_text: str) -> Dict[str, Any]:
    tokenizer = _get_tokenizer(trainer)
    model = trainer.model
    device = _get_model_device(model)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    args = trainer.args
    generate_kwargs = {
        "max_new_tokens": int(getattr(args, "max_completion_length", 512)),
        "do_sample": True,
        "temperature": float(getattr(args, "temperature", 1.0) or 1.0),
        "top_p": float(getattr(args, "top_p", 1.0) or 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    repetition_penalty = getattr(args, "repetition_penalty", None)
    if repetition_penalty:
        generate_kwargs["repetition_penalty"] = repetition_penalty

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    prompt_len = input_ids.shape[1]
    completion_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    prompt_ids = input_ids[0].detach().cpu().tolist()
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    logprobs = _compute_completion_logprobs(model, output_ids[:, :], prompt_len)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "text": completion_text,
    }


def rollout_once(
    *,
    trainer: Any,
    env: Text2SQLRLEnv,
    seed: SeedRecord,
    use_chat_template: bool = False,
) -> Dict[str, Any]:
    tokenizer = _get_tokenizer(trainer)
    runner = GRPORolloutRunner(env)
    state = env.reset(seed)
    system_prompt = env.build_system_prompt()

    all_prompt_ids: List[int] = []
    all_completion_ids: List[int] = []
    all_logprobs: List[float] = []
    generation_seconds = 0.0
    env_step_seconds = 0.0
    sql_exec_seconds = 0.0

    def _policy_fn(_: str, __: str, current_state) -> str:
        nonlocal generation_seconds
        prompt_text = render_prompt_text(
            seed_id=seed["seed_id"],
            system_prompt=system_prompt,
            user_prompt=env.build_user_prompt(current_state),
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
        started = time.perf_counter()
        generated = generate_completion_with_model(trainer, prompt_text)
        generation_seconds += time.perf_counter() - started
        all_prompt_ids.extend(generated["prompt_ids"])
        all_completion_ids.extend(generated["completion_ids"])
        all_logprobs.extend(generated["logprobs"])
        return generated["text"]

    started = time.perf_counter()
    episode: RolloutEpisode = runner.rollout_seed(seed, _policy_fn)
    rollout_seconds = time.perf_counter() - started

    for step in episode.steps:
        timing_info = step.timing_info or {}
        if timing_info:
            env_step_seconds += float(timing_info.get("step_seconds", 0.0) or 0.0)
            sql_exec_seconds += float(timing_info.get("sql_exec_seconds", 0.0) or 0.0)

    final_result_match = False
    if episode.steps:
        last_step = episode.steps[-1]
        final_result_match = bool(last_step.reward_breakdown.get("r_final_result", 0.0) > 0)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": float(episode.total_reward),
        "seed_id": seed["seed_id"],
        "difficulty": episode.difficulty,
        "turn_count": len(episode.steps),
        "final_failure_reason": episode.final_failure_reason,
        "final_result_match": final_result_match,
        "rollout_seconds": rollout_seconds,
        "generation_seconds": generation_seconds,
        "env_step_seconds": env_step_seconds,
        "sql_exec_seconds": sql_exec_seconds,
        "episode_json": json.dumps(asdict(episode), ensure_ascii=False),
    }
