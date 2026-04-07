import json
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List

import torch

from agent_rl.grpo_runner import RolloutEpisode, RolloutStep
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


def _compute_batched_completion_logprobs(model: Any, sequences: torch.Tensor, prompt_seq_len: int) -> List[List[float]]:
    if sequences.shape[1] <= prompt_seq_len:
        return [[] for _ in range(sequences.shape[0])]
    with torch.no_grad():
        outputs = model(input_ids=sequences)
        logits = outputs.logits[:, :-1, :]
        target_ids = sequences[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        start = max(prompt_seq_len - 1, 0)
        rows: List[List[float]] = []
        for row_idx in range(sequences.shape[0]):
            rows.append(token_log_probs[row_idx, start:].detach().cpu().tolist())
        return rows


def generate_completions_with_model(trainer: Any, prompt_texts: List[str]) -> List[Dict[str, Any]]:
    tokenizer = _get_tokenizer(trainer)
    model = trainer.model
    device = _get_model_device(model)
    args = trainer.args
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "add_special_tokens": False,
    }
    max_prompt_length = getattr(args, "max_prompt_length", None)
    if max_prompt_length:
        tokenizer_kwargs["max_length"] = int(max_prompt_length)
    encoded = tokenizer(prompt_texts, **tokenizer_kwargs)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

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

    prompt_seq_len = input_ids.shape[1]
    all_logprobs = _compute_batched_completion_logprobs(model, output_ids[:, :], prompt_seq_len)
    results: List[Dict[str, Any]] = []
    for row_idx in range(output_ids.shape[0]):
        prompt_token_count = int(attention_mask[row_idx].sum().item()) if attention_mask is not None else prompt_seq_len
        prompt_ids = input_ids[row_idx, :prompt_token_count].detach().cpu().tolist()
        completion_ids = output_ids[row_idx, prompt_seq_len:].detach().cpu().tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append(
            {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": all_logprobs[row_idx][: len(completion_ids)],
                "text": completion_text,
            }
        )
    return results


def rollout_many_for_seed(
    *,
    trainer: Any,
    env: Text2SQLRLEnv,
    seed: SeedRecord,
    num_generations: int,
    use_chat_template: bool = False,
) -> List[Dict[str, Any]]:
    tokenizer = _get_tokenizer(trainer)
    states = [env.reset(seed) for _ in range(num_generations)]
    system_prompt = env.build_system_prompt()
    trajectories: List[Dict[str, Any]] = []
    for _ in range(num_generations):
        trajectories.append(
            {
                "prompt_ids": [],
                "completion_ids": [],
                "logprobs": [],
                "steps": [],
                "generation_seconds": 0.0,
                "env_step_seconds": 0.0,
                "sql_exec_seconds": 0.0,
            }
        )

    rollout_started_at = time.perf_counter()
    while True:
        active_indices = [idx for idx, state in enumerate(states) if not state.done]
        if not active_indices:
            break

        prompt_texts: List[str] = []
        user_prompts: List[str] = []
        for idx in active_indices:
            user_prompt = env.build_user_prompt(states[idx])
            user_prompts.append(user_prompt)
            prompt_texts.append(
                render_prompt_text(
                    seed_id=seed["seed_id"],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    tokenizer=tokenizer,
                    use_chat_template=use_chat_template,
                )
            )

        started = time.perf_counter()
        generated_batch = generate_completions_with_model(trainer, prompt_texts)
        batch_generation_seconds = time.perf_counter() - started

        for active_pos, state_idx in enumerate(active_indices):
            generated = generated_batch[active_pos]
            traj = trajectories[state_idx]
            traj["generation_seconds"] += batch_generation_seconds / max(1, len(active_indices))
            traj["prompt_ids"].extend(generated["prompt_ids"])
            traj["completion_ids"].extend(generated["completion_ids"])
            traj["logprobs"].extend(generated["logprobs"])

        step_results = env.batch_step(
            [states[state_idx] for state_idx in active_indices],
            [generated["text"] for generated in generated_batch],
        )
        for active_pos, state_idx in enumerate(active_indices):
            step_result = step_results[active_pos]
            generated = generated_batch[active_pos]
            traj = trajectories[state_idx]
            timing_info = step_result.get("timing_info", {}) or {}
            traj["env_step_seconds"] += float(timing_info.get("step_seconds", 0.0) or 0.0)
            traj["sql_exec_seconds"] += float(timing_info.get("sql_exec_seconds", 0.0) or 0.0)
            action = step_result["action"]
            observation = step_result["observation"]
            traj["steps"].append(
                RolloutStep(
                    turn_id=states[state_idx].history[-1].turn_id if states[state_idx].history else len(traj["steps"]) + 1,
                    prompt=user_prompts[active_pos],
                    completion=generated["text"],
                    reward=step_result["reward"],
                    reward_breakdown=step_result["reward_breakdown"],
                    done=step_result["done"],
                    action_type=action.action_type if action else None,
                    sql=action.sql if action else "",
                    reasoning=action.reasoning if action else "",
                    observation=observation,
                    timing_info=timing_info,
                )
            )

    rollout_seconds = time.perf_counter() - rollout_started_at
    results: List[Dict[str, Any]] = []
    for idx, state in enumerate(states):
        steps = trajectories[idx]["steps"]
        episode = RolloutEpisode(
            seed_id=seed["seed_id"],
            difficulty=state.difficulty,
            total_reward=sum(step.reward for step in steps),
            done=state.done,
            final_failure_reason=state.final_failure_reason,
            steps=steps,
        )
        final_result_match = False
        if steps:
            final_result_match = bool(steps[-1].reward_breakdown.get("r_final_result", 0.0) > 0)
        results.append(
            {
                "prompt_ids": trajectories[idx]["prompt_ids"],
                "completion_ids": trajectories[idx]["completion_ids"],
                "logprobs": trajectories[idx]["logprobs"],
                "env_reward": float(episode.total_reward),
                "seed_id": seed["seed_id"],
                "difficulty": episode.difficulty,
                "turn_count": len(episode.steps),
                "final_failure_reason": episode.final_failure_reason,
                "final_result_match": final_result_match,
                "rollout_seconds": rollout_seconds / max(1, num_generations),
                "generation_seconds": trajectories[idx]["generation_seconds"],
                "env_step_seconds": trajectories[idx]["env_step_seconds"],
                "sql_exec_seconds": trajectories[idx]["sql_exec_seconds"],
                "episode_json": json.dumps(asdict(episode), ensure_ascii=False),
            }
        )
    return results
