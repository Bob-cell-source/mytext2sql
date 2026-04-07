from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_rl.rl_env import RLEpisodeState, Text2SQLRLEnv
from agent_rl.schemas import SeedRecord


PolicyFn = Callable[[str, str, RLEpisodeState], str]


@dataclass
class RolloutStep:
    turn_id: int
    prompt: str
    completion: str
    reward: float
    reward_breakdown: Dict[str, float]
    done: bool
    action_type: Optional[str] = None
    sql: str = ""
    reasoning: str = ""
    observation: Optional[Dict[str, Any]] = None


@dataclass
class RolloutEpisode:
    seed_id: str
    difficulty: str
    total_reward: float
    done: bool
    final_failure_reason: str
    steps: List[RolloutStep] = field(default_factory=list)


class GRPORolloutRunner:
    def __init__(self, env: Text2SQLRLEnv):
        self.env = env

    def rollout_seed(self, seed: SeedRecord, policy_fn: PolicyFn) -> RolloutEpisode:
        state = self.env.reset(seed)
        steps: List[RolloutStep] = []

        while not state.done:
            system_prompt = self.env.build_system_prompt()
            user_prompt = self.env.build_user_prompt(state)
            completion = policy_fn(system_prompt, user_prompt, state)
            step_result = self.env.step(state, completion)
            action = step_result["action"]
            observation = step_result["observation"]

            steps.append(
                RolloutStep(
                    turn_id=state.history[-1].turn_id if state.history else len(steps) + 1,
                    prompt=user_prompt,
                    completion=completion,
                    reward=step_result["reward"],
                    reward_breakdown=step_result["reward_breakdown"],
                    done=step_result["done"],
                    action_type=action.action_type if action else None,
                    sql=action.sql if action else "",
                    reasoning=action.reasoning if action else "",
                    observation=observation,
                )
            )

        total_reward = sum(step.reward for step in steps)
        return RolloutEpisode(
            seed_id=seed["seed_id"],
            difficulty=state.difficulty,
            total_reward=total_reward,
            done=state.done,
            final_failure_reason=state.final_failure_reason,
            steps=steps,
        )

    def rollout_many(self, seeds: List[SeedRecord], policy_fn: PolicyFn) -> List[RolloutEpisode]:
        return [self.rollout_seed(seed, policy_fn) for seed in seeds]

    @staticmethod
    def to_trl_step_records(episodes: List[RolloutEpisode]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for episode in episodes:
            for step in episode.steps:
                records.append(
                    {
                        "seed_id": episode.seed_id,
                        "difficulty": episode.difficulty,
                        "prompt": step.prompt,
                        "completion": step.completion,
                        "reward": step.reward,
                        "reward_breakdown": step.reward_breakdown,
                        "done": step.done,
                        "action_type": step.action_type,
                        "sql": step.sql,
                        "reasoning": step.reasoning,
                        "observation": step.observation,
                        "episode_total_reward": episode.total_reward,
                        "final_failure_reason": episode.final_failure_reason,
                    }
                )
        return records

    @staticmethod
    def to_serializable(episodes: List[RolloutEpisode]) -> List[Dict[str, Any]]:
        return [asdict(episode) for episode in episodes]
