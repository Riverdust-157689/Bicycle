"""Fixed forward-speed reward and episode metrics; no speed-based termination."""

import math

import torch


def validate_speed_objective(objective, episode_seconds):
    if objective is None:
        return
    keys = (
        "target_mps",
        "reward_sigma_mps",
        "minimum_mps",
        "maximum_mps",
        "low_speed_penalty_weight",
        "success_fraction",
        "success_tail_seconds",
    )
    if any(not math.isfinite(objective[k]) for k in keys):
        raise ValueError("Speed objective parameters must be finite")
    if not (0 < objective["minimum_mps"] < objective["target_mps"] < objective["maximum_mps"]):
        raise ValueError("Speed objective requires 0 < minimum < target < maximum")
    if objective["reward_sigma_mps"] <= 0 or objective["low_speed_penalty_weight"] <= 0:
        raise ValueError("Speed reward width and low-speed penalty must be positive")
    if not 0 < objective["success_fraction"] <= 1:
        raise ValueError("Speed success fraction must be in (0, 1]")
    if not 0 < objective["success_tail_seconds"] <= episode_seconds:
        raise ValueError("Speed success tail must fit within the episode")


def speed_reward_terms(forward_speed, objective):
    score = torch.exp(
        -((forward_speed - objective["target_mps"]) / objective["reward_sigma_mps"]).square()
    )
    deficit = ((objective["minimum_mps"] - forward_speed) / objective["minimum_mps"]).clamp(0, 1)
    return score, objective["low_speed_penalty_weight"] * deficit


class EpisodeSpeedMetrics:
    """Accumulate post-step samples, with a fixed final window before timeout.

    Resets are per-world. Qualification is only a success metric and never
    changes terminated/truncated, replay bootstrapping or reward.
    """

    def __init__(self, num_envs, max_steps, dt, objective, device):
        validate_speed_objective(objective, max_steps * dt)
        self.objective = objective
        self.tail_start = max_steps - max(1, round(objective["success_tail_seconds"] / dt))
        self.total = torch.zeros(num_envs, device=device)
        self.good = torch.zeros_like(self.total)
        self.tail_good = torch.zeros_like(self.total)
        self.tail_count = torch.zeros_like(self.total)

    def reset(self, ids):
        for values in (self.total, self.good, self.tail_good, self.tail_count):
            values[ids] = 0

    def update(self, speed, length, truncated):
        good = (speed >= self.objective["minimum_mps"]) & (speed <= self.objective["maximum_mps"])
        tail = length > self.tail_start
        self.total += speed
        self.good += good
        self.tail_good += good & tail
        self.tail_count += tail
        fraction = self.good / length.clamp_min(1)
        tail_fraction = self.tail_good / self.tail_count.clamp_min(1)
        required = self.objective["success_fraction"]
        return {
            "episode_mean_forward_speed_mps": self.total / length.clamp_min(1),
            "episode_speed_in_band_fraction": fraction.clone(),
            "episode_tail_speed_in_band_fraction": tail_fraction.clone(),
            "speed_qualified_success": truncated
            & (fraction >= required)
            & (tail_fraction >= required),
        }
