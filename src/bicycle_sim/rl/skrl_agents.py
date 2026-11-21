"""PPO/SAC models for the optional, pinned skrl 1.4.2 training dependency."""

import copy
import math

import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch import nn


def mlp(inputs, outputs, hidden):
    layers = []
    for size in hidden:
        layers += [nn.Linear(inputs, size), nn.ELU()]
        inputs = size
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden, *, raw_gaussian=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=not raw_gaussian, clip_log_std=True, min_log_std=-5, max_log_std=1
        )
        self.raw_gaussian = raw_gaussian
        self.net = mlp(self.num_observations, self.num_actions, hidden)
        self.log_std = nn.Parameter(torch.full((self.num_actions,), -1.0))

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        if not self.raw_gaussian:
            mean = torch.tanh(mean)
        return mean, self.log_std, {}


class SquashedPolicy(Policy):
    """Reparameterized tanh Gaussian with the Jacobian required by SAC."""

    def act(self, inputs, role=""):
        mean = self.net(inputs["states"])
        std = self.log_std.clamp(-5, 1).exp()
        distribution = torch.distributions.Normal(mean, std)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        # Stable log(1-tanh(raw)^2), including near action limits.
        correction = 2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
        log_prob = (distribution.log_prob(raw) - correction).sum(-1, keepdim=True)
        return action, log_prob, {"mean_actions": torch.tanh(mean)}


def deterministic_actions(policy, states, algorithm):
    """Match the mean-action path used by the GPU evaluator, without sampling."""
    if algorithm == "sac":
        return torch.tanh(policy.net(states))
    if algorithm == "ppo":
        return policy.compute({"states": states}, "policy")[0]
    raise ValueError(f"Unknown algorithm: {algorithm}")


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden, *, q_value=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.q_value = q_value
        self.net = mlp(self.num_observations + (self.num_actions if q_value else 0), 1, hidden)

    def compute(self, inputs, role):
        x = inputs["states"]
        if self.q_value:
            x = torch.cat((x, inputs["taken_actions"]), -1)
        return self.net(x), {}


def make_agent(algorithm, env, cfg, output, *, smoke=False, evaluation=False):
    import numpy as np
    from gymnasium.spaces import Box

    obs = Box(-np.inf, np.inf, shape=(env.num_observations,), dtype=np.float32)
    act = Box(-1, 1, shape=(getattr(env, "num_actions", 14),), dtype=np.float32)
    common = (obs, act, env.device, cfg["hidden_dims"])
    experiment = {
        "directory": str(output),
        "experiment_name": "logs",
        "write_interval": 100,
        "checkpoint_interval": 2000,
        "wandb": False,
    }
    if algorithm == "ppo":
        settings = copy.deepcopy(PPO_DEFAULT_CONFIG)
        settings.update(
            rollouts=cfg["ppo_rollouts"],
            learning_epochs=4,
            mini_batches=4,
            learning_rate=cfg["learning_rate"],
            entropy_loss_scale=0.002,
            time_limit_bootstrap=True,
        )
        memory = RandomMemory(
            memory_size=cfg["ppo_rollouts"], num_envs=env.num_envs, device=env.device
        )
        models = {
            # Store the original Gaussian draw and its matching density in PPO.
            # The physical environment alone clips the executed arm command.
            # Missing flag preserves evaluation of old censored-action runs.
            "policy": Policy(*common, raw_gaussian=cfg.get("ppo_raw_gaussian", False)),
            "value": Critic(*common),
        }
        cls = PPO
    else:
        settings = copy.deepcopy(SAC_DEFAULT_CONFIG)
        warmup = max(1, math.ceil(cfg["sac_learning_starts_transitions"] / env.num_envs))
        settings.update(
            batch_size=32 if smoke else cfg["sac_batch_size"],
            actor_learning_rate=cfg["learning_rate"],
            critic_learning_rate=cfg["learning_rate"],
            entropy_learning_rate=cfg["learning_rate"],
            gradient_steps=1,
            random_timesteps=1 if smoke else warmup,
            learning_starts=1 if smoke else warmup,
        )
        rows = 128 if smoke else max(1, math.ceil(cfg["sac_replay_transitions"] / env.num_envs))
        memory = RandomMemory(memory_size=rows, num_envs=env.num_envs, device=env.device)
        models = {"policy": SquashedPolicy(*common)}
        for name in ("critic_1", "critic_2", "target_critic_1", "target_critic_2"):
            models[name] = Critic(*common, q_value=True)
        cls = SAC
    if evaluation:
        settings["random_timesteps"] = 0
    settings["experiment"] = experiment
    for model in models.values():
        model.to(env.device)
    return cls(
        models=models,
        memory=memory,
        cfg=settings,
        observation_space=obs,
        action_space=act,
        device=env.device,
    )
