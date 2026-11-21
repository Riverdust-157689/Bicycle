"""Small CPU reference adapter for installed RSL-RL; not a GPU simulator."""

import copy

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from bicycle_sim.rl.bicycle_env import BicycleEnv


class BicycleVecEnv(VecEnv):
    def __init__(self, bike_config, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.device = "cpu"
        self.num_envs, self.num_actions = cfg["num_envs"], 2
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        first = BicycleEnv(bike_config, cfg, seed=cfg["seed"])
        self.envs = [first] + [
            BicycleEnv(bike_config, cfg, seed=cfg["seed"] + i, model=first.model)
            for i in range(1, self.num_envs)
        ]
        self.max_episode_length = first.max_steps
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.returns = np.zeros(self.num_envs)

    def get_observations(self):
        obs = torch.from_numpy(np.stack([e.observe() for e in self.envs]))
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def step(self, actions):
        actions = actions.detach().cpu().numpy()
        rewards, dones, timeouts = [], [], []
        completed_returns, completed_lengths = [], []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.returns[i] += reward
            self.episode_length_buf[i] += 1
            if done:
                completed_returns.append(self.returns[i])
                completed_lengths.append(env.steps * env.dt)
                env.reset()
                self.episode_length_buf[i] = 0
                self.returns[i] = 0
            rewards.append(reward)
            dones.append(done)
            timeouts.append(truncated)
        extras = {"time_outs": torch.tensor(timeouts, dtype=torch.bool)}
        if completed_returns:
            extras["log"] = {
                "Episode/return": float(np.mean(completed_returns)),
                "Episode/seconds": float(np.mean(completed_lengths)),
            }
        return (
            self.get_observations(),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
            extras,
        )
