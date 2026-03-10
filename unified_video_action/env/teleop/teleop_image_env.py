"""
Teleop image env to play back episodes from zarr dataset.
Modular design compatible with any arm+hand config.
"""

import os
import gym
from gym import spaces
import numpy as np
import zarr
from unified_video_action.dataset.teleop_dataset import load_arm_hand_config


class TeleopImageEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, dataset_path, arm_hand_name, render_size=96):
        super().__init__()
        self._seed = None
        self.seed()
        self.render_size = render_size
        self.dataset_path = os.path.expanduser(dataset_path)
        self.cfg = load_arm_hand_config(arm_hand_name)
        self._root = None
        self._episode_ends = None
        self._n_episodes = None
        self._load_meta()

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(3, render_size, render_size),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.cfg.state_dim,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )
        self.reward_range = (0.0, 1.0)

        self._img = None
        self._state = None
        self._action = None
        self._t = 0
        self._T = 0

    def _load_meta(self):
        root = zarr.open(self.dataset_path, "r")
        self._episode_ends = np.array(root["meta"]["episode_ends"])
        self._n_episodes = len(self._episode_ends)
        self._root = root

    def seed(self, seed=None):
        self._seed = seed
        return [seed]

    def reset(self):
        if self._n_episodes == 0:
            raise RuntimeError("Zarr has no episodes")
        episode_idx = (self._seed if self._seed is not None else 0) % self._n_episodes
        start = 0 if episode_idx == 0 else int(self._episode_ends[episode_idx - 1])
        end = int(self._episode_ends[episode_idx])

        data = self._root["data"]
        self._img = np.array(data["img"][start:end])
        self._state = np.array(data["state"][start:end], dtype=np.float32)
        self._action = np.array(data["action"][start:end], dtype=np.float32)
        self._t = 0
        self._T = end - start

        return self._get_obs(0)

    def step(self, action):
        self._t += 1
        done = self._t >= self._T
        obs = self._get_obs(self._t - 1 if done else self._t)
        return obs, 0.0, done, {}

    def _get_obs(self, t):
        image = np.moveaxis(self._img[t].astype(np.float32) / 255.0, -1, 0)
        state = self._state[t].astype(np.float32)
        return {"image": image, "state": state}

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        if self._img is None or self._t >= self._T:
            return np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)
        t = min(self._t, self._T - 1)
        frame = self._img[t]
        if frame.shape[0] != self.render_size or frame.shape[1] != self.render_size:
            import cv2
            frame = cv2.resize(frame, (self.render_size, self.render_size), interpolation=cv2.INTER_LINEAR)
        return frame
