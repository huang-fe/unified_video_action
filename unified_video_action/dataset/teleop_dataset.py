import os
from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
import copy
from omegaconf import OmegaConf
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.common.replay_buffer import ReplayBuffer
from unified_video_action.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.dataset.base_dataset import BaseImageDataset
from unified_video_action.common.normalize_util import get_image_range_normalizer


def load_arm_hand_config(arm_hand_name):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "arm_hand", f"{arm_hand_name}.yaml"
    )
    return OmegaConf.load(config_path)


class TeleopDataset(BaseImageDataset):
    arm_hand_name: str = None  # defined in subclasses

    def __init__(
        self,
        dataset_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        language_emb_model=None,
        normalizer_type=None,
        dataset_type="singletask",
        arm_hand_name=None,
    ):
        super().__init__()
        if arm_hand_name is not None:
            self.arm_hand_name = arm_hand_name
        assert self.arm_hand_name is not None, "arm_hand_name must be set"
        self.cfg = load_arm_hand_config(self.arm_hand_name)
        self.image_right_resolution = self.cfg.image_right_resolution

        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path,
            keys=["img", "state", "action", "img_right", "ee_left", "ee_right", "q_left", "q_right"],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_type = dataset_type

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "state": self.replay_buffer["state"],
            "ee_left": self.replay_buffer["ee_left"],
            "ee_right": self.replay_buffer["ee_right"],
            "q_left": self.replay_buffer["q_left"],
            "q_right": self.replay_buffer["q_right"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        normalizer["image_right"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0
        state = sample["state"].astype(np.float32)
        action = sample["action"].astype(np.float32)

        img_right = sample["img_right"]
        if (
            img_right.shape[1] != self.image_right_resolution
            or img_right.shape[2] != self.image_right_resolution
        ):
            th = torch.from_numpy(np.moveaxis(img_right, -1, 1).astype(np.float32) / 255.0)
            th = F.interpolate(
                th,
                size=(self.image_right_resolution, self.image_right_resolution),
                mode="bilinear",
                align_corners=False,
            )
            image_right = th.numpy()
        else:
            image_right = np.moveaxis(img_right, -1, 1).astype(np.float32) / 255.0

        obs = {
            "image": image,
            "state": state,
            "image_right": image_right,
            "ee_left": sample["ee_left"].astype(np.float32),
            "ee_right": sample["ee_right"].astype(np.float32),
            "q_left": sample["q_left"].astype(np.float32),
            "q_right": sample["q_right"].astype(np.float32),
        }
        return {"obs": obs, "action": action}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)


class RealKinovaXHandDataset(TeleopDataset):
    arm_hand_name = "realkinova_xhand"


class RealKinovaSharpaHandDataset(TeleopDataset):
    arm_hand_name = "realkinova_sharpa_hand"
