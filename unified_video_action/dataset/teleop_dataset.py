import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from unified_video_action.dataset.utils.teleop_dataset_utils import (
    load_arm_hand_config,
    load_robot_replay_buffer,
    load_human_replay_buffer,
)


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
        human_dataset_path: Optional[str] = None,
        p_robot: float = 1.0,
        p_human: float = 0.0,
        image_right_resolution=None,
    ):
        super().__init__()
        if arm_hand_name is not None:
            self.arm_hand_name = arm_hand_name
        assert self.arm_hand_name is not None, "arm_hand_name must be set"
        self.cfg = load_arm_hand_config(self.arm_hand_name)
        self.image_right_resolution = image_right_resolution if image_right_resolution is not None else self.cfg.image_right_resolution

        _keys = ["state", "action", "img_right", "ee_left", "ee_right", "q_left", "q_right"]

        # robot data
        self.replay_buffer = load_robot_replay_buffer(dataset_path, keys=_keys)
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
        self._n_robot = len(self.sampler)

        # human data
        self.human_replay_buffer = None
        self.human_sampler = None
        self.human_train_mask = None
        self._n_human = 0

        if human_dataset_path is not None and p_human > 0:
            self.human_replay_buffer = load_human_replay_buffer(
                human_dataset_path,
                img_res=self.cfg.image_resolution,
                img_right_res=self.image_right_resolution,
            )
            human_val_mask = get_val_mask(
                n_episodes=self.human_replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
            )
            human_train_mask = ~human_val_mask
            human_train_mask = downsample_mask(
                mask=human_train_mask, max_n=max_train_episodes, seed=seed
            )
            self.human_sampler = SequenceSampler(
                replay_buffer=self.human_replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=human_train_mask,
            )
            self.human_train_mask = human_train_mask
            self._n_human = len(self.human_sampler)

        # normalize proportions
        total = p_robot + p_human
        self.p_robot = p_robot / total
        self.p_human = p_human / total

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
        val_set._n_robot = len(val_set.sampler)

        if self.human_sampler is not None:
            val_set.human_sampler = SequenceSampler(
                replay_buffer=self.human_replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.human_train_mask,
            )
            val_set.human_train_mask = ~self.human_train_mask
            val_set._n_human = len(val_set.human_sampler)

        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        _keys = ["action", "state", "ee_left", "ee_right", "q_left", "q_right"]
        data = {k: self.replay_buffer[k] for k in _keys}

        if self.human_replay_buffer is not None:
            data = {k: np.concatenate([data[k], self.human_replay_buffer[k]], axis=0) for k in _keys}

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        normalizer["image_right"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return self._n_robot + self._n_human

    def _sample_to_data(self, sample):
        state = sample["state"].astype(np.float32)
        action = sample["action"].astype(np.float32)

        # center crop img_right
        img_right_raw = sample["img_right"]  # (T, H, W, 3)
        T, H, W, C = img_right_raw.shape
        crop = min(H, W)
        top, left = (H - crop) // 2, (W - crop) // 2
        img_right_cropped = img_right_raw[:, top:top + crop, left:left + crop, :]
        th = torch.from_numpy(img_right_cropped.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        res = self.cfg.image_resolution
        image = F.interpolate(th, size=(res, res), mode="bilinear", align_corners=False).numpy()
        image_right = F.interpolate(
            th, size=(self.image_right_resolution, self.image_right_resolution),
            mode="bilinear", align_corners=False,
        ).numpy()

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
        if idx < self._n_robot:
            sample = self.sampler.sample_sequence(idx)
        else:
            sample = self.human_sampler.sample_sequence(idx - self._n_robot)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)

    # weighted random sampler enforcing robot/human props
    def get_weighted_sampler(self, num_samples: Optional[int] = None) -> Optional[WeightedRandomSampler]:
        if self._n_human == 0 or self._n_robot == 0:
            return None
        weights = (
            [self.p_robot / self._n_robot] * self._n_robot
            + [self.p_human / self._n_human] * self._n_human
        )
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float64),
            num_samples=num_samples or len(self),
            replacement=True,
        )

    def get_dataloader(self, batch_size, num_workers=8, pin_memory=True,
                       persistent_workers=False, shuffle=True, **kwargs) -> DataLoader:
        sampler = self.get_weighted_sampler()
        if sampler is not None:
            return DataLoader(self, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent_workers, **kwargs)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=persistent_workers, **kwargs)


class RealKinovaXHandDataset(TeleopDataset):
    arm_hand_name = "realkinova_xhand"


class RealKinovaSharpaHandDataset(TeleopDataset):
    arm_hand_name = "realkinova_sharpa_hand"
