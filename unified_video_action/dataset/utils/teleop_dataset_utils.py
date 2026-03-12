import glob
import io
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from unified_video_action.common.replay_buffer import ReplayBuffer


def load_arm_hand_config(arm_hand_name):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "arm_hand", f"{arm_hand_name}.yaml"
    )
    return OmegaConf.load(config_path)


def center_crop_resize(arr: np.ndarray, target_hw: int) -> np.ndarray:
    if arr.shape[1] == target_hw and arr.shape[2] == target_hw:
        return arr
    T, H, W, C = arr.shape
    crop = min(H, W)
    top = (H - crop) // 2
    left = (W - crop) // 2
    arr = arr[:, top:top + crop, left:left + crop, :]
    th = torch.from_numpy(arr.astype(np.float32)).permute(0, 3, 1, 2)  # (T,C,H,W)
    th = F.interpolate(th, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return th.permute(0, 2, 3, 1).numpy().astype(np.uint8)


# human data helpers
_HAND_KEY_JOINTS: List[str] = [
    "wrist",
    "forearmWrist",
    "thumbTip",
    "indexFingerTip",
    "middleFingerMetacarpal",
    "ringFingerMetacarpal",
]


def _transform_to_ee(joint_transforms: dict, joint_name: str) -> np.ndarray:
    mat = np.array(joint_transforms[joint_name], dtype=np.float64)  # (4, 4)
    pos = mat[3, :3].astype(np.float32)
    R = mat[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = float(np.arctan2(-R[1, 2], R[1, 1]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return np.array([pos[0], pos[1], pos[2], rx, ry, rz], dtype=np.float32)


def _joints_to_q(joint_transforms: dict, q_dim: int = 19) -> np.ndarray:
    vals: List[float] = []
    for name in _HAND_KEY_JOINTS:
        if name in joint_transforms:
            mat = np.array(joint_transforms[name], dtype=np.float64)
            vals.extend(mat[3, :3].tolist())
    arr = np.array(vals, dtype=np.float32)
    if len(arr) >= q_dim:
        return arr[:q_dim]
    return np.pad(arr, (0, q_dim - len(arr)), constant_values=0.0).astype(np.float32)


def _read_zst_jsonl(path: str) -> List[dict]:
    import zstandard
    records = []
    with open(path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_human_demo(demo_dir: str, img_res: int = 96, img_right_res: int = 256) -> dict:
    frames_meta = sorted(
        _read_zst_jsonl(os.path.join(demo_dir, "frames_meta.jsonl.zst")),
        key=lambda x: x["frame_id"],
    )
    frame_ts = np.array([fm["ts"] for fm in frames_meta])

    telemetry = _read_zst_jsonl(os.path.join(demo_dir, "telemetry.jsonl.zst"))
    telem_ts = np.array([t["ts"] for t in telemetry])
    nearest = np.argmin(np.abs(frame_ts[:, None] - telem_ts[None, :]), axis=1)

    frames_dir = os.path.join(demo_dir, "frames")
    imgs_left, imgs_right = [], []
    ee_lefts, ee_rights, q_lefts, q_rights = [], [], [], []

    for i, fm in enumerate(frames_meta):
        fid = fm["frame_id"]
        left_files = glob.glob(os.path.join(frames_dir, f"frame_{fid:06d}_*_left.jpg"))
        right_files = glob.glob(os.path.join(frames_dir, f"frame_{fid:06d}_*_right.jpg"))

        if left_files:
            pil = Image.open(left_files[0])
            w, h = pil.size
            crop = min(w, h)
            pil = pil.crop(((w - crop) // 2, (h - crop) // 2,
                            (w + crop) // 2, (h + crop) // 2))
            img_l = np.array(pil.resize((img_res, img_res), Image.BILINEAR), dtype=np.uint8)
        else:
            img_l = np.zeros((img_res, img_res, 3), dtype=np.uint8)

        if right_files:
            pil = Image.open(right_files[0])
            w, h = pil.size
            crop = min(w, h)
            pil = pil.crop(((w - crop) // 2, (h - crop) // 2,
                            (w + crop) // 2, (h + crop) // 2))
            img_r = np.array(pil.resize((img_right_res, img_right_res), Image.BILINEAR), dtype=np.uint8)
        else:
            img_r = np.zeros((img_right_res, img_right_res, 3), dtype=np.uint8)

        imgs_left.append(img_l)
        imgs_right.append(img_r)

        telem = telemetry[nearest[i]]
        ee_lefts.append(_transform_to_ee(telem["left_joint_transforms"], "wrist"))
        ee_rights.append(_transform_to_ee(telem["right_joint_transforms"], "wrist"))
        q_lefts.append(_joints_to_q(telem["left_joint_transforms"]))
        q_rights.append(_joints_to_q(telem["right_joint_transforms"]))

    q_left_arr = np.stack(q_lefts)
    q_right_arr = np.stack(q_rights)
    state = np.concatenate([q_left_arr, q_right_arr], axis=1)
    action = np.concatenate([state[1:], state[-1:]], axis=0)

    return {
        "img":       np.stack(imgs_left),
        "img_right": np.stack(imgs_right),
        "ee_left":   np.stack(ee_lefts),
        "ee_right":  np.stack(ee_rights),
        "q_left":    q_left_arr,
        "q_right":   q_right_arr,
        "state":     state,
        "action":    action,
    }


def load_human_replay_buffer(dataset_dir: str, img_res: int = 96, img_right_res: int = 256) -> ReplayBuffer:
    demo_dirs = sorted(
        d for d in (os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "frames_meta.jsonl.zst"))
    )
    if not demo_dirs:
        raise ValueError(f"No valid human demo folders found in {dataset_dir}")
    buf = ReplayBuffer.create_empty_numpy()
    for demo_dir in demo_dirs:
        print(f"  Loading human demo: {os.path.basename(demo_dir)}")
        buf.add_episode(load_human_demo(demo_dir, img_res=img_res, img_right_res=img_right_res))
    return buf


def load_robot_replay_buffer(path: str, keys: list, img_res: int = 96, img_right_res: int = 256) -> ReplayBuffer:
    is_zarr_store = os.path.exists(os.path.join(path, "meta"))
    if is_zarr_store:
        return ReplayBuffer.copy_from_path(path, keys=keys)

    zarr_paths = sorted(glob.glob(os.path.join(path, "*.zarr")))
    if not zarr_paths:
        raise ValueError(f"No *.zarr files found in {path}")
    buf = ReplayBuffer.create_empty_numpy()
    for zpath in zarr_paths:
        print(f"  Loading robot zarr: {os.path.basename(zpath)}")
        ep_buf = ReplayBuffer.copy_from_path(zpath, keys=keys)
        ends = ep_buf.episode_ends[:]
        for ep_idx in range(ep_buf.n_episodes):
            start = int(ends[ep_idx - 1]) if ep_idx > 0 else 0
            end = int(ends[ep_idx])
            buf.add_episode({k: ep_buf[k][start:end] for k in keys if k in ep_buf})
    return buf
