# Real-robot inference server for teleop arm+hand configs adapted from eval_real.py.
#
# Author: Timothy Yu
# Date: 3.12.2026

import os
import time
import traceback

import click
import dill
import hydra
import numpy as np
import omegaconf
import torch
import zmq
from omegaconf import open_dict

from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.teleop_utils import center_crop_resize


def preprocess_obs(obs_dict_np: dict, device: torch.device) -> dict:
    processed = {}

    img_field = obs_dict_np["image"]
    if isinstance(img_field, dict):
        img_np = img_field.get("right", next(iter(img_field.values())))  # (T, H, W, 3)
    else:
        img_np = img_field

    if img_np.ndim == 4 and img_np.shape[1] == 3:  # (T, 3, H, W) -> (T, H, W, 3)
        img_np = img_np.transpose(0, 2, 3, 1)

    if img_np.shape[1] != 256 or img_np.shape[2] != 256:
        img_np = np.stack([center_crop_resize(img_np[t]) for t in range(len(img_np))])

    # (T, H, W, 3) -> (1, T, 3, H, W) float [0, 1]
    img_t = torch.from_numpy(img_np).float().permute(0, 3, 1, 2).unsqueeze(0)
    processed["image"] = img_t.to(device)

    # other keys
    for k, v in obs_dict_np.items():
        if k == "image":
            continue
        if isinstance(v, np.ndarray):
            processed[k] = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device)

    return processed


def split_action(action_flat: np.ndarray, q_dim: int) -> dict:
    return {
        "left": action_flat[:, :q_dim],
        "right": action_flat[:, q_dim:],
    }


# inference node 
class TeleopInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int, device: str, output_dir: str):
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")

        payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
        self.cfg = payload["cfg"]

        with open_dict(self.cfg):
            if "autoregressive_model_params" in self.cfg.model.policy:
                self.cfg.model.policy.autoregressive_model_params.num_sampling_steps = "100"

        cfg_path = ckpt_path.replace(".ckpt", ".yaml")
        with open(cfg_path, "w") as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
        print(f"Config exported to {cfg_path}")
        print(f"Task: {self.cfg.task.name} | Workspace: {self.cfg.model._target_}")

        cls = hydra.utils.get_class(self.cfg.model._target_)
        self.workspace = cls(self.cfg, output_dir=output_dir)
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy: BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")

        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()

        action_dim = self.cfg.task.shape_meta.action.shape[0]
        self.q_dim = action_dim // 2

        self.ip = ip
        self.port = port

    def predict(self, obs_dict_np: dict) -> dict:
        obs_dict = preprocess_obs(obs_dict_np, self.device)

        with torch.no_grad():
            result = self.policy.predict_action(obs_dict=obs_dict, language_goal=None)

        action_flat = result["action_pred"][0].detach().cpu().numpy()  # (n_steps, action_dim)
        return split_action(action_flat, self.q_dim)

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://{self.ip}:{self.port}")
        print(f"Listening on {self.ip}:{self.port}")

        while True:
            obs_dict_np = sock.recv_pyobj()
            try:
                t0 = time.monotonic()
                action = self.predict(obs_dict_np)
                print(f"Inference: {time.monotonic() - t0:.3f}s")
            except Exception:
                tb = traceback.format_exc()
                print(tb)
                action = tb
            sock.send_pyobj(action)


@click.command()
@click.option("--input", "-i", required=True, help="Checkpoint path or directory")
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default=8767, type=int)
@click.option("--device", default="cuda")
@click.option("--output_dir", required=True)
def main(input, ip, port, device, output_dir):
    node = TeleopInferenceNode(input, ip, port, device, output_dir)
    node.run()


if __name__ == "__main__":
    main()
