import argparse
import zarr
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--arm-hand", type=str, default="realkinova_xhand",
                    choices=["realkinova_xhand", "realkinova_sharpa_hand"])
parser.add_argument("--zarr-path", type=str, default="demo_1772431982_e94f08.zarr")
parser.add_argument("--checkpoint", type=str, default="checkpoints/libero10.ckpt")
args = parser.parse_args()

z = zarr.open(args.zarr_path, "r")

imgs = z["data/img"][:]        # (T,H,W,C)

imgs = imgs.transpose(0,3,1,2) # (T,C,H,W)

imgs = torch.tensor(imgs).float() / 255.0

import torch.nn.functional as F

imgs = F.interpolate(
    imgs,
    size=(128,128),
    mode="bilinear"
)

from omegaconf import OmegaConf
import torch
from unified_video_action.policy.unified_video_action_policy import UnifiedVideoActionPolicy

ckpt = torch.load(args.checkpoint, map_location="cpu")
# print(ckpt["state_dicts"].keys())
policy_cfg = ckpt["cfg"]["model"]["policy"]

policy_cfg = OmegaConf.to_container(policy_cfg, resolve=True)

policy_cfg["normalizer_type"] = "all"
policy_cfg["task_name"] = "libero_10"

# disable missing pretrained model
policy_cfg["autoregressive_model_params"]["pretrained_model_path"] = None

policy_cfg = OmegaConf.create(policy_cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedVideoActionPolicy(**policy_cfg)
model.load_state_dict(ckpt["state_dicts"]["ema_model"], strict=False)
model = model.to(device)
model.eval()

window = 16
pred_actions = []

for i in range(len(imgs) - window):

    obs = imgs[i:i+window].unsqueeze(0).to(device)   # (1,16,3,128,128)

    batch = {
      "image": obs
    }

    with torch.no_grad():
        out = model.predict_action(batch)

    # print(out["action"])

    pred_actions.append(out["action"].cpu().numpy())
  
gt_actions = z["data/action"][:]

print(gt_actions.shape)
print(np.array(pred_actions).shape)

import matplotlib.pyplot as plt

gt = gt_actions[16:16+211,0]
pred = np.array(pred_actions)[:,0,0,0]

fig = plt.figure()

plt.plot(gt, label="gt")
plt.plot(pred, label="pred")
plt.legend()

fig.savefig("actions_plot.png", dpi=300, bbox_inches="tight")
plt.close(fig)

mse = np.mean((gt - pred)**2)
print("MSE:", mse)
corr = np.corrcoef(gt, pred)[0,1]
print("Correlation:", corr)
print("gt mean/std", gt.mean(), gt.std())
print("pred mean/std", pred.mean(), pred.std())