import cv2
import numpy as np
import torch
import torch.nn.functional as F


def center_crop_resize(img_hwc: np.ndarray, target: int) -> np.ndarray:
    h, w = img_hwc.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    cropped = img_hwc[top:top + s, left:left + s]
    if s != target:
        cropped = cv2.resize(cropped, (target, target), interpolation=cv2.INTER_LINEAR)
    return cropped


def center_crop_resize_batch(arr: np.ndarray, target: int) -> np.ndarray:
    if arr.shape[1] == target and arr.shape[2] == target:
        return arr
    T, H, W, C = arr.shape
    s = min(H, W)
    top, left = (H - s) // 2, (W - s) // 2
    arr = arr[:, top:top + s, left:left + s, :]
    th = torch.from_numpy(arr.astype(np.float32)).permute(0, 3, 1, 2)  # (T, C, H, W)
    th = F.interpolate(th, size=(target, target), mode="bilinear", align_corners=False)
    return th.permute(0, 2, 3, 1).numpy().astype(np.uint8)
