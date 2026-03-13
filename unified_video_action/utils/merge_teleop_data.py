"""
Merge multiple per-episode teleop zarr files into a single consolidated zarr.
"""

import argparse
import glob
import os

import numpy as np
import zarr


DEFAULT_KEYS = [
    "action",
    "ee_left",
    "ee_right",
    "img",
    "img_right",
    "q_left",
    "q_right",
    "state",
]


def merge_teleop_zarrs(input_dir: str, output_path: str, keys: list = None):
    if keys is None:
        keys = DEFAULT_KEYS

    zarr_paths = sorted(glob.glob(os.path.join(input_dir, "*.zarr")))
    zarr_paths = [p for p in zarr_paths if os.path.abspath(p) != os.path.abspath(output_path)]

    if not zarr_paths:
        raise ValueError(f"No *.zarr files found in {input_dir}")

    print(f"Merging {len(zarr_paths)} zarr files into {output_path}")

    # first pass: collect metadata without loading data
    episode_lengths = []
    present_keys = None
    key_meta = {}

    for zpath in zarr_paths:
        z = zarr.open(zpath, "r")
        if "meta" not in z or "episode_ends" not in z["meta"]:
            print(f"  Skipping (no meta/episode_ends): {os.path.basename(zpath)}")
            continue

        episode_ends = np.array(z["meta"]["episode_ends"])
        for ep_idx in range(len(episode_ends)):
            start = int(episode_ends[ep_idx - 1]) if ep_idx > 0 else 0
            end = int(episode_ends[ep_idx])
            episode_lengths.append((zpath, ep_idx, start, end, end - start))
            print(f"  {os.path.basename(zpath)} ep{ep_idx}: {end - start} steps")

        if present_keys is None:
            present_keys = sorted(set(z["data"].keys()) & set(keys))
            for k in present_keys:
                arr = z["data"][k]
                key_meta[k] = (arr.shape[1:], arr.dtype)

    if not episode_lengths:
        raise ValueError("No valid episodes found")

    for zpath, _, _, _, _ in episode_lengths:
        z = zarr.open(zpath, "r")
        present_keys = [k for k in present_keys if k in z["data"]]

    total_steps = sum(l for _, _, _, _, l in episode_lengths)
    print(f"\nKeys in output: {present_keys}")
    print(f"Total: {len(episode_lengths)} episodes, {total_steps} steps")

    # pre-allocate output arrays
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = zarr.open(output_path, mode="w")
    data_group = out.require_group("data")
    meta_group = out.require_group("meta")

    out_arrays = {}
    for k in present_keys:
        shape_per_step, dtype = key_meta[k]
        out_arrays[k] = data_group.zeros(
            k,
            shape=(total_steps,) + shape_per_step,
            chunks=(1,) + shape_per_step,
            dtype=dtype,
            overwrite=True,
        )
        print(f"  pre-allocated {k}: {out_arrays[k].shape} {dtype}")

    # second pass: write directly into pre-allocated arrays
    out_episode_ends = []
    cursor = 0
    for zpath, ep_idx, start, end, length in episode_lengths:
        z = zarr.open(zpath, "r")
        for k in present_keys:
            out_arrays[k][cursor:cursor + length] = z["data"][k][start:end]
        cursor += length
        out_episode_ends.append(cursor)

    meta_group.array(
        "episode_ends",
        np.array(out_episode_ends, dtype=np.int64),
        chunks=(len(out_episode_ends),),
        overwrite=True,
    )

    print(f"\nDone: {len(episode_lengths)} episodes, {total_steps} total steps")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/teleop/robot/realkinova_xhand")
    parser.add_argument("--output_path", type=str, default="data/teleop/robot/realkinova_xhand/combined/demo.zarr")
    parser.add_argument("--include_orig", action="store_true",
                        help="Also include img_left_orig and img_right_orig (large)")
    args = parser.parse_args()

    keys = DEFAULT_KEYS
    if args.include_orig:
        keys = keys + ["img_left_orig", "img_right_orig"]

    merge_teleop_zarrs(args.input_dir, args.output_path, keys=keys)
