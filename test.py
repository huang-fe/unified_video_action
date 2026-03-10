import argparse
from unified_video_action.dataset.teleop_dataset import RealKinovaXHandDataset, RealKinovaSharpaHandDataset

DATASET_CLASSES = {
    "realkinova_xhand": RealKinovaXHandDataset,
    "realkinova_sharpa_hand": RealKinovaSharpaHandDataset,
}

parser = argparse.ArgumentParser()
parser.add_argument("--arm-hand", type=str, default="realkinova_xhand", choices=list(DATASET_CLASSES.keys()))
parser.add_argument("--dataset-path", type=str, default=None)
args = parser.parse_args()

dataset_path = args.dataset_path or f"data/{args.arm_hand}/demo.zarr"
DatasetClass = DATASET_CLASSES[args.arm_hand]
d = DatasetClass(dataset_path=dataset_path)

print("len:", len(d))
sample = d[0]
print(type(sample))
print(sample.keys() if sample else None)
