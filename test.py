# import zarr
# # import imageio

# z = zarr.open("data/raw/demo_umi.zarr", mode="r")
# print(z.tree())

# imgs = z["data/img"]
# print(imgs.shape)

# imageio.mimsave("demo.mp4", imgs, fps=30,codec="libx264")

# from unified_video_action.dataset.xhand_dataset import XHandDataset

# d = XHandDataset("demo_1772431982_e94f08.zarr")

# print(len(d))
# print(d[0]["obs"]["img"].shape)
# print(d[0]["action"].shape)

# loader = d.get_dataloader()

# for batch in loader:
#     print(batch["obs"]["img"].shape)
#     break

from unified_video_action.dataset.xhand_dataset import XHandDataset

d = XHandDataset(dataset_path="/home/huangfe/unified_video_action/data/raw/demo.zarr")

print("len:", len(d))
sample = d[0]
print(type(sample))
print(sample.keys() if sample else None)