import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LettuceDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path) as f:
            self.data = json.load(f)["Measurements"]
        self.keys = list(self.data.keys())
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]
        rgb_path = os.path.join(self.image_dir, item["RGB_Image"])
        depth_path = os.path.join(self.image_dir, item["Depth_Information"])

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        if self.transform:
            rgb = self.transform(rgb)
            depth = self.transform(depth)

        target = {
            "fresh_weight": torch.tensor(item["FreshWeightShoot"], dtype=torch.float32),
            "dry_weight": torch.tensor(item["DryWeightShoot"], dtype=torch.float32),
            "height": torch.tensor(item["Height"], dtype=torch.float32),
            "diameter": torch.tensor(item["Diameter"], dtype=torch.float32),
            "leaf_area": torch.tensor(item["LeafArea"], dtype=torch.float32),
        }
        return rgb, depth, target
