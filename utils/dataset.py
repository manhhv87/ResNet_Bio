import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class LettuceDataset(Dataset):
    def __init__(self, json_path, image_dir):
        with open(json_path, 'r') as f:
            data = json.load(f)["Measurements"]

        self.samples = list(data.values())
        self.image_dir = image_dir

        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        rgb_path = os.path.join(self.image_dir, entry['RGB_Image'])
        depth_path = os.path.join(self.image_dir, entry['Depth_Information'])

        rgb = self.rgb_transform(Image.open(rgb_path).convert('RGB'))
        depth = self.depth_transform(Image.open(depth_path).convert('L'))

        # Labels
        fresh = entry['FreshWeightShoot']
        diameter = entry['Diameter']
        leaf = entry['LeafArea']
        height = entry['Height']
        dry = entry['DryWeightShoot']

        y1 = torch.tensor([fresh, diameter, leaf], dtype=torch.float32)
        y2 = torch.tensor([height], dtype=torch.float32)
        y3 = torch.tensor([dry], dtype=torch.float32)
        y4 = torch.tensor([leaf], dtype=torch.float32)  # again for Output 4

        return rgb, depth, y1, y2, y3, y4


def get_dataloaders(json_path, image_dir, batch_size=16, val_split=0.2):
    dataset = LettuceDataset(json_path, image_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
