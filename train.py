from models.model import RGBDepthNet
from data.dataset import LettuceDataset
from utils import EarlyStopping, get_scheduler
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = LettuceDataset("dataset/train/train.json", "dataset/train/", transform=transform)
train_len = int(0.7 * len(dataset))
val_len = int(0.15 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RGBDepthNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = get_scheduler(optimizer)
early_stopping = EarlyStopping(patience=10, save_path="best_model.pt")

for epoch in range(100):
    model.train()
    for rgb, depth, target in train_loader:
        rgb, depth = rgb.to(device), depth.to(device)
        y = torch.stack([
            target["fresh_weight"],
            target["dry_weight"],
            target["diameter"],
            target["height"],
            target["leaf_area"]
        ], dim=1).to(device)

        out1, out2, out3, out4 = model(rgb, depth)
        loss = criterion(out1[:, 0], y[:, 0]) + criterion(out1[:, 1], y[:, 1]) + \
               criterion(out1[:, 2], y[:, 2]) + criterion(out2[:, 0], y[:, 3]) + \
               criterion(out3[:, 0], y[:, 1]) + criterion(out4[:, 0], y[:, 4])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for rgb, depth, target in val_loader:
            rgb, depth = rgb.to(device), depth.to(device)
            y = torch.stack([
                target["fresh_weight"],
                target["dry_weight"],
                target["diameter"],
                target["height"],
                target["leaf_area"]
            ], dim=1).to(device)
            out1, out2, out3, out4 = model(rgb, depth)
            val_loss += criterion(out1[:, 0], y[:, 0]) + criterion(out2[:, 0], y[:, 3])

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping!")
        break