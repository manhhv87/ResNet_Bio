import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import LettuceNet
from utils.dataset import get_dataloaders
import numpy as np

# Setup
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Data
train_loader, val_loader = get_dataloaders(
    json_path='data/train/train.json',
    image_dir='data/train',
    batch_size=16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LettuceNet().to(device)

# Loss + Optimizer
mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

best_val_loss = float('inf')
patience = 10
early_stop_counter = 0

# History dictionary
history = {
    'train': [], 'val': [],
    'train_fresh': [], 'train_diameter': [], 'train_height': [], 'train_dry': [], 'train_leaf': [],
    'val_fresh': [], 'val_diameter': [], 'val_height': [], 'val_dry': [], 'val_leaf': [],
}

# Training loop
for epoch in range(100):
    model.train()
    fresh_loss, dia_loss, height_loss, dry_loss, leaf_loss = 0, 0, 0, 0, 0

    for rgb, depth, y1, y2, y3, y4 in train_loader:
        # o1 = [fresh, diameter, leaf]
        # o2 = [height]
        # o3 = [dry]
        # o4 = [leaf]
        # y1 = [fresh, diameter, leaf]
        # y2 = [height]
        # y3 = [dry]
        # y4 = [leaf]        
        rgb, depth = rgb.to(device), depth.to(device)
        y1, y2, y3, y4 = y1.to(device), y2.to(device), y3.to(device), y4.to(device)

        optimizer.zero_grad()
        o1, o2, o3, o4 = model(rgb, depth)

        loss_fresh = mse(o1[:, 0], y1[:, 0])
        loss_dia = mse(o1[:, 1], y1[:, 1])
        loss_height = mse(o2.squeeze(), y2.squeeze())
        loss_dry = mse(o3.squeeze(), y3.squeeze())
        loss_leaf = mse(o1[:, 2], y4.squeeze()) + mse(o4.squeeze(), y4.squeeze())  # combined

        loss = loss_fresh + loss_dia + loss_height + loss_dry + loss_leaf
        loss.backward()
        optimizer.step()

        fresh_loss += loss_fresh.item()
        dia_loss += loss_dia.item()
        height_loss += loss_height.item()
        dry_loss += loss_dry.item()
        leaf_loss += loss_leaf.item()

    # Average
    n = len(train_loader)
    history['train_fresh'].append(fresh_loss / n)
    history['train_diameter'].append(dia_loss / n)
    history['train_height'].append(height_loss / n)
    history['train_dry'].append(dry_loss / n)
    history['train_leaf'].append(leaf_loss / n)
    train_total = sum([fresh_loss, dia_loss, height_loss, dry_loss, leaf_loss]) / n
    history['train'].append(train_total)

    # Validation
    model.eval()
    fresh_loss, dia_loss, height_loss, dry_loss, leaf_loss = 0, 0, 0, 0, 0

    with torch.no_grad():
        for rgb, depth, y1, y2, y3, y4 in val_loader:
            rgb, depth = rgb.to(device), depth.to(device)
            y1, y2, y3, y4 = y1.to(device), y2.to(device), y3.to(device), y4.to(device)

            o1, o2, o3, o4 = model(rgb, depth)

            fresh_loss += mse(o1[:, 0], y1[:, 0]).item()
            dia_loss += mse(o1[:, 1], y1[:, 1]).item()
            height_loss += mse(o2.squeeze(), y2.squeeze()).item()
            dry_loss += mse(o3.squeeze(), y3.squeeze()).item()
            leaf_loss += (mse(o1[:, 2], y4.squeeze()) + mse(o4.squeeze(), y4.squeeze())).item()

    history['val_fresh'].append(fresh_loss / len(val_loader))
    history['val_diameter'].append(dia_loss / len(val_loader))
    history['val_height'].append(height_loss / len(val_loader))
    history['val_dry'].append(dry_loss / len(val_loader))
    history['val_leaf'].append(leaf_loss / len(val_loader))
    val_total = sum([fresh_loss, dia_loss, height_loss, dry_loss, leaf_loss]) / len(val_loader)
    history['val'].append(val_total)

    print(f"Epoch {epoch+1:02d} | Train: {train_total:.4f} "
          f"(fresh={history['train_fresh'][-1]:.4f}, dia={history['train_diameter'][-1]:.4f}, "
          f"height={history['train_height'][-1]:.4f}, dry={history['train_dry'][-1]:.4f}, "
          f"leaf={history['train_leaf'][-1]:.4f}) | "
          f"Val: {val_total:.4f} "
          f"(fresh={history['val_fresh'][-1]:.4f}, dia={history['val_diameter'][-1]:.4f}, "
          f"height={history['val_height'][-1]:.4f}, dry={history['val_dry'][-1]:.4f}, "
          f"leaf={history['val_leaf'][-1]:.4f})")

    # Checkpoint
    if val_total < best_val_loss:
        best_val_loss = val_total
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping.")
            break

    scheduler.step(val_total)

# Save history
np.save("results/train_class_losses.npy", history)
