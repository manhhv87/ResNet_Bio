import torch
from torch.utils.data import DataLoader
from data.dataset import LettuceDataset
from models.model import RGBDepthNet
from torchvision import transforms
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for rgb, depth, target in dataloader:
            rgb, depth = rgb.to(device), depth.to(device)
            target_stack = torch.stack([
                target["fresh_weight"],
                target["dry_weight"],
                target["diameter"],
                target["height"],
                target["leaf_area"]
            ], dim=1).to(device)

            out1, out2, out3, out4 = model(rgb, depth)
            outputs = torch.cat([
                out1[:, 0:3],           # FreshWeight, DryWeight (1), Diameter
                out2,                  # Height
                out4                   # LeafArea
            ], dim=1)

            y_pred.append(outputs.cpu().numpy())
            y_true.append(target_stack.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    mse = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')

    print("Evaluation Results:")
    metrics = ["FreshWeight", "DryWeight", "Diameter", "Height", "LeafArea"]
    for i, name in enumerate(metrics):
        print(f"{name}: MSE={mse[i]:.4f}, R²={r2[i]:.4f}")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = LettuceDataset(
        "GroundTruth_SendJuly13.json", "path_to_images", transform=transform)
    _, _, test_set = torch.utils.data.random_split(dataset, [int(
        0.7*len(dataset)), int(0.15*len(dataset)), len(dataset) - int(0.85*len(dataset))])
    test_loader = DataLoader(test_set, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBDepthNet()
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)

    evaluate(model, test_loader, device)
