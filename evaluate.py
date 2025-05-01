import os
import torch
import numpy as np
from models.cnn_model import LettuceCNN
from utils.dataset import LettuceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

model = LettuceCNN()
model.load_state_dict(torch.load('checkpoints/epoch_99.pt'))
model.eval()

# Load test data
dataset = LettuceDataset('data/GroundTruth_All_388_Images.json', 'data/test')
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Containers
preds_out1, preds_out2, preds_out3, preds_out4 = [], [], [], []
gt_out1, gt_out2, gt_out3, gt_out4 = [], [], [], []

with torch.no_grad():
    for rgb, depth, y1, y2, y3, y4 in test_loader:
        o1, o2, o3, o4 = model(rgb, depth)
        preds_out1.append(o1.numpy())
        preds_out2.append(o2.numpy())
        preds_out3.append(o3.numpy())
        preds_out4.append(o4.numpy())

        gt_out1.append(y1.numpy())
        gt_out2.append(y2.numpy())
        gt_out3.append(y3.numpy())
        gt_out4.append(y4.numpy())

# Convert to arrays
preds_out1 = np.concatenate(preds_out1, axis=0)
preds_out2 = np.concatenate(preds_out2, axis=0)
preds_out3 = np.concatenate(preds_out3, axis=0)
preds_out4 = np.concatenate(preds_out4, axis=0)

gt_out1 = np.concatenate(gt_out1, axis=0)
gt_out2 = np.concatenate(gt_out2, axis=0)
gt_out3 = np.concatenate(gt_out3, axis=0)
gt_out4 = np.concatenate(gt_out4, axis=0)

# Compute metrics
def compute_metrics(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)
    nmse = mean_squared_error(y_true, y_pred) / np.var(y_true)
    print(f"{name} - R2: {r2:.4f}, NRMSE: {nrmse:.4f}, NMSE: {nmse:.4f}")
    return {'r2': r2, 'nrmse': nrmse, 'nmse': nmse}

metrics = {
    'output1': compute_metrics(gt_out1, preds_out1, "Output1 (fresh, diameter, leaf)"),
    'output2': compute_metrics(gt_out2, preds_out2, "Output2 (height)"),
    'output3': compute_metrics(gt_out3, preds_out3, "Output3 (dry weight)"),
    'output4': compute_metrics(gt_out4, preds_out4, "Output4 (leaf area)"),
}

# Ensure result directory exists
os.makedirs("results", exist_ok=True)

# Save predictions and metrics
np.save("results/pred_output1.npy", preds_out1)
np.save("results/pred_output2.npy", preds_out2)
np.save("results/pred_output3.npy", preds_out3)
np.save("results/pred_output4.npy", preds_out4)

np.save("results/gt_output1.npy", gt_out1)
np.save("results/gt_output2.npy", gt_out2)
np.save("results/gt_output3.npy", gt_out3)
np.save("results/gt_output4.npy", gt_out4)

np.save("results/metrics.npy", metrics)