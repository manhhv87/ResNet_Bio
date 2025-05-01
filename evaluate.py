import os
import torch
import numpy as np
from models.cnn_model import LettuceNet
from utils.dataset import LettuceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

# Tạo thư mục kết quả nếu chưa có
os.makedirs("results", exist_ok=True)

# Load mô hình
model = LettuceNet()
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location="cpu"))
model.eval()

# Load dữ liệu test
dataset = LettuceDataset("data/test/test.json", "data/test")
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Tạo container
preds = {
    "fresh": [], "diameter": [], "leaf_out1": [],
    "height": [], "dry": [], "leaf_out4": []
}
gts = {
    "fresh": [], "diameter": [], "leaf": [],
    "height": [], "dry": []
}

# Dự đoán
with torch.no_grad():
    for rgb, depth, y1, y2, y3, y4 in test_loader:
        o1, o2, o3, o4 = model(rgb, depth)

        preds["fresh"].append(o1[:, 0].numpy())
        preds["diameter"].append(o1[:, 1].numpy())
        preds["leaf_out1"].append(o1[:, 2].numpy())

        preds["height"].append(o2.squeeze().numpy())
        preds["dry"].append(o3.squeeze().numpy())
        preds["leaf_out4"].append(o4.squeeze().numpy())

        gts["fresh"].append(y1[:, 0].numpy())
        gts["diameter"].append(y1[:, 1].numpy())
        gts["leaf"].append(y1[:, 2].numpy())
        gts["height"].append(y2.squeeze().numpy())
        gts["dry"].append(y3.squeeze().numpy())

# Ghép dữ liệu
for k in preds:
    preds[k] = np.concatenate(preds[k])
for k in gts:
    gts[k] = np.concatenate(gts[k])

# Hàm đánh giá
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.mean(y_true) + 1e-8)
    nmse = mean_squared_error(y_true, y_pred) / (np.var(y_true) + 1e-8)
    return {"r2_score": r2, "nrmse": nrmse, "nmse": nmse}

# Tính metrics
metrics = {
    "fresh_weight": compute_metrics(gts["fresh"], preds["fresh"]),
    "diameter": compute_metrics(gts["diameter"], preds["diameter"]),
    "leaf_area_out1": compute_metrics(gts["leaf"], preds["leaf_out1"]),
    "height": compute_metrics(gts["height"], preds["height"]),
    "dry_weight": compute_metrics(gts["dry"], preds["dry"]),
    "leaf_area_out4": compute_metrics(gts["leaf"], preds["leaf_out4"]),
}

# In kết quả
print("\n=== Evaluation Results ===")
for name, m in metrics.items():
    print(f"{name:<20} | R2 = {m['r2_score']:.4f}, NRMSE = {m['nrmse']:.4f}, NMSE = {m['nmse']:.4f}")

# Lưu kết quả
np.save("results/preds_eval.npy", preds)
np.save("results/gts_eval.npy", gts)
np.save("results/metrics_eval.npy", metrics)
