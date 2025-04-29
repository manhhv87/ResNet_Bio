import torch
from PIL import Image
from torchvision import transforms
from models.model import RGBDepthNet


def load_model(weights_path, device):
    model = RGBDepthNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(rgb_path, depth_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    rgb = Image.open(rgb_path).convert("RGB")
    depth = Image.open(depth_path).convert("L")

    rgb = transform(rgb).unsqueeze(0)
    depth = transform(depth).unsqueeze(0)
    return rgb, depth


def predict(model, rgb_tensor, depth_tensor, device):
    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)
    with torch.no_grad():
        out1, out2, out3, out4 = model(rgb_tensor, depth_tensor)

    results = {
        "FreshWeight": out1[0, 0].item(),
        "DryWeight": out1[0, 1].item(),  # out3 alternative
        "Diameter": out1[0, 2].item(),
        "Height": out2[0, 0].item(),
        "LeafArea": out4[0, 0].item()
    }
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("best_model.pt", device)

    rgb_path = "path_to_single_RGB.png"
    depth_path = "path_to_single_Depth.png"

    rgb, depth = preprocess_image(rgb_path, depth_path)
    prediction = predict(model, rgb, depth, device)

    for key, value in prediction.items():
        print(f"{key}: {value:.2f}")
