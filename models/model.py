import torch
import torch.nn as nn
import torchvision.models as models


class RGBDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.ReLU()
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU()
        )

        self.rgb_resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Modify the first conv layer to accept 32 channels instead of 3
        self.rgb_resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.depth_resnet = models.resnet50(weights=None)
        self.depth_resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.rgb_resnet.fc = nn.Identity()
        self.depth_resnet.fc = nn.Identity()

        self.rgb_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(2048),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(2048, 256),
            nn.Dropout(),
            nn.Linear(256, 4)  # FW, DW, Diameter, ?
        )

        self.depth_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(2048),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(2048, 256),
            nn.Dropout(),
            nn.Linear(256, 1)  # Height
        )

        self.shared_fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(5, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout()
        )

        self.dry_weight_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1)
        )
        self.leaf_area_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 1)
        )

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_conv(rgb)
        depth_feat = self.depth_conv(depth)

        rgb_feat = self.rgb_resnet(rgb_feat)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), rgb_feat.size(1), 1, 1)

        depth_feat = self.depth_resnet(depth_feat)
        depth_feat = depth_feat.view(
            depth_feat.size(0), rgb_feat.size(1), 1, 1)

        out1 = self.rgb_regressor(rgb_feat)
        out2 = self.depth_regressor(depth_feat)

        combined = torch.cat([out1, out2], dim=1)
        shared = self.shared_fc(combined)

        out3 = self.dry_weight_head(shared)
        out4 = self.leaf_area_head(shared)

        return out1, out2, out3, out4
