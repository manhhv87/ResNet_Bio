import torch
import torch.nn as nn
import torchvision.models as models


class RegressorBlock(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(RegressorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, out_features)
        )

    def forward(self, x):
        return self.block(x)


class LettuceNet(nn.Module):
    def __init__(self):
        super(LettuceNet, self).__init__()

        # RGB branch
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.ReLU()
        )
        resnet_rgb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.rgb_backbone = nn.Sequential(*list(resnet_rgb.children())[:-2])  # exclude avgpool & fc
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rgb_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)  # includes: fresh, dry, diameter, leaf area
        )

        # Depth branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU()
        )
        resnet_depth = models.resnet50(weights=None)
        self.depth_backbone = nn.Sequential(*list(resnet_depth.children())[:-2])
        self.depth_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.depth_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # height
        )

        # Merge from both branches
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 + 256, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.output1 = nn.Linear(2048, 3)           # fresh, diameter, leaf
        self.output3 = RegressorBlock(2048, 1)      # dry weight
        self.output4 = RegressorBlock(2048, 1)      # leaf area

    def forward(self, rgb, depth):
        # RGB branch
        x_rgb = self.rgb_conv(rgb)
        x_rgb = self.rgb_backbone(x_rgb)
        x_rgb = self.rgb_pool(x_rgb)
        x_rgb = torch.flatten(x_rgb, 1)
        x_rgb_feat = self.rgb_fc[:-1](x_rgb)        # extract 256-dim

        # Depth branch
        x_depth = self.depth_conv(depth)
        x_depth = x_depth.repeat(1, 3, 1, 1)        # fix: convert 1 channel to 3
        x_depth = self.depth_backbone(x_depth)
        x_depth = self.depth_pool(x_depth)
        x_depth = torch.flatten(x_depth, 1)
        x_depth_feat = self.depth_fc[:-1](x_depth)  # [B, 256]

        # Merged features
        merged = torch.cat([x_rgb_feat, x_depth_feat], dim=1)  # [B, 512]
        shared_out = self.shared_fc(merged)

        # Outputs
        out1 = self.output1(shared_out)         # fresh, diameter, leaf area
        out2 = self.depth_fc[-1](x_depth_feat)  # height
        out3 = self.output3(shared_out)         # dry weight
        out4 = self.output4(shared_out)         # leaf area (again)

        return out1, out2, out3, out4
