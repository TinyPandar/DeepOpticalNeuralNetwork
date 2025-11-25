import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision.models as tvm
except Exception:
    tvm = None


class SimpleCNN(nn.Module):
    """A lightweight CNN baseline that maps [B,3,H_in,W_in] to [B,H_out,W_out].

    It downsamples spatially using strided convs and then upsamples or pools to match
    the requested output size. The final output is a single-channel heatmap.
    """

    def __init__(self, input_hw: tuple[int, int], output_hw: tuple[int, int]):
        super().__init__()
        h_in, w_in = int(input_hw[0]), int(input_hw[1])
        h_out, w_out = int(output_hw[0]), int(output_hw[1])
        if h_in <= 0 or w_in <= 0 or h_out <= 0 or w_out <= 0:
            raise ValueError("Invalid spatial sizes")
        self.h_in = h_in
        self.w_in = w_in
        self.h_out = h_out
        self.w_out = w_out

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected input shape [B,3,H,W]")
        feat = self.backbone(x)
        heat = self.head(feat)
        heat = F.softplus(heat)
        heat = F.interpolate(heat, size=(self.h_out, self.w_out), mode="bilinear", align_corners=False)
        return heat[:, 0]


class ResNetHeatmap(nn.Module):
    """ResNet backbone producing a single-channel heatmap resized to output_hw.

    Supports resnet18/34/50 via the `variant` argument. The classification head is
    removed and replaced by a small conv head to produce a heatmap.
    """

    def __init__(self, input_hw: tuple[int, int], output_hw: tuple[int, int], variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        h_in, w_in = int(input_hw[0]), int(input_hw[1])
        h_out, w_out = int(output_hw[0]), int(output_hw[1])
        if h_in <= 0 or w_in <= 0 or h_out <= 0 or w_out <= 0:
            raise ValueError("Invalid spatial sizes")
        self.h_in = h_in
        self.w_in = w_in
        self.h_out = h_out
        self.w_out = w_out

        if tvm is None:
            raise ImportError("torchvision is required for ResNetHeatmap but is not available")

        variant = str(variant).lower()
        # Instantiate backbone without the classification head
        if variant == "resnet18":
            backbone = tvm.resnet18(weights=(tvm.ResNet18_Weights.DEFAULT if pretrained else None))
            feat_channels = 512
        elif variant == "resnet34":
            backbone = tvm.resnet34(weights=(tvm.ResNet34_Weights.DEFAULT if pretrained else None))
            feat_channels = 512
        elif variant == "resnet50":
            backbone = tvm.resnet50(weights=(tvm.ResNet50_Weights.DEFAULT if pretrained else None))
            feat_channels = 2048
        else:
            raise ValueError("variant must be one of {'resnet18','resnet34','resnet50'}")

        # Keep only convolutional feature extractor layers
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Heatmap head
        self.head = nn.Sequential(
            nn.Conv2d(feat_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected input shape [B,3,H,W]")
        # ResNet accepts arbitrary H,W (multiples of 32 preferred), so pass-through
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        heat = self.head(x)
        heat = F.softplus(heat)
        heat = F.interpolate(heat, size=(self.h_out, self.w_out), mode="bilinear", align_corners=False)
        return heat[:, 0]


__all__ = ["SimpleCNN", "ResNetHeatmap"]


