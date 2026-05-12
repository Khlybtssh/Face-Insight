"""
Facial Expression Recognition Model (PyTorch)
Pre-trained MobileNetV2 + CBAM Attention

Architecture:
    Input (224x224x3) -> MobileNetV2 backbone (pretrained) -> CBAM -> Classifier -> 7 Emotions

Using a pretrained backbone gives much better features than training from scratch,
and CBAM focuses the model on the most important facial regions.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# =============================================================================
# CBAM Components
# =============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention: learns which feature channels (filters) are most important.
    Uses Global Avg Pool + Global Max Pool -> shared MLP -> sigmoid gating.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])                          # (B, C)
        max_pool = x.amax(dim=[2, 3])                          # (B, C)
        attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))  # (B, C)
        return x * attention.unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    """
    Spatial Attention: learns which spatial locations matter most.
    Uses channel-wise Avg+Max pool -> Conv(7x7) -> sigmoid gating.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        max_pool = x.amax(dim=1, keepdim=True)     # (B, 1, H, W)
        attention = self.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module: Channel Attention + Spatial Attention."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# Full Model: MobileNetV2 + CBAM
# =============================================================================

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
    'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}


class ExpressionModel(nn.Module):
    """
    MobileNetV2 (pretrained on ImageNet) + CBAM Attention + Classifier.

    Pipeline:
        Input (3, 224, 224) -> MobileNetV2 features (1280, 7, 7)
        -> CBAM -> AdaptiveAvgPool -> Dense(256) -> Dropout -> Dense(7)
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # Load pretrained MobileNetV2 backbone
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Use only the feature extractor (drop the classifier)
        self.backbone = mobilenet.features       # Output: (B, 1280, 7, 7)

        # Freeze early layers, fine-tune later layers
        # Freeze first 14 out of 19 blocks for stable training
        for i, layer in enumerate(self.backbone):
            if i < 14:
                for param in layer.parameters():
                    param.requires_grad = False

        # CBAM Attention on backbone output
        self.cbam = CBAM(channels=1280, reduction=16, kernel_size=7)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # (B, 1280, 1, 1)
            nn.Flatten(),                           # (B, 1280)
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)      # (B, 1280, 7, 7)
        attended = self.cbam(features)    # (B, 1280, 7, 7) - attention applied
        output = self.classifier(attended)
        return output


def build_expression_model(num_classes=7, pretrained=True):
    """Build and return the expression recognition model."""
    return ExpressionModel(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    model = build_expression_model()
    print(model)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")
    print(f"Emotion labels:   {EMOTION_LABELS}")

    # Test forward pass
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape:     {out.shape}")
