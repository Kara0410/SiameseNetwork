import torch
import torch.nn as nn

from ai_services.explainability.gradcam import GradCAM


class TinyCNN(nn.Module):
    """Small CNN used to exercise Grad-CAM without downloading pretrained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_gradcam_returns_normalized_heatmap_matching_input_size():
    model = TinyCNN()
    target_layer = model.features[2]  # second Conv2d
    cam = GradCAM(model, target_layer)

    tensor = torch.randn(1, 3, 32, 32)
    try:
        heatmap = cam(tensor, model)
    finally:
        cam.remove()

    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_raises_if_target_layer_not_in_forward_path():
    model = TinyCNN()
    detached_layer = nn.Conv2d(3, 4, kernel_size=3)  # not part of `model`'s forward pass
    cam = GradCAM(model, detached_layer)

    tensor = torch.randn(1, 3, 32, 32)
    try:
        try:
            cam(tensor, model)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError when hooks never fire")
    finally:
        cam.remove()
