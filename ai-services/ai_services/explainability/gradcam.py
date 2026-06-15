"""Grad-CAM for CNN-based embedding backbones (EfficientNet, Siamese/ResNet18).

Standard Grad-CAM (Selvaraju et al., 2017), adapted for embedding models: instead of
backpropagating a class logit, we backpropagate the L2 norm of the output embedding,
highlighting the regions that most influence *where* the image lands in embedding
space.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes a Grad-CAM heatmap for `target_layer` within `model`.

    Usage::

        cam = GradCAM(model, model.features[-1])
        try:
            heatmap = cam(input_tensor, lambda x: model(x))
        finally:
            cam.remove()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, embed_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Run `embed_fn(input_tensor)`, backpropagate its norm, and return a
        `(H, W)` numpy heatmap normalized to `[0, 1]` at the input's spatial size."""
        self.model.zero_grad(set_to_none=True)
        embedding = embed_fn(input_tensor)
        score = embedding.norm()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire - is `target_layer` part of the forward pass?")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value
        return cam

    def remove(self) -> None:
        """Detach the forward/backward hooks."""
        self._forward_handle.remove()
        self._backward_handle.remove()
