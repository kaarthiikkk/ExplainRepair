"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Explainer.
Uses PyTorch forward and backward hooks to extract activations and gradients
from convolutional layers, computing target class saliency maps.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from explainers.base_explainer import BaseExplainer
from utils.metrics import normalize_map


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM implementation with robust hook lifecycle management.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str = "layer4",
        name: str = "Grad-CAM"
    ):
        super().__init__(model=model, name=name)
        self.target_layer_name = target_layer_name
        self.target_layer = self._find_target_layer(model, target_layer_name)
        
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.metadata = {
            "target_layer": target_layer_name,
            "method": "Grad-CAM",
            "reference": "Selvaraju et al., ICCV 2017"
        }

    def _find_target_layer(self, model: nn.Module, layer_name: str) -> nn.Module:
        """Locates submodule in the model by name or hierarchical dot path."""
        # Check in backbone if wrapped
        target_module = model
        if hasattr(model, "backbone"):
            target_module = model.backbone

        for name, module in target_module.named_modules():
            if name == layer_name:
                return module
                
        # Fallback: look for common layers
        for name, module in target_module.named_modules():
            if layer_name in name:
                return module
                
        # Last resort: find last Conv2d layer
        conv_layers = [m for m in target_module.modules() if isinstance(m, nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]
            
        raise ValueError(f"Could not find layer '{layer_name}' in model.")

    def _forward_hook(self, module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        self.activations = out

    def _backward_hook(self, module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        self.gradients = grad_out[0]

    def attribute(
        self,
        tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM attribution map.
        tensor: [1, 3, H, W]
        """
        self.model.eval()
        x = tensor.clone().detach().requires_grad_(True)
        device = next(self.model.parameters()).device
        x = x.to(device)

        # Register hooks
        handle_fwd = self.target_layer.register_forward_hook(self._forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            handle_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)
        else:
            handle_bwd = self.target_layer.register_backward_hook(self._backward_hook)

        try:
            logits = self.model(x)
            
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Failed to capture activations or gradients during backward pass.")

            # Calculate alpha weights via Global Average Pooling of gradients: shape [1, C, 1, 1]
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
            
            # Weighted combination of activation maps: [1, 1, H_act, W_act]
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            
            # Apply ReLU (only positive influence features)
            cam = F.relu(cam)
            
            # Upsample to input image size [1, 1, H, W]
            h_in, w_in = tensor.shape[2], tensor.shape[3]
            cam_upsampled = F.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)
            
            # Convert to 2D numpy array [H, W]
            cam_np = cam_upsampled.squeeze().detach().cpu().numpy()
            
            # Normalize to [0, 1]
            cam_normalized = normalize_map(cam_np, method="minmax")
            return cam_normalized

        finally:
            # Always cleanly remove hooks
            handle_fwd.remove()
            handle_bwd.remove()
            self.activations = None
            self.gradients = None


class SmoothGradCAMExplainer(BaseExplainer):
    """
    Smoothed Grad-CAM: Averages Grad-CAM maps over n noise-perturbed samples.
    Mitigates gradient noise and enhances explanation stability.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str = "layer4",
        num_samples: int = 10,
        noise_level: float = 0.15,
        name: str = "Smoothed Grad-CAM"
    ):
        super().__init__(model=model, name=name)
        self.base_explainer = GradCAMExplainer(
            model=model, target_layer_name=target_layer_name, name=name
        )
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.metadata = {
            "target_layer": target_layer_name,
            "method": "SmoothGrad-CAM",
            "samples": num_samples,
            "noise_level": noise_level,
            "reference": "Smilkov et al., 2017 (adapted for CAM)"
        }

    def attribute(
        self,
        tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        accumulated_cam = np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float64)
        
        for _ in range(self.num_samples):
            # Add small Gaussian noise to input
            noise = torch.randn_like(tensor) * self.noise_level
            noisy_input = tensor + noise
            cam = self.base_explainer.attribute(noisy_input, target_class=target_class)
            accumulated_cam += cam

        smoothed = accumulated_cam / float(self.num_samples)
        return normalize_map(smoothed, method="minmax")
