"""
Integrated Gradients Explainer with configurable baselines, integration steps,
and convergence diagnostics (Sundararajan et al., ICML 2017).
Uses Captum when available, with a resilient native Riemann-sum fallback.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

from explainers.base_explainer import BaseExplainer
from utils.image_utils import denormalize_tensor, preprocess_image
from utils.metrics import normalize_map

try:
    from captum.attr import IntegratedGradients as CaptumIG
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients attribution method.
    Approximates the path integral of gradients along the straight line
    between a chosen baseline and the input image.
    """
    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        baseline_type: str = "black",
        name: str = "Integrated Gradients"
    ):
        super().__init__(model=model, name=name)
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        
        if HAS_CAPTUM:
            self.captum_ig = CaptumIG(self.model)
        else:
            self.captum_ig = None

        self.metadata = {
            "method": "Integrated Gradients",
            "n_steps": n_steps,
            "baseline_type": baseline_type,
            "engine": "Captum" if HAS_CAPTUM else "Native_Riemann_Fallback",
            "reference": "Sundararajan et al., ICML 2017"
        }

    def _generate_baseline(self, tensor: torch.Tensor) -> torch.Tensor:
        """Constructs the reference baseline tensor matching input shape [1, 3, H, W]."""
        device = tensor.device
        if self.baseline_type == "black":
            # For normalized CIFAR-10, black image (RGB [0,0,0]) normalized
            rgb_black = np.zeros((tensor.shape[2], tensor.shape[3], 3), dtype=np.uint8)
            t_base, _ = preprocess_image(Image.fromarray(rgb_black), target_size=(tensor.shape[2], tensor.shape[3]))
            return t_base.to(device)
            
        elif self.baseline_type == "white":
            rgb_white = np.ones((tensor.shape[2], tensor.shape[3], 3), dtype=np.uint8) * 255
            t_base, _ = preprocess_image(Image.fromarray(rgb_white), target_size=(tensor.shape[2], tensor.shape[3]))
            return t_base.to(device)
            
        elif self.baseline_type == "blur":
            rgb = denormalize_tensor(tensor)
            pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=3.0))
            t_base, _ = preprocess_image(blurred_pil, target_size=(tensor.shape[2], tensor.shape[3]))
            return t_base.to(device)
            
        elif self.baseline_type == "random":
            # Small random uniform noise
            return torch.randn_like(tensor) * 0.1
            
        else:
            # Default zero tensor
            return torch.zeros_like(tensor)

    def _native_integrated_gradients(
        self,
        tensor: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Native implementation of path integral using Gauss-Legendre / Riemann sum:
        IG_i(x) = (x_i - x'_i) * (1/m) sum_{k=1}^m grad_F(x' + (k/m)*(x - x'))
        """
        device = tensor.device
        alphas = torch.linspace(0.0, 1.0, steps=self.n_steps, device=device)
        
        # Scale inputs along straight line path
        diff = tensor - baseline
        interpolated = baseline + alphas.view(-1, 1, 1, 1) * diff  # [n_steps, 3, H, W]
        
        grads = []
        # Batch evaluation for memory efficiency
        batch_size = 16
        for i in range(0, self.n_steps, batch_size):
            batch_inputs = interpolated[i:i+batch_size].clone().detach().requires_grad_(True)
            logits = self.model(batch_inputs)
            scores = logits[:, target_class]
            
            self.model.zero_grad()
            batch_grads = torch.autograd.grad(
                outputs=scores,
                inputs=batch_inputs,
                grad_outputs=torch.ones_like(scores),
                create_graph=False,
                retain_graph=False
            )[0]
            grads.append(batch_grads)

        all_grads = torch.cat(grads, dim=0)  # [n_steps, 3, H, W]
        avg_grads = torch.mean(all_grads, dim=0, keepdim=True)  # [1, 3, H, W]
        attributions = diff * avg_grads  # [1, 3, H, W]
        
        # Completeness delta: sum(attr) vs (F(x) - F(baseline))
        with torch.no_grad():
            f_x = self.model(tensor)[0, target_class].item()
            f_base = self.model(baseline)[0, target_class].item()
            sum_attr = attributions.sum().item()
            delta = abs((f_x - f_base) - sum_attr)

        return attributions, delta

    def attribute(
        self,
        tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution map.
        tensor: [1, 3, H, W]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = tensor.clone().detach().to(device)
        baseline = self._generate_baseline(x)

        if target_class is None:
            with torch.no_grad():
                target_class = int(torch.argmax(self.model(x), dim=1).item())

        delta = 0.0
        if self.captum_ig is not None:
            try:
                # Captum execution
                x_req = x.clone().requires_grad_(True)
                attributions, delta_tensor = self.captum_ig.attribute(
                    inputs=x_req,
                    baselines=baseline,
                    target=target_class,
                    n_steps=self.n_steps,
                    return_convergence_delta=True
                )
                delta = float(torch.mean(torch.abs(delta_tensor)).item())
            except Exception:
                # Fallback to native
                attributions, delta = self._native_integrated_gradients(x, baseline, target_class)
        else:
            attributions, delta = self._native_integrated_gradients(x, baseline, target_class)

        self.metadata["convergence_delta"] = round(delta, 5)

        # Convert [1, 3, H, W] to 2D spatial heatmap
        # Take L2 norm across color channels or absolute mean
        attr_np = attributions.squeeze(0).detach().cpu().numpy()  # [3, H, W]
        spatial_map = np.linalg.norm(attr_np, axis=0)  # [H, W]

        # Normalize to [0, 1]
        return normalize_map(spatial_map, method="minmax")
