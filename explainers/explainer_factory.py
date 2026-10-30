"""
Factory pattern for creating explainer instances dynamically.
"""

from typing import Dict, Any, Optional
import torch.nn as nn

from explainers.base_explainer import BaseExplainer
from explainers.gradcam_explainer import GradCAMExplainer, SmoothGradCAMExplainer
from explainers.integrated_gradients_explainer import IntegratedGradientsExplainer


class ExplainerFactory:
    """
    Factory creating explainers based on configuration parameters.
    """
    @staticmethod
    def create_explainer(
        model: nn.Module,
        config: Dict[str, Any]
    ) -> BaseExplainer:
        raw_method = config.get("method", "gradcam")
        method = raw_method.lower().replace("-", "").replace("_", "").replace(" ", "")
        label = config.get("label", config.get("explainer_name", raw_method))
        layer = config.get("target_layer") or config.get("layer", "layer4")

        if method == "gradcam":
            return GradCAMExplainer(
                model=model,
                target_layer_name=layer,
                name=label or f"Grad-CAM ({layer})"
            )
        elif method in ["smoothgradcam", "smoothgrad"]:
            samples = int(config.get("samples", 10))
            return SmoothGradCAMExplainer(
                model=model,
                target_layer_name=layer,
                num_samples=samples,
                name=label or f"SmoothGrad-CAM ({layer})"
            )
        elif method in ["integratedgradients", "ig"]:
            baseline = config.get("baseline", config.get("baseline_type", "black"))
            steps = int(config.get("steps", config.get("n_steps", 50)))
            return IntegratedGradientsExplainer(
                model=model,
                n_steps=steps,
                baseline_type=baseline,
                name=label or f"Integrated Gradients ({baseline}, {steps} steps)"
            )
        else:
            raise ValueError(f"Unknown explainer method: {raw_method}")
