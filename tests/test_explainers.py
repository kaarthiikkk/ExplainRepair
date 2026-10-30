"""
Unit tests for Grad-CAM and Integrated Gradients explainers.
Uses small synthetic inputs and models for instant, deterministic testing.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from explainers.gradcam_explainer import GradCAMExplainer, SmoothGradCAMExplainer
from explainers.integrated_gradients_explainer import IntegratedGradientsExplainer
from models.model import ResNet18CIFAR10


@pytest.fixture
def dummy_model():
    """Lightweight dummy CNN for fast testing."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.fc = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            feats = self.layer4(x)
            out = self.fc(feats.view(feats.size(0), -1))
            return out

    model = SimpleCNN()
    model.eval()
    return model


@pytest.fixture
def dummy_tensor():
    # Shape [1, 3, 8, 8]
    torch.manual_seed(42)
    return torch.randn(1, 3, 8, 8)


def test_gradcam_output_shape_and_normalization(dummy_model, dummy_tensor):
    explainer = GradCAMExplainer(dummy_model, target_layer_name="layer4")
    attr = explainer.attribute(dummy_tensor, target_class=2)
    
    # Check shape
    assert attr.shape == (8, 8), f"Expected shape (8, 8), got {attr.shape}"
    # Check normalization in [0, 1]
    assert np.min(attr) >= 0.0, "Attribution has negative values"
    assert np.max(attr) <= 1.0, "Attribution has values > 1.0"
    # Check hook cleanup
    assert explainer.activations is None
    assert explainer.gradients is None


def test_smooth_gradcam(dummy_model, dummy_tensor):
    explainer = SmoothGradCAMExplainer(dummy_model, target_layer_name="layer4", num_samples=3)
    attr = explainer.attribute(dummy_tensor, target_class=0)
    assert attr.shape == (8, 8)
    assert 0.0 <= np.min(attr) <= np.max(attr) <= 1.0


def test_integrated_gradients_output_shape(dummy_model, dummy_tensor):
    explainer = IntegratedGradientsExplainer(dummy_model, n_steps=10, baseline_type="black")
    attr = explainer.attribute(dummy_tensor, target_class=1)
    
    assert attr.shape == (8, 8), f"Expected shape (8, 8), got {attr.shape}"
    assert np.min(attr) >= 0.0
    assert np.max(attr) <= 1.0
    assert "convergence_delta" in explainer.get_metadata()


def test_integrated_gradients_baselines(dummy_model, dummy_tensor):
    for b_type in ["black", "white", "random"]:
        expl = IntegratedGradientsExplainer(dummy_model, n_steps=5, baseline_type=b_type)
        attr = expl.attribute(dummy_tensor, target_class=0)
        assert attr.shape == (8, 8)
        assert np.isfinite(attr).all()
