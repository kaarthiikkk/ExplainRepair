"""
Unit tests for Adaptive Repair Engine: non-invasive guarantee, honest reporting, determinism.
"""

import copy
import pytest
import torch
import torch.nn as nn

from config import AppConfig, RepairConfig, ReliabilityConfig
from explainers.gradcam_explainer import GradCAMExplainer
from models.predictor import ModelPredictor
from reliability.reliability_evaluator import ReliabilityEvaluator
from repair.repair_engine import AdaptiveRepairEngine
from repair.validation import compute_model_hash, RepairValidator


class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.layer3 = nn.Conv2d(16, 16, 3, padding=1)
        self.layer4 = nn.Conv2d(16, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.layer3(h))
        h = torch.relu(self.layer4(h))
        return self.fc(h.view(h.size(0), -1))


@pytest.fixture
def predictor_setup():
    torch.manual_seed(42)
    model = MiniResNet()
    model.eval()
    pred = ModelPredictor.__new__(ModelPredictor)
    pred.model = model
    pred.device = torch.device("cpu")
    pred.model_status = {"is_trained": True, "source": "test_mock"}
    return pred


@pytest.fixture
def test_tensor():
    torch.manual_seed(99)
    return torch.randn(1, 3, 32, 32)


def test_repair_preserves_model_weights_and_predictions(predictor_setup, test_tensor):
    pred = predictor_setup
    hash_start = compute_model_hash(pred.model)
    
    with torch.no_grad():
        orig_logits = pred.model(test_tensor)
        orig_class = int(torch.argmax(orig_logits, dim=-1).item())
        orig_conf = float(torch.softmax(orig_logits, dim=-1)[0, orig_class].item())

    # Build repair engine
    cfg = AppConfig()
    cfg.repair.candidate_pool = [
        {"method": "gradcam", "layer": "layer3", "label": "Grad-CAM (Layer 3)"},
        {"method": "integrated_gradients", "baseline": "black", "steps": "10", "label": "IG (Black)"}
    ]
    evaluator = ReliabilityEvaluator(cfg.reliability)
    repair_engine = AdaptiveRepairEngine(predictor=pred, evaluator=evaluator, config=cfg)

    initial_explainer = GradCAMExplainer(pred.model, target_layer_name="layer4", name="Grad-CAM (Layer 4)")
    result = repair_engine.repair(initial_explainer, test_tensor, target_class=orig_class)

    # 1. Weights must be completely unaltered
    hash_end = compute_model_hash(pred.model)
    assert hash_start == hash_end, "Model weights were modified during repair!"

    # 2. Prediction must be completely unaltered
    with torch.no_grad():
        after_logits = pred.model(test_tensor)
        after_class = int(torch.argmax(after_logits, dim=-1).item())
        after_conf = float(torch.softmax(after_logits, dim=-1)[0, after_class].item())

    assert orig_class == after_class
    assert abs(orig_conf - after_conf) < 1e-5
    assert result.validation_status["weights_preserved"] is True
    assert result.validation_status["prediction_preserved"] is True


def test_repair_honesty_on_unsuccessful_candidate(predictor_setup, test_tensor):
    """If no candidate improves, the engine must honestly declare UNSUCCESSFUL."""
    pred = predictor_setup
    
    cfg = AppConfig()
    # Provide candidate that will not meet improvement threshold
    cfg.repair.min_improvement_delta = 0.99  # Impossible threshold
    cfg.repair.candidate_pool = [
        {"method": "gradcam", "layer": "layer3", "label": "Grad-CAM (Layer 3)"}
    ]
    
    repair_engine = AdaptiveRepairEngine(predictor=pred, config=cfg)
    initial_explainer = GradCAMExplainer(pred.model, target_layer_name="layer4")
    
    result = repair_engine.repair(initial_explainer, test_tensor, target_class=0)
    
    # Must report UNSUCCESSFUL or NO_REPAIR_NEEDED (never falsely reported SUCCESS)
    assert result.outcome in ["UNSUCCESSFUL", "NO_REPAIR_NEEDED"]
    assert result.outcome != "SUCCESS"


def test_repair_determinism(predictor_setup, test_tensor):
    """Running repair twice with identical seed and inputs produces identical outputs."""
    pred = predictor_setup
    cfg = AppConfig()
    cfg.repair.candidate_pool = [
        {"method": "gradcam", "layer": "layer3", "label": "Grad-CAM (Layer 3)"}
    ]
    
    engine1 = AdaptiveRepairEngine(predictor=pred, config=cfg)
    engine2 = AdaptiveRepairEngine(predictor=pred, config=cfg)
    
    init1 = GradCAMExplainer(pred.model, target_layer_name="layer4")
    init2 = GradCAMExplainer(pred.model, target_layer_name="layer4")
    
    res1 = engine1.repair(init1, test_tensor, target_class=0)
    res2 = engine2.repair(init2, test_tensor, target_class=0)
    
    assert res1.outcome == res2.outcome
    assert res1.selected_explainer_name == res2.selected_explainer_name
    assert res1.delta_composite == pytest.approx(res2.delta_composite, abs=1e-3)


def test_repair_validator_cross_device(predictor_setup):
    """RepairValidator handles CPU tensor input even if model is on another device."""
    pred = predictor_setup
    cpu_tensor = torch.randn(1, 3, 32, 32)  # explicitly on CPU
    snapshot = RepairValidator.capture_snapshot(pred.model, cpu_tensor)
    
    assert snapshot.parameter_hash is not None
    assert 0 <= snapshot.prediction_class < 10
    assert 0.0 <= snapshot.confidence <= 1.0

    integrity = RepairValidator.verify_integrity(snapshot, pred.model, cpu_tensor)
    assert integrity["weights_preserved"] is True
    assert integrity["prediction_preserved"] is True


def test_repair_validation_outcome_classification(predictor_setup, test_tensor):
    """RepairResult must assign a valid validation_outcome and attach a proxy disclaimer."""
    pred = predictor_setup
    cfg = AppConfig()
    cfg.repair.candidate_pool = [
        {"method": "gradcam", "layer": "layer3", "label": "Grad-CAM (Layer 3)"}
    ]
    engine = AdaptiveRepairEngine(predictor=pred, config=cfg)
    init = GradCAMExplainer(pred.model, target_layer_name="layer4")
    res = engine.repair(init, test_tensor, target_class=0)

    assert res.validation_outcome in [
        "VALIDATED_IMPROVEMENT",
        "SELECTION_IMPROVED_ONLY",
        "PARTIAL_VALIDATION",
        "VALIDATION_FAILED",
        "NO_REPAIR_NEEDED"
    ]
    assert "proxy" in res.validation_disclaimer.lower() or "disclaimer" in res.validation_disclaimer.lower()

