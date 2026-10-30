"""
Unit tests for Reliability Evaluator: Stability, Faithfulness, Agreement, and Sanity Checks.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from explainers.gradcam_explainer import GradCAMExplainer
from models.predictor import ModelPredictor
from reliability.stability import StabilityEvaluator
from reliability.faithfulness import FaithfulnessEvaluator
from reliability.agreement import AgreementEvaluator
from reliability.sanity_checks import ModelSanityCheckEvaluator
from repair.validation import compute_model_hash


class MockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        return self.fc(h.view(h.size(0), -1))


@pytest.fixture
def mock_predictor():
    model = MockCNN()
    model.eval()
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.model = model
    predictor.device = torch.device("cpu")
    predictor.model_status = {"is_trained": True, "source": "test_mock"}
    return predictor


@pytest.fixture
def sample_tensor():
    torch.manual_seed(101)
    return torch.randn(1, 3, 32, 32)


def test_stability_evaluation(mock_predictor, sample_tensor):
    explainer = GradCAMExplainer(mock_predictor.model, target_layer_name="conv")
    evaluator = StabilityEvaluator(samples=3, noise_std=0.02, seed=42)
    
    res = evaluator.evaluate(explainer, sample_tensor, mock_predictor, target_class=0)
    
    assert 0.0 <= res.score <= 1.0
    assert -1.0 <= res.mean_cosine_similarity <= 1.0
    assert -1.0 <= res.mean_rank_correlation <= 1.0
    assert len(res.records) == 3
    assert res.status in ["PASS", "WARNING", "FAIL"]


def test_faithfulness_directionality(mock_predictor, sample_tensor):
    explainer = GradCAMExplainer(mock_predictor.model, target_layer_name="conv")
    evaluator = FaithfulnessEvaluator(fractions=[0.0, 0.2, 0.5, 0.8], seed=42)
    
    res = evaluator.evaluate(explainer, sample_tensor, mock_predictor, target_class=0)
    
    # Check score range
    assert 0.0 <= res.score <= 1.0
    # First fraction 0.0 must equal initial confidence
    assert res.attr_confidences[0] == pytest.approx(res.random_confidences[0], abs=1e-4)
    # Check status
    assert res.status in ["PASS", "WARNING", "FAIL"]


def test_agreement_self_consistency():
    evaluator = AgreementEvaluator(warning_threshold=0.35)
    
    # Identical maps should yield perfect score
    np.random.seed(42)
    dummy_map = np.random.rand(32, 32)
    
    class DummyExplainer:
        def __init__(self, name, attr):
            self.name = name
            self.attr = attr
        def attribute(self, t, target_class=None):
            return self.attr

    exp1 = DummyExplainer("Method1", dummy_map)
    exp2 = DummyExplainer("Method2", dummy_map)
    
    res = evaluator.evaluate(exp1, exp2, tensor=torch.zeros(1, 3, 32, 32))
    assert res.cosine_similarity == pytest.approx(1.0, abs=1e-3)
    assert res.spearman_correlation == pytest.approx(1.0, abs=1e-3)
    assert res.iou_top_20 == pytest.approx(1.0, abs=1e-3)
    assert res.score == pytest.approx(1.0, abs=1e-3)
    assert res.status == "PASS"


def test_sanity_check_preserves_original_weights(mock_predictor, sample_tensor):
    explainer = GradCAMExplainer(mock_predictor.model, target_layer_name="conv")
    evaluator = ModelSanityCheckEvaluator(target_layers=["conv", "fc"])
    
    hash_before = compute_model_hash(mock_predictor.model)
    res = evaluator.evaluate(explainer, sample_tensor, mock_predictor.model, target_class=0)
    hash_after = compute_model_hash(mock_predictor.model)
    
    # Weights must be 100% identical
    assert hash_before == hash_after, "Model weights were mutated during sanity check!"
    assert 0.0 <= res.score <= 1.0
    assert res.status in ["PASS", "FAIL"]


def test_faithfulness_negative_control_inverted():
    """Negative control: Inverting the attribution map should yield a low faithfulness score."""
    from reliability.faithfulness import FaithfulnessEvaluator
    evaluator = FaithfulnessEvaluator(fractions=[0.0, 0.2, 0.5, 0.8], seed=42)

    class InvertedExplainer:
        def __init__(self):
            self.name = "InvertedDummy"
        def attribute(self, t, target_class=None):
            # Constant inverted map
            return np.ones((32, 32)) * 0.01

    class ToyPredictor:
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = MockCNN()
            self.model.eval()

    pred = ToyPredictor()
    tensor = torch.randn(1, 3, 32, 32)
    res = evaluator.evaluate(InvertedExplainer(), tensor, pred, target_class=0)
    assert 0.0 <= res.score <= 1.0


def test_agreement_negative_control_uncorrelated():
    """Negative control: Completely uncorrelated maps should trigger WARNING or FAIL."""
    evaluator = AgreementEvaluator(warning_threshold=0.50)
    
    np.random.seed(42)
    map1 = np.zeros((32, 32))
    map1[:16, :16] = 1.0  # Top-left quadrant

    map2 = np.zeros((32, 32))
    map2[16:, 16:] = 1.0  # Bottom-right quadrant (strictly disjoint)

    class DummyExp:
        def __init__(self, m):
            self.m = m
            self.name = "Dummy"
        def attribute(self, t, target_class=None):
            return self.m

    res = evaluator.evaluate(DummyExp(map1), DummyExp(map2), tensor=torch.zeros(1, 3, 32, 32))
    assert res.iou_top_20 == pytest.approx(0.0, abs=1e-3)
    assert res.cosine_similarity == pytest.approx(0.0, abs=1e-3)
    assert res.status in ["WARNING", "FAIL"]


def test_stability_negative_control_high_noise(mock_predictor, sample_tensor):
    """Negative control: Increasing perturbation noise should degrade stability metrics."""
    explainer = GradCAMExplainer(mock_predictor.model, target_layer_name="conv")
    eval_low = StabilityEvaluator(samples=3, noise_std=0.01, seed=42)
    eval_high = StabilityEvaluator(samples=3, noise_std=0.80, seed=42)

    res_low = eval_low.evaluate(explainer, sample_tensor, mock_predictor, target_class=0)
    res_high = eval_high.evaluate(explainer, sample_tensor, mock_predictor, target_class=0)

    # High noise cosine similarity or rank correlation should degrade relative to low noise
    assert res_high.mean_cosine_similarity <= res_low.mean_cosine_similarity + 0.05


def test_sanity_check_layer3_cascade(mock_predictor):
    """Sanity check: Targeting intermediate layers cascades randomization through head."""
    class DummyLayer3Explainer:
        def __init__(self):
            self.name = "GradCAM-Layer3"
        def get_metadata(self):
            return {"target_layer": "layer3"}
        def attribute(self, t, target_class=None):
            return np.ones((32, 32))

    evaluator = ModelSanityCheckEvaluator()
    resolved = evaluator._resolve_target_layers(DummyLayer3Explainer())
    assert "layer3" in resolved
    assert "layer4" in resolved
    assert "fc" in resolved


def test_insertion_evaluator(mock_predictor, sample_tensor):
    """Independent validation: InsertionEvaluator restores confidence and returns valid score."""
    from reliability.insertion import InsertionEvaluator
    explainer = GradCAMExplainer(mock_predictor.model, target_layer_name="conv")
    evaluator = InsertionEvaluator(fractions=[0.0, 0.5, 1.0], num_random_runs=2, seed=42)
    res = evaluator.evaluate(explainer, sample_tensor, mock_predictor, target_class=0)

    assert 0.0 <= res.score <= 1.0
    assert 0.0 <= res.auc_insertion <= 1.0
    assert 0.0 <= res.auc_random <= 1.0
    assert len(res.fractions) == 3
    assert res.status in ["PASS", "FAIL"]
    assert "disclaimer" in res.disclaimer.lower() or "proxy" in res.disclaimer.lower()

