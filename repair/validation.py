"""
Validation Engine for Adaptive Explanation Repair.
Enforces non-invasive repair constraints: guarantees model weights, predictions,
and class confidences are 100% invariant throughout explanation repair.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import hashlib
import torch
import torch.nn as nn


@dataclass
class ModelIntegritySnapshot:
    parameter_hash: str
    prediction_class: int
    confidence: float


def compute_model_hash(model: nn.Module) -> str:
    """
    Computes a cryptographic SHA-256 fingerprint of all model parameters
    to verify that model weights were never mutated during repair.
    """
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            param_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(name.encode("utf-8"))
            hasher.update(param_bytes)
    return hasher.hexdigest()


class RepairValidator:
    """
    Validates experimental integrity and prevents deceptive or circular repairs.
    """
    @staticmethod
    def capture_snapshot(
        model: nn.Module,
        tensor: torch.Tensor
    ) -> ModelIntegritySnapshot:
        """Capture parameter fingerprint and model output before repair."""
        model_hash = compute_model_hash(model)
        device = next(model.parameters()).device
        x = tensor.to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred_class = int(torch.argmax(probs, dim=-1).item())
            conf = float(probs[0, pred_class].item())

        return ModelIntegritySnapshot(
            parameter_hash=model_hash,
            prediction_class=pred_class,
            confidence=conf
        )

    @staticmethod
    def verify_integrity(
        before_snapshot: ModelIntegritySnapshot,
        model: nn.Module,
        tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Verify that model weights and inference predictions are unaltered.
        """
        after_hash = compute_model_hash(model)
        device = next(model.parameters()).device
        x = tensor.to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            after_class = int(torch.argmax(probs, dim=-1).item())
            after_conf = float(probs[0, after_class].item())

        weights_intact = (before_snapshot.parameter_hash == after_hash)
        prediction_intact = (
            before_snapshot.prediction_class == after_class and
            abs(before_snapshot.confidence - after_conf) < 1e-5
        )

        return {
            "weights_preserved": weights_intact,
            "prediction_preserved": prediction_intact,
            "before_class": before_snapshot.prediction_class,
            "after_class": after_class,
            "before_conf": round(before_snapshot.confidence, 4),
            "after_conf": round(after_conf, 4),
            "is_valid": weights_intact and prediction_intact
        }
