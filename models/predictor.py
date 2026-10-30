"""
Predictor wrapper for ResNet-18 inference on CIFAR-10.
Handles preprocessing, device allocation, top-k predictions, and confidence estimation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CIFAR10_CLASSES, default_config
from models.model import build_model
from utils.image_utils import preprocess_image


@dataclass
class PredictionResult:
    predicted_class_idx: int
    predicted_class_name: str
    confidence: float
    probabilities: np.ndarray
    top_k: List[Tuple[str, float]]
    device_used: str
    is_model_trained: bool
    model_source: str

    @property
    def predicted_class(self) -> int:
        return self.predicted_class_idx

    @property
    def class_name(self) -> str:
        return self.predicted_class_name

    @property
    def top_classes(self) -> List[int]:
        return [CIFAR10_CLASSES.index(name) if name in CIFAR10_CLASSES else 0 for name, _ in self.top_k]

    @property
    def top_probabilities(self) -> List[float]:
        return [prob for _, prob in self.top_k]


class ModelPredictor:
    """
    Inference manager for CIFAR-10 ResNet-18 model.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = default_config.device
        else:
            self.device = device

        if checkpoint_path is None:
            checkpoint_path = default_config.model.checkpoint_path

        self.model, self.model_status = build_model(
            num_classes=default_config.model.num_classes,
            checkpoint_path=checkpoint_path,
            device=self.device
        )
        self.model.eval()

    def predict_tensor(self, tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """
        Inference on preprocessed tensor of shape [1, 3, H, W] or [B, 3, H, W].
        Returns (predicted_class_idx, confidence, probabilities).
        """
        self.model.eval()
        with torch.no_grad():
            x = tensor.to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
            
            top_prob, top_idx = torch.max(probs, dim=-1)
            
            # For single sample
            if tensor.size(0) == 1:
                return (
                    int(top_idx.item()),
                    float(top_prob.item()),
                    probs.squeeze(0).cpu().numpy()
                )
            return (
                int(top_idx[0].item()),
                float(top_prob[0].item()),
                probs[0].cpu().numpy()
            )

    def predict_image(
        self,
        image: Union[Image.Image, np.ndarray],
        top_k: int = 5
    ) -> PredictionResult:
        """
        Full inference pipeline from PIL image or numpy array to structured PredictionResult.
        """
        tensor, _ = preprocess_image(image)
        pred_idx, confidence, probs = self.predict_tensor(tensor)
        
        # Extract top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [(CIFAR10_CLASSES[i], float(probs[i])) for i in top_indices]
        
        return PredictionResult(
            predicted_class_idx=pred_idx,
            predicted_class_name=CIFAR10_CLASSES[pred_idx],
            confidence=confidence,
            probabilities=probs,
            top_k=top_k_list,
            device_used=str(self.device),
            is_model_trained=self.model_status.get("is_trained", False),
            model_source=self.model_status.get("source", "unknown")
        )

    def predict(
        self,
        input_data: Union[torch.Tensor, Image.Image, np.ndarray],
        top_k: int = 5
    ) -> PredictionResult:
        """
        Unified inference method accepting a preprocessed torch.Tensor, PIL.Image, or np.ndarray.
        """
        if isinstance(input_data, torch.Tensor):
            pred_idx, confidence, probs = self.predict_tensor(input_data)
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_k_list = [(CIFAR10_CLASSES[i], float(probs[i])) for i in top_indices]
            return PredictionResult(
                predicted_class_idx=pred_idx,
                predicted_class_name=CIFAR10_CLASSES[pred_idx],
                confidence=confidence,
                probabilities=probs,
                top_k=top_k_list,
                device_used=str(self.device),
                is_model_trained=self.model_status.get("is_trained", False),
                model_source=self.model_status.get("source", "unknown")
            )
        else:
            return self.predict_image(input_data, top_k=top_k)

