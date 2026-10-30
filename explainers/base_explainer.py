"""
Abstract Base Class for Model Explainers.
Ensures uniform interface for attribution generation and metadata tracking.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import time
import numpy as np
import torch
import torch.nn as nn


class BaseExplainer(ABC):
    """
    Common interface for all attribution and explanation generators.
    """
    def __init__(self, model: nn.Module, name: str):
        self.model = model
        self.name = name
        self.last_execution_time: float = 0.0
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def attribute(
        self,
        tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate 2D attribution heatmap [H, W] normalized to [0, 1].
        tensor: Shape [1, 3, H, W]
        target_class: Integer class index to explain. If None, uses top predicted class.
        """
        pass

    def explain(
        self,
        tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Wrapper that measures execution time and returns (attribution_map, metadata).
        """
        start_t = time.perf_counter()
        attr_map = self.attribute(tensor, target_class=target_class)
        self.last_execution_time = time.perf_counter() - start_t
        
        meta = self.get_metadata()
        meta["execution_time_seconds"] = round(self.last_execution_time, 4)
        return attr_map, meta

    def get_metadata(self) -> Dict[str, Any]:
        """Returns parameters and metadata of the explainer."""
        return {
            "explainer_name": self.name,
            **self.metadata
        }
