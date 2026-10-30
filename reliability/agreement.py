"""
Cross-Method Agreement Evaluation Engine.
Measures spatial and rank consistency between distinct explanation algorithms
(e.g., Grad-CAM vs. Integrated Gradients).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from explainers.base_explainer import BaseExplainer
from utils.metrics import cosine_similarity_2d, spearman_rank_correlation, iou_top_k, normalize_map


@dataclass
class AgreementResult:
    score: float  # [0, 1], higher indicates stronger consensus
    cosine_similarity: float
    spearman_correlation: float
    iou_top_20: float
    method_a: str
    method_b: str
    status: str  # 'PASS', 'WARNING', 'FAIL'
    evidence: str
    caveat: str


class AgreementEvaluator:
    """
    Evaluates whether two explanation methods converge on the same salient features.
    """
    def __init__(
        self,
        warning_threshold: float = 0.35,
        top_k_fraction: float = 0.20
    ):
        self.warning_threshold = warning_threshold
        self.top_k_fraction = top_k_fraction

    def evaluate(
        self,
        explainer_a: BaseExplainer,
        explainer_b: BaseExplainer,
        tensor: torch.Tensor,
        target_class: Optional[int] = None,
        cached_attr_a: Optional[np.ndarray] = None,
        cached_attr_b: Optional[np.ndarray] = None
    ) -> AgreementResult:
        """
        Compute agreement metrics between two explainers.
        Accepts optional cached attributions to avoid redundant computations.
        """
        attr_a = cached_attr_a if cached_attr_a is not None else explainer_a.attribute(tensor, target_class=target_class)
        attr_b = cached_attr_b if cached_attr_b is not None else explainer_b.attribute(tensor, target_class=target_class)

        # Normalize both maps to [0, 1]
        norm_a = normalize_map(attr_a)
        norm_b = normalize_map(attr_b)

        # 1. Cosine similarity
        cos_sim = cosine_similarity_2d(norm_a, norm_b)
        cos_norm = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

        # 2. Spearman rank correlation
        spearman_rho = spearman_rank_correlation(norm_a, norm_b)
        spearman_norm = float(np.clip((spearman_rho + 1.0) / 2.0, 0.0, 1.0))

        # 3. Top-20% salient pixel IoU
        iou = iou_top_k(norm_a, norm_b, top_fraction=self.top_k_fraction)

        # Composite agreement score
        composite_score = float(np.clip(0.40 * cos_norm + 0.30 * spearman_norm + 0.30 * iou, 0.0, 1.0))

        caveat = (
            "Disagreement does not necessarily invalidate an explanation: "
            "Grad-CAM tracks high-level activation maps, whereas Integrated Gradients "
            "computes pixel-level path integrals. Divergence indicates methodological sensitivity."
        )

        if composite_score >= self.warning_threshold:
            status = "PASS"
            evidence = (
                f"Consensus between {explainer_a.name} and {explainer_b.name} "
                f"(IoU Top-20%: {iou:.2f}, Cosine: {cos_sim:.2f}, Spearman: {spearman_rho:.2f})."
            )
        else:
            status = "WARNING"
            evidence = (
                f"Low spatial agreement between {explainer_a.name} and {explainer_b.name} "
                f"(Composite: {composite_score:.2f} < {self.warning_threshold}; IoU: {iou:.2f})."
            )

        return AgreementResult(
            score=round(composite_score, 4),
            cosine_similarity=round(cos_sim, 4),
            spearman_correlation=round(spearman_rho, 4),
            iou_top_20=round(iou, 4),
            method_a=explainer_a.name,
            method_b=explainer_b.name,
            status=status,
            evidence=evidence,
            caveat=caveat
        )
