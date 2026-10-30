"""
Perturbation Stress Testing Suite.
Applies parameterized noise and blur spectra to evaluate explanation fragility.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch

from explainers.base_explainer import BaseExplainer
from models.predictor import ModelPredictor
from utils.image_utils import apply_perturbation
from utils.metrics import cosine_similarity_2d, spearman_rank_correlation


def run_perturbation_stress_test(
    explainer: BaseExplainer,
    tensor: torch.Tensor,
    predictor: ModelPredictor,
    noise_levels: List[float] = [0.01, 0.03, 0.05, 0.10, 0.15],
    target_class: int = None,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    Stress-tests an explainer across an ascending gradient of noise magnitudes.
    """
    orig_class, _, _ = predictor.predict_tensor(tensor)
    if target_class is None:
        target_class = orig_class

    base_attr = explainer.attribute(tensor, target_class=target_class)

    cosine_similarities = []
    rank_correlations = []
    preservation_flags = []

    for idx, noise_mag in enumerate(noise_levels):
        pert_tensor = apply_perturbation(
            tensor=tensor,
            perturbation_type="gaussian_noise",
            magnitude=noise_mag,
            seed=seed + idx * 10
        )
        
        # Check label preservation
        p_class, _, _ = predictor.predict_tensor(pert_tensor)
        label_preserved = (p_class == orig_class)
        preservation_flags.append(label_preserved)

        pert_attr = explainer.attribute(pert_tensor, target_class=target_class)
        cos_sim = cosine_similarity_2d(base_attr, pert_attr)
        rank_corr = spearman_rank_correlation(base_attr, pert_attr)

        cosine_similarities.append(round(cos_sim, 4))
        rank_correlations.append(round(rank_corr, 4))

    return {
        "noise_levels": noise_levels,
        "cosine_similarities": cosine_similarities,
        "rank_correlations": rank_correlations,
        "label_preserved": preservation_flags
    }
