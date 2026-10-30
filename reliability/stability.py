"""
Explanation Stability Evaluation Engine.
Measures attribution invariance under small, label-preserving input perturbations
using cosine similarity and Spearman rank correlation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch

from explainers.base_explainer import BaseExplainer
from models.predictor import ModelPredictor
from utils.image_utils import apply_perturbation
from utils.metrics import cosine_similarity_2d, spearman_rank_correlation


@dataclass
class PerturbationRecord:
    perturbation_type: str
    magnitude: float
    seed: int
    cosine_similarity: float
    rank_correlation: float
    label_preserved: bool
    original_class: int
    perturbed_class: int


@dataclass
class StabilityResult:
    score: float  # [0, 1], higher is more stable
    mean_cosine_similarity: float
    mean_rank_correlation: float
    label_preservation_rate: float
    records: List[PerturbationRecord] = field(default_factory=list)
    status: str = "PASS"  # 'PASS', 'WARNING', 'FAIL'
    evidence: str = ""


class StabilityEvaluator:
    """
    Evaluates attribution stability under controlled input perturbations.
    """
    def __init__(
        self,
        samples: int = 5,
        noise_std: float = 0.05,
        warning_threshold: float = 0.60,
        fail_threshold: float = 0.40,
        seed: int = 42
    ):
        self.samples = samples
        self.noise_std = noise_std
        self.warning_threshold = warning_threshold
        self.fail_threshold = fail_threshold
        self.seed = seed

    def evaluate(
        self,
        explainer: BaseExplainer,
        tensor: torch.Tensor,
        predictor: ModelPredictor,
        target_class: Optional[int] = None
    ) -> StabilityResult:
        """
        Evaluate stability by applying controlled perturbations and comparing attributions.
        """
        # Baseline prediction and attribution
        orig_class, orig_conf, _ = predictor.predict_tensor(tensor)
        if target_class is None:
            target_class = orig_class

        base_attr = explainer.attribute(tensor, target_class=target_class)

        perturbation_types = ["gaussian_noise", "slight_blur", "brightness_jitter"]
        records: List[PerturbationRecord] = []
        
        sim_scores: List[float] = []
        rank_scores: List[float] = []
        preserved_count = 0

        total_runs = self.samples

        for i in range(total_runs):
            p_type = perturbation_types[i % len(perturbation_types)]
            run_seed = self.seed + i * 17
            
            # Perturb tensor
            perturbed_tensor = apply_perturbation(
                tensor=tensor,
                perturbation_type=p_type,
                magnitude=self.noise_std,
                seed=run_seed
            )

            # Check if model prediction changed
            p_class, _, _ = predictor.predict_tensor(perturbed_tensor)
            label_preserved = (p_class == orig_class)
            if label_preserved:
                preserved_count += 1

            # Compute attribution on perturbed input
            pert_attr = explainer.attribute(perturbed_tensor, target_class=target_class)

            cos_sim = cosine_similarity_2d(base_attr, pert_attr)
            rank_corr = spearman_rank_correlation(base_attr, pert_attr)

            record = PerturbationRecord(
                perturbation_type=p_type,
                magnitude=self.noise_std,
                seed=run_seed,
                cosine_similarity=round(cos_sim, 4),
                rank_correlation=round(rank_corr, 4),
                label_preserved=label_preserved,
                original_class=orig_class,
                perturbed_class=p_class
            )
            records.append(record)

            # Only accumulate similarity metrics when the label is preserved
            # (otherwise, the explanation legitimately ought to shift if the prediction shifted)
            if label_preserved:
                sim_scores.append(cos_sim)
                rank_scores.append(rank_corr)

        label_preservation_rate = preserved_count / max(1, total_runs)

        if len(sim_scores) == 0:
            # All perturbations flipped label (extreme sensitivity of model)
            mean_cos = 0.0
            mean_rank = 0.0
            norm_score = 0.0
            status = "FAIL"
            evidence = "Model prediction flipped on all perturbations; cannot reliably evaluate explanation stability."
        else:
            mean_cos = float(np.mean(sim_scores))
            mean_rank = float(np.mean(rank_scores))

            # Map cosine [-1, 1] and rank [-1, 1] to [0, 1]
            cos_norm = float(np.clip((mean_cos + 1.0) / 2.0, 0.0, 1.0))
            rank_norm = float(np.clip((mean_rank + 1.0) / 2.0, 0.0, 1.0))
            norm_score = float(np.clip(0.6 * cos_norm + 0.4 * rank_norm, 0.0, 1.0))

            if norm_score >= self.warning_threshold:
                status = "PASS"
                evidence = f"Attribution exhibits high consistency under perturbations (Mean Cosine: {mean_cos:.3f}, Spearman: {mean_rank:.3f})."
            elif norm_score >= self.fail_threshold:
                status = "WARNING"
                evidence = f"Moderate attribution sensitivity detected under small noise (Score: {norm_score:.3f} < {self.warning_threshold})."
            else:
                status = "FAIL"
                evidence = f"Severe attribution instability detected: small noise causes drastic explanation shifts (Score: {norm_score:.3f} < {self.fail_threshold})."

        return StabilityResult(
            score=round(norm_score, 4),
            mean_cosine_similarity=round(mean_cos, 4),
            mean_rank_correlation=round(mean_rank, 4),
            label_preservation_rate=round(label_preservation_rate, 3),
            records=records,
            status=status,
            evidence=evidence
        )
