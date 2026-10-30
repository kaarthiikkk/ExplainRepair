"""
Explanation Faithfulness Evaluation Engine.
Evaluates whether attributed salient features causally drive model confidence
via progressive feature deletion curves compared with random masking baselines.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from explainers.base_explainer import BaseExplainer
from models.predictor import ModelPredictor
from utils.metrics import compute_auc


@dataclass
class FaithfulnessResult:
    score: float  # [0, 1], higher is more faithful
    auc_attribution: float
    auc_random: float
    auc_difference: float
    fractions: List[float] = field(default_factory=list)
    attr_confidences: List[float] = field(default_factory=list)
    random_confidences: List[float] = field(default_factory=list)
    status: str = "PASS"  # 'PASS', 'WARNING', 'FAIL'
    evidence: str = ""


class FaithfulnessEvaluator:
    """
    Evaluates attribution faithfulness using progressive deletion experiments.
    """
    def __init__(
        self,
        fractions: Optional[List[float]] = None,
        warning_threshold: float = 0.50,
        fail_threshold: float = 0.35,
        mask_value: str = "mean",
        seed: int = 42
    ):
        if fractions is None:
            self.fractions = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]
        else:
            self.fractions = sorted(fractions)
        self.warning_threshold = warning_threshold
        self.fail_threshold = fail_threshold
        self.mask_value = mask_value
        self.seed = seed

    def _apply_mask(
        self,
        tensor: torch.Tensor,
        mask_indices_flat: Sequence[int]
    ) -> torch.Tensor:
        """
        Mask specified spatial pixels in tensor [1, 3, H, W].
        """
        masked = tensor.clone()
        c, h, w = masked.shape[1], masked.shape[2], masked.shape[3]
        
        # Flatten spatial dimensions [1, 3, H*W]
        flat = masked.view(1, c, h * w)
        
        idx = np.asarray(mask_indices_flat, dtype=np.int64).copy()
        
        if self.mask_value == "mean":
            # Set to mean channel values
            mean_vals = flat.mean(dim=2, keepdim=True)  # [1, 3, 1]
            flat[:, :, idx] = mean_vals
        elif self.mask_value == "zero":
            flat[:, :, idx] = 0.0
        else:
            flat[:, :, idx] = 0.0

        return flat.view(1, c, h, w)

    def evaluate(
        self,
        explainer: BaseExplainer,
        tensor: torch.Tensor,
        predictor: ModelPredictor,
        target_class: Optional[int] = None
    ) -> FaithfulnessResult:
        """
        Compute deletion curves for attribution ranking vs random ranking.
        """
        device = predictor.device
        x = tensor.clone().detach().to(device)

        with torch.no_grad():
            orig_logits = predictor.model(x)
            orig_probs = F.softmax(orig_logits, dim=-1)
            if target_class is None:
                target_class = int(torch.argmax(orig_probs, dim=-1).item())
            orig_conf = float(orig_probs[0, target_class].item())

        # Generate attribution map
        attr_map = explainer.attribute(tensor, target_class=target_class)  # [H, W]
        h, w = attr_map.shape
        num_pixels = h * w

        # Order pixels by attribution descending
        flat_attr = attr_map.flatten()
        sorted_attr_indices = np.argsort(flat_attr)[::-1]

        # Multi-run random baseline to reduce single-trial variance
        num_runs = getattr(self, "num_random_runs", 3)
        rng = np.random.RandomState(self.seed)
        random_perms = [rng.permutation(num_pixels) for _ in range(num_runs)]

        attr_confs: List[float] = []
        rand_confs_all: List[List[float]] = [[] for _ in range(num_runs)]

        predictor.model.eval()
        with torch.no_grad():
            for frac in self.fractions:
                k = int(num_pixels * frac)
                
                if k == 0:
                    attr_confs.append(orig_conf)
                    for r_i in range(num_runs):
                        rand_confs_all[r_i].append(orig_conf)
                    continue

                # 1. Mask top k attributed pixels
                attr_mask_idx = sorted_attr_indices[:k]
                t_attr_masked = self._apply_mask(x, attr_mask_idx)
                p_attr = F.softmax(predictor.model(t_attr_masked), dim=-1)[0, target_class].item()
                attr_confs.append(float(p_attr))

                # 2. Mask k random pixels across runs
                for r_i, perm in enumerate(random_perms):
                    rand_mask_idx = perm[:k]
                    t_rand_masked = self._apply_mask(x, rand_mask_idx)
                    p_rand = F.softmax(predictor.model(t_rand_masked), dim=-1)[0, target_class].item()
                    rand_confs_all[r_i].append(float(p_rand))

        # Average random baseline curves
        rand_confs_mean = [float(np.mean([rand_confs_all[r_i][step_i] for r_i in range(num_runs)]))
                           for step_i in range(len(self.fractions))]

        # Compute AUCs of deletion curves
        auc_attr = compute_auc(self.fractions, attr_confs)
        auc_rand = compute_auc(self.fractions, rand_confs_mean)
        auc_diff = auc_rand - auc_attr  # Positive means attribution dropped faster than random (faithful)

        # Map to [0, 1] scale where higher is better:
        # If auc_diff == 0 (attribution behaves same as random), score = 0.50
        # If auc_diff > 0 (attribution degrades target score faster), score > 0.50
        # Normalization denominator based on max possible drop
        norm_denom = max(0.10, auc_rand)
        normalized_score = float(np.clip(0.50 + (auc_diff / (2.0 * norm_denom)), 0.0, 1.0))

        if normalized_score >= self.warning_threshold:
            status = "PASS"
            evidence = (
                f"Feature removal rapidly degraded target confidence (AUC Attr: {auc_attr:.3f} "
                f"vs Random: {auc_rand:.3f}, Δ: +{auc_diff:.3f})."
            )
        elif normalized_score >= self.fail_threshold:
            status = "WARNING"
            evidence = (
                f"Marginal faithfulness: removing attributed features is only slightly more impactful "
                f"than random masking (Score: {normalized_score:.3f})."
            )
        else:
            status = "FAIL"
            evidence = (
                f"Faithfulness failure: removing attributed features did NOT degrade model prediction "
                f"faster than random pixels (AUC Attr: {auc_attr:.3f} >= Random: {auc_rand:.3f})."
            )

        return FaithfulnessResult(
            score=round(normalized_score, 4),
            auc_attribution=round(auc_attr, 4),
            auc_random=round(auc_rand, 4),
            auc_difference=round(auc_diff, 4),
            fractions=self.fractions,
            attr_confidences=[round(c, 4) for c in attr_confs],
            random_confidences=[round(c, 4) for c in rand_confs_mean],
            status=status,
            evidence=evidence
        )
