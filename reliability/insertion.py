"""
Insertion Evaluation Engine (Petsiuk et al., 2018; Samek et al., 2016).
Provides an independent proxy metric for explanation validation decoupled
from deletion-based candidate selection.

Measures feature sufficiency: starting from a baseline uninformative image
(e.g., heavily blurred or mean-filled), features are progressively inserted
in descending order of attribution salience. A faithful explanation should
cause target class confidence to surge rapidly (high Area Under Insertion Curve).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from explainers.base_explainer import BaseExplainer
from models.predictor import ModelPredictor
from utils.image_utils import denormalize_tensor, preprocess_image
from utils.metrics import compute_auc
from PIL import Image, ImageFilter


@dataclass
class InsertionResult:
    score: float  # [0, 1], higher indicates greater causal sufficiency
    auc_insertion: float
    auc_random: float
    auc_difference: float
    fractions: List[float] = field(default_factory=list)
    attr_confidences: List[float] = field(default_factory=list)
    random_confidences: List[float] = field(default_factory=list)
    status: str = "PASS"  # 'PASS', 'WARNING', 'FAIL'
    evidence: str = ""
    disclaimer: str = (
        "Proxy Metric Disclaimer: Insertion AUC measures feature sufficiency under baseline synthesis. "
        "It provides an independent corroborative check but does not constitute ground-truth proof."
    )


class InsertionEvaluator:
    """
    Evaluates attribution sufficiency via progressive feature insertion experiments.
    Used as an independent validation metric decoupled from deletion-based candidate selection.
    """
    def __init__(
        self,
        fractions: Optional[List[float]] = None,
        warning_threshold: float = 0.50,
        baseline_mode: str = "blur",  # 'blur' or 'mean'
        num_random_runs: int = 5,
        seed: int = 42
    ):
        if fractions is None:
            self.fractions = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.0]
        else:
            self.fractions = sorted(fractions)
        self.warning_threshold = warning_threshold
        self.baseline_mode = baseline_mode
        self.num_random_runs = num_random_runs
        self.seed = seed

    def _generate_baseline_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Construct uninformative baseline: Gaussian blurred input or channel mean."""
        if self.baseline_mode == "blur":
            rgb = denormalize_tensor(tensor)
            pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=5.0))
            t_base, _ = preprocess_image(blurred_pil, target_size=(tensor.shape[2], tensor.shape[3]))
            return t_base.to(tensor.device)
        else:
            c = tensor.shape[1]
            means = tensor.view(1, c, -1).mean(dim=2, keepdim=True).view(1, c, 1, 1)
            return means.expand_as(tensor).clone()

    def _apply_insertion(
        self,
        base_tensor: torch.Tensor,
        original_tensor: torch.Tensor,
        insert_indices_flat: Sequence[int]
    ) -> torch.Tensor:
        """Insert original image features at specified spatial positions into base image."""
        composite = base_tensor.clone()
        c, h, w = composite.shape[1], composite.shape[2], composite.shape[3]
        
        flat_comp = composite.view(1, c, h * w)
        flat_orig = original_tensor.view(1, c, h * w)
        
        idx = np.asarray(insert_indices_flat, dtype=np.int64).copy()
        flat_comp[:, :, idx] = flat_orig[:, :, idx]
        
        return flat_comp.view(1, c, h, w)

    def evaluate(
        self,
        explainer: BaseExplainer,
        tensor: torch.Tensor,
        predictor: ModelPredictor,
        target_class: Optional[int] = None,
        cached_attr_map: Optional[np.ndarray] = None
    ) -> InsertionResult:
        """
        Evaluate insertion curve: measure target class confidence recovery.
        """
        device = predictor.device
        x = tensor.clone().detach().to(device)
        baseline = self._generate_baseline_tensor(x)

        with torch.no_grad():
            orig_probs = F.softmax(predictor.model(x), dim=-1)
            base_probs = F.softmax(predictor.model(baseline), dim=-1)
            if target_class is None:
                target_class = int(torch.argmax(orig_probs, dim=-1).item())
            
            orig_conf = float(orig_probs[0, target_class].item())
            base_conf = float(base_probs[0, target_class].item())

        # Generate or use cached attribution
        if cached_attr_map is not None:
            attr_map = cached_attr_map
        else:
            attr_map = explainer.attribute(tensor, target_class=target_class)

        h, w = attr_map.shape
        num_pixels = h * w

        flat_attr = attr_map.flatten()
        sorted_attr_indices = np.argsort(flat_attr)[::-1].copy()

        # Multi-run random baseline to eliminate single-trial variance
        rng = np.random.RandomState(self.seed)
        random_perms = [rng.permutation(num_pixels) for _ in range(self.num_random_runs)]

        attr_confs: List[float] = []
        rand_confs_all: List[List[float]] = [[] for _ in range(self.num_random_runs)]

        predictor.model.eval()
        with torch.no_grad():
            for frac in self.fractions:
                k = int(num_pixels * frac)
                
                if k == 0:
                    attr_confs.append(base_conf)
                    for r_i in range(self.num_random_runs):
                        rand_confs_all[r_i].append(base_conf)
                    continue

                if k >= num_pixels:
                    attr_confs.append(orig_conf)
                    for r_i in range(self.num_random_runs):
                        rand_confs_all[r_i].append(orig_conf)
                    continue

                # 1. Attribution-guided insertion
                attr_idx = sorted_attr_indices[:k]
                t_attr_ins = self._apply_insertion(baseline, x, attr_idx)
                p_attr = F.softmax(predictor.model(t_attr_ins), dim=-1)[0, target_class].item()
                attr_confs.append(float(p_attr))

                # 2. Random insertion runs
                for r_i, perm in enumerate(random_perms):
                    r_idx = perm[:k]
                    t_rand_ins = self._apply_insertion(baseline, x, r_idx)
                    p_rand = F.softmax(predictor.model(t_rand_ins), dim=-1)[0, target_class].item()
                    rand_confs_all[r_i].append(float(p_rand))

        # Average random baseline over runs
        rand_confs_mean = [float(np.mean([rand_confs_all[r_i][step_i] for r_i in range(self.num_random_runs)]))
                           for step_i in range(len(self.fractions))]

        auc_attr = compute_auc(self.fractions, attr_confs)
        auc_rand = compute_auc(self.fractions, rand_confs_mean)
        auc_diff = auc_attr - auc_rand  # Positive means attribution recovered confidence FASTER than random

        # Normalize score into [0, 1]
        # Max possible gain is ~ 1.0 - auc_rand
        norm_denom = max(0.10, 1.0 - auc_rand)
        normalized_score = float(np.clip(0.50 + (auc_diff / (2.0 * norm_denom)), 0.0, 1.0))

        if normalized_score >= self.warning_threshold:
            status = "PASS"
            evidence = (
                f"Feature insertion rapidly restored target confidence (AUC Attr: {auc_attr:.3f} "
                f"vs Random: {auc_rand:.3f}, Δ: +{auc_diff:.3f})."
            )
        else:
            status = "FAIL"
            evidence = (
                f"Feature insertion did not restore target confidence faster than random features "
                f"(AUC Attr: {auc_attr:.3f} <= Random: {auc_rand:.3f})."
            )

        return InsertionResult(
            score=round(normalized_score, 4),
            auc_insertion=round(auc_attr, 4),
            auc_random=round(auc_rand, 4),
            auc_difference=round(auc_diff, 4),
            fractions=self.fractions,
            attr_confidences=[round(c, 4) for c in attr_confs],
            random_confidences=[round(c, 4) for c in rand_confs_mean],
            status=status,
            evidence=evidence
        )
