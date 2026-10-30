"""
Model Sensitivity Sanity Check Engine (Adebayo et al., NeurIPS 2018).
Evaluates whether explanations collapse when model parameters are randomized.
An explanation that remains unchanged on a randomized model fails to reflect learned weights.
"""

from dataclasses import dataclass
from typing import List, Optional
import copy
import numpy as np
import torch
import torch.nn as nn

from explainers.base_explainer import BaseExplainer
from explainers.explainer_factory import ExplainerFactory
from utils.metrics import cosine_similarity_2d, spearman_rank_correlation, normalize_map


@dataclass
class SanityCheckResult:
    score: float  # [0, 1], higher indicates superior sensitivity to learned weights (1.0 = ideal collapse)
    similarity_with_randomized: float
    spearman_with_randomized: float
    randomized_layers: List[str]
    status: str  # 'PASS', 'WARNING', 'FAIL'
    evidence: str
    limitation: str


class ModelSanityCheckEvaluator:
    """
    Evaluates model sensitivity via parameter randomization test.
    Preserves original model state with 100% guarantee.
    """
    def __init__(
        self,
        max_similarity_threshold: float = 0.65,
        target_layers: Optional[List[str]] = None
    ):
        self.max_similarity_threshold = max_similarity_threshold
        self.target_layers = target_layers or ["fc", "layer4"]

    def _randomize_module_weights(self, module: nn.Module) -> None:
        """Re-initializes weights and biases of a PyTorch module."""
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        elif hasattr(module, "weight") and module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _resolve_target_layers(self, explainer: BaseExplainer) -> List[str]:
        """
        Determines the appropriate layers to randomize based on the explainer's target.
        Ensures intermediate layers (e.g., layer3) cannot pass by avoiding randomization.
        """
        meta = explainer.get_metadata()
        explainer_layer = meta.get("target_layer", "")

        hierarchy = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
        
        # If explainer targets a specific layer in hierarchy, randomize that layer and subsequent layers
        matched_idx = None
        for i, h_name in enumerate(hierarchy):
            if h_name in explainer_layer:
                matched_idx = i
                break

        if matched_idx is not None:
            # Randomize from target layer to the classification head
            return hierarchy[matched_idx:]
        
        # Default: randomize classification head and last block
        return list(dict.fromkeys(self.target_layers + [explainer_layer] if explainer_layer else self.target_layers))

    def evaluate(
        self,
        explainer: BaseExplainer,
        tensor: torch.Tensor,
        original_model: nn.Module,
        target_class: Optional[int] = None,
        cached_attr_original: Optional[np.ndarray] = None
    ) -> SanityCheckResult:
        """
        Runs parameter randomization test on an isolated deepcopy of the model.
        Guarantees original model weights are restored even if an exception occurs.
        """
        # Backup original model weights to enforce non-invasive invariant under any circumstance
        original_state_backup = copy.deepcopy(original_model.state_dict())

        try:
            # 1. Base attribution with trained model
            if cached_attr_original is not None:
                attr_original = cached_attr_original
            else:
                attr_original = explainer.attribute(tensor, target_class=target_class)

            # 2. Create isolated deepcopy of model to mutate
            model_copy = copy.deepcopy(original_model)
            model_copy.eval()
            
            target_module = model_copy.backbone if hasattr(model_copy, "backbone") else model_copy

            layers_to_randomize = self._resolve_target_layers(explainer)

            randomized_found = []
            for name, mod in target_module.named_modules():
                for target_name in layers_to_randomize:
                    if target_name in name and isinstance(mod, (nn.Linear, nn.Conv2d)):
                        self._randomize_module_weights(mod)
                        randomized_found.append(name)
                        break

            if not randomized_found:
                # Fallback: reinit fc
                if hasattr(target_module, "fc"):
                    self._randomize_module_weights(target_module.fc)
                    randomized_found.append("fc")

            # 3. Create explainer copy for the randomized model
            explainer_config = explainer.get_metadata()
            explainer_randomized = ExplainerFactory.create_explainer(
                model=model_copy,
                config=explainer_config
            )

            attr_randomized = explainer_randomized.attribute(tensor, target_class=target_class)

            # Clean up model copy
            del model_copy
            del explainer_randomized

            # 4. Compare original vs randomized attribution
            norm_orig = normalize_map(attr_original)
            norm_rand = normalize_map(attr_randomized)

            cos_sim = cosine_similarity_2d(norm_orig, norm_rand)
            spearman_rho = spearman_rank_correlation(norm_orig, norm_rand)

            # The MORE dissimilar, the BETTER the sanity check passed
            sim_pos = max(0.0, cos_sim)
            sanity_score = float(np.clip(1.0 - sim_pos, 0.0, 1.0))

            limitation = (
                "A passed sanity check confirms sensitivity to learned weights; "
                "however, it does not guarantee optimal explanation accuracy or ground-truth alignment."
            )

            if cos_sim <= self.max_similarity_threshold:
                status = "PASS"
                evidence = (
                    f"Explanation successfully collapsed under parameter randomization "
                    f"(Cosine Similarity: {cos_sim:.2f} <= {self.max_similarity_threshold}; "
                    f"Randomized layers: {', '.join(randomized_found[:4])})."
                )
            else:
                status = "FAIL"
                evidence = (
                    f"Sanity check warning: explanation persisted despite randomized weights "
                    f"(Cosine Similarity: {cos_sim:.2f} > {self.max_similarity_threshold}). "
                    f"Method may act as an input edge filter rather than reflecting model reasoning."
                )

            return SanityCheckResult(
                score=round(sanity_score, 4),
                similarity_with_randomized=round(cos_sim, 4),
                spearman_with_randomized=round(spearman_rho, 4),
                randomized_layers=randomized_found,
                status=status,
                evidence=evidence,
                limitation=limitation
            )
        finally:
            # Enforce 100% parameter restoration guarantee
            original_model.load_state_dict(original_state_backup)
