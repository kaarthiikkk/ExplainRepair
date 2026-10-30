"""
Reliability Evaluator Module.
Combines Stability, Faithfulness, Cross-Method Agreement, and Parameter Sanity Checks
into a normalized multi-dimensional Reliability Score and structured failure diagnosis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np
import torch

from config import ReliabilityConfig, default_config
from explainers.base_explainer import BaseExplainer
from models.predictor import ModelPredictor
from reliability.stability import StabilityEvaluator, StabilityResult
from reliability.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from reliability.agreement import AgreementEvaluator, AgreementResult
from reliability.sanity_checks import ModelSanityCheckEvaluator, SanityCheckResult


@dataclass
class ReliabilityReport:
    explainer_name: str
    composite_score: float  # [0, 1], higher is more reliable
    stability: StabilityResult
    faithfulness: FaithfulnessResult
    agreement: Optional[AgreementResult] = None
    sanity_check: Optional[SanityCheckResult] = None
    status_summary: Dict[str, str] = field(default_factory=dict)
    diagnoses: List[str] = field(default_factory=list)
    recommended_action: str = ""
    weights_used: Dict[str, float] = field(default_factory=dict)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    execution_time_seconds: float = 0.0


class ReliabilityEvaluator:
    """
    Orchestrates independent reliability assessments and synthesizes a diagnostic report.
    """
    def __init__(self, config: Optional[ReliabilityConfig] = None):
        self.config = config or default_config.reliability
        
        self.stability_evaluator = StabilityEvaluator(
            samples=self.config.stability_samples,
            noise_std=self.config.noise_std,
            warning_threshold=self.config.stability_warning_threshold,
            fail_threshold=self.config.stability_fail_threshold
        )
        self.faithfulness_evaluator = FaithfulnessEvaluator(
            fractions=self.config.deletion_fractions,
            warning_threshold=self.config.faithfulness_warning_threshold,
            fail_threshold=self.config.faithfulness_fail_threshold
        )
        self.agreement_evaluator = AgreementEvaluator(
            warning_threshold=self.config.agreement_warning_threshold
        )
        self.sanity_evaluator = ModelSanityCheckEvaluator(
            max_similarity_threshold=self.config.sanity_similarity_max
        )

    def evaluate(
        self,
        explainer: BaseExplainer,
        tensor: torch.Tensor,
        predictor: ModelPredictor,
        target_class: Optional[int] = None,
        reference_explainer: Optional[BaseExplainer] = None,
        run_sanity_check: bool = True
    ) -> ReliabilityReport:
        """
        Execute full reliability battery for a given explainer and input tensor.
        """
        start_time = time.perf_counter()

        # 1. Base explanation
        base_attr = explainer.attribute(tensor, target_class=target_class)

        # 2. Evaluate Stability
        stability_res = self.stability_evaluator.evaluate(
            explainer=explainer,
            tensor=tensor,
            predictor=predictor,
            target_class=target_class
        )

        # 3. Evaluate Faithfulness
        faithfulness_res = self.faithfulness_evaluator.evaluate(
            explainer=explainer,
            tensor=tensor,
            predictor=predictor,
            target_class=target_class
        )

        # 4. Optional Cross-Method Agreement
        agreement_res = None
        if reference_explainer is not None and reference_explainer.name != explainer.name:
            agreement_res = self.agreement_evaluator.evaluate(
                explainer_a=explainer,
                explainer_b=reference_explainer,
                tensor=tensor,
                target_class=target_class,
                cached_attr_a=base_attr
            )

        # 5. Optional Model Sanity Check
        sanity_res = None
        if run_sanity_check:
            sanity_res = self.sanity_evaluator.evaluate(
                explainer=explainer,
                tensor=tensor,
                original_model=predictor.model,
                target_class=target_class,
                cached_attr_original=base_attr
            )

        # 6. Composite Score Calculation
        w_stab = self.config.weight_stability
        w_faith = self.config.weight_faithfulness
        w_agr = self.config.weight_agreement if agreement_res is not None else 0.0
        w_san = self.config.weight_sanity if sanity_res is not None else 0.0
        
        total_weight = w_stab + w_faith + w_agr + w_san
        
        weighted_sum = (
            w_stab * stability_res.score +
            w_faith * faithfulness_res.score +
            (w_agr * agreement_res.score if agreement_res else 0.0) +
            (w_san * sanity_res.score if sanity_res else 0.0)
        )
        composite_score = float(np.clip(weighted_sum / total_weight, 0.0, 1.0))

        # 7. Diagnoses and Status Summary
        diagnoses: List[str] = []
        status_summary = {
            "stability": stability_res.status,
            "faithfulness": faithfulness_res.status
        }

        if stability_res.status in ["WARNING", "FAIL"]:
            diagnoses.append(
                f"STABILITY_DEFECT: Explanation is volatile under small input perturbations "
                f"(Score: {stability_res.score:.2f} < {self.config.stability_warning_threshold})."
            )

        if faithfulness_res.status in ["WARNING", "FAIL"]:
            diagnoses.append(
                f"FAITHFULNESS_DEFECT: Deleting top attributed features fails to degrade confidence "
                f"faster than random baseline (Score: {faithfulness_res.score:.2f} < {self.config.faithfulness_warning_threshold})."
            )

        if agreement_res is not None:
            status_summary["agreement"] = agreement_res.status
            if agreement_res.status in ["WARNING", "FAIL"]:
                diagnoses.append(
                    f"CROSS_METHOD_DISAGREEMENT: Low consensus with {agreement_res.method_b} "
                    f"(Score: {agreement_res.score:.2f} < {self.config.agreement_warning_threshold})."
                )

        if sanity_res is not None:
            status_summary["sanity_check"] = sanity_res.status
            if sanity_res.status == "FAIL":
                diagnoses.append(
                    f"MODEL_INSENSITIVITY_DEFECT: Explanation persists despite model parameter randomization "
                    f"(Similarity: {sanity_res.similarity_with_randomized:.2f} > {self.config.sanity_similarity_max})."
                )

        # 8. Recommended Action
        if not diagnoses:
            recommended_action = "Explanation meets all reliability thresholds. No repair required."
        elif any("STABILITY_DEFECT" in d for d in diagnoses):
            recommended_action = "Apply noise-smoothing (SmoothGrad) or deeper layer aggregation to stabilize gradients."
        elif any("FAITHFULNESS_DEFECT" in d for d in diagnoses):
            recommended_action = "Switch to path-integrated gradients or higher-resolution intermediate layers to restore causal fidelity."
        elif any("MODEL_INSENSITIVITY_DEFECT" in d for d in diagnoses):
            recommended_action = "Reject current explainer: method acts as an edge detector; switch to parameter-sensitive path methods."
        else:
            recommended_action = "Execute adaptive repair candidate exploration."

        exec_time = time.perf_counter() - start_time

        return ReliabilityReport(
            explainer_name=explainer.name,
            composite_score=round(composite_score, 4),
            stability=stability_res,
            faithfulness=faithfulness_res,
            agreement=agreement_res,
            sanity_check=sanity_res,
            status_summary=status_summary,
            diagnoses=diagnoses,
            recommended_action=recommended_action,
            weights_used={
                "stability": w_stab,
                "faithfulness": w_faith,
                "agreement": w_agr,
                "sanity": w_san
            },
            thresholds_used={
                "stability_warning": self.config.stability_warning_threshold,
                "stability_fail": self.config.stability_fail_threshold,
                "faithfulness_warning": self.config.faithfulness_warning_threshold,
                "faithfulness_fail": self.config.faithfulness_fail_threshold,
                "agreement_warning": self.config.agreement_warning_threshold,
                "sanity_max_similarity": self.config.sanity_similarity_max,
            },
            execution_time_seconds=round(exec_time, 4)
        )
