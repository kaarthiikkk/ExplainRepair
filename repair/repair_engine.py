"""
Adaptive Repair Engine.
Executes closed-loop detection, diagnosis, candidate generation, evaluation,
and selection of repaired explanations without modifying the underlying model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import torch

from config import AppConfig, default_config
from explainers.base_explainer import BaseExplainer
from explainers.explainer_factory import ExplainerFactory
from models.predictor import ModelPredictor
from reliability.reliability_evaluator import ReliabilityEvaluator, ReliabilityReport
from reliability.insertion import InsertionEvaluator, InsertionResult
from repair.diagnosis import DiagnosticEngine, FailureMode
from repair.validation import RepairValidator


@dataclass
class CandidateEvaluationSummary:
    name: str
    composite_score: float
    stability_score: float
    faithfulness_score: float
    is_eligible: bool
    rejection_reason: Optional[str] = None


@dataclass
class RepairResult:
    initial_explainer_name: str
    initial_attribution: np.ndarray
    initial_report: ReliabilityReport
    selected_explainer_name: str
    selected_attribution: np.ndarray
    selected_report: ReliabilityReport
    outcome: str  # 'SUCCESS', 'PARTIAL_IMPROVEMENT', 'UNSUCCESSFUL', 'NO_REPAIR_NEEDED'
    diagnosed_failures: List[str]
    delta_composite: float
    delta_stability: float
    delta_faithfulness: float
    candidates_evaluated: List[CandidateEvaluationSummary]
    rationale: str
    validation_status: Dict[str, Any]
    independent_validation_initial: Optional[InsertionResult] = None
    independent_validation_selected: Optional[InsertionResult] = None
    delta_validation: float = 0.0
    validation_outcome: str = "NO_REPAIR_NEEDED"  # 'VALIDATED_IMPROVEMENT', 'SELECTION_IMPROVED_ONLY', 'VALIDATION_FAILED', 'NO_REPAIR_NEEDED'
    validation_disclaimer: str = (
        "Independent Proxy Disclaimer: Insertion AUC evaluates causal feature sufficiency independently from deletion-based "
        "candidate selection. It is a secondary proxy metric, not absolute ground-truth proof."
    )


class AdaptiveRepairEngine:
    """
    Closed-loop adaptive explanation repair orchestrator.
    Decouples candidate selection from independent insertion validation.
    """
    def __init__(
        self,
        predictor: ModelPredictor,
        evaluator: Optional[ReliabilityEvaluator] = None,
        insertion_evaluator: Optional[InsertionEvaluator] = None,
        config: Optional[AppConfig] = None
    ):
        self.predictor = predictor
        self.config = config or default_config
        self.evaluator = evaluator or ReliabilityEvaluator(self.config.reliability)
        seed_val = getattr(self.config.reliability, "seed", getattr(self.config.model, "seed", 42))
        self.insertion_evaluator = insertion_evaluator or InsertionEvaluator(seed=seed_val)
        self.repair_config = self.config.repair

    def repair(
        self,
        initial_explainer: BaseExplainer,
        tensor: torch.Tensor,
        target_class: Optional[int] = None,
        reference_explainer: Optional[BaseExplainer] = None
    ) -> RepairResult:
        """
        Executes the formal repair protocol.
        """
        tensor = tensor.to(self.predictor.device)
        # Step 0: Capture pre-repair model integrity snapshot
        snapshot = RepairValidator.capture_snapshot(self.predictor.model, tensor)

        # Step 1: Generate initial explanation
        initial_attr = initial_explainer.attribute(tensor, target_class=target_class)

        # Step 2: Evaluate initial explanation
        initial_report = self.evaluator.evaluate(
            explainer=initial_explainer,
            tensor=tensor,
            predictor=self.predictor,
            target_class=target_class,
            reference_explainer=reference_explainer,
            run_sanity_check=True
        )

        # Step 2b: Independent proxy validation (Insertion test) on initial explanation
        val_initial = self.insertion_evaluator.evaluate(
            explainer=initial_explainer,
            tensor=tensor,
            predictor=self.predictor,
            target_class=target_class,
            cached_attr_map=initial_attr
        )

        # Step 3: Diagnose failure modes
        failure_modes = DiagnosticEngine.diagnose(initial_report)
        diagnosed_str_list = [f.value for f in failure_modes]

        # If healthy and no defects, no repair needed
        if failure_modes == [FailureMode.HEALTHY]:
            val_res = RepairValidator.verify_integrity(snapshot, self.predictor.model, tensor)
            return RepairResult(
                initial_explainer_name=initial_explainer.name,
                initial_attribution=initial_attr,
                initial_report=initial_report,
                selected_explainer_name=initial_explainer.name,
                selected_attribution=initial_attr,
                selected_report=initial_report,
                outcome="NO_REPAIR_NEEDED",
                diagnosed_failures=diagnosed_str_list,
                delta_composite=0.0,
                delta_stability=0.0,
                delta_faithfulness=0.0,
                candidates_evaluated=[],
                rationale="Initial explanation satisfied all reliability thresholds; no repair required.",
                validation_status=val_res,
                independent_validation_initial=val_initial,
                independent_validation_selected=val_initial,
                delta_validation=0.0,
                validation_outcome="NO_REPAIR_NEEDED"
            )

        # Step 4: Generate candidate configurations
        candidate_configs = [
            cfg for cfg in self.repair_config.candidate_pool
            if cfg.get("label") != initial_explainer.name
        ][:self.repair_config.max_candidates]

        candidate_summaries: List[CandidateEvaluationSummary] = []
        best_candidate_explainer: Optional[BaseExplainer] = None
        best_candidate_attr: Optional[np.ndarray] = None
        best_candidate_report: Optional[ReliabilityReport] = None
        best_composite_score = initial_report.composite_score

        # Avoid circular agreement bias: evaluate candidates against independent reference if provided
        candidate_reference = reference_explainer if (reference_explainer is not None and reference_explainer.name != initial_explainer.name) else None

        # Step 5: Evaluate candidates using the identical protocol
        for cfg in candidate_configs:
            try:
                candidate_expl = ExplainerFactory.create_explainer(self.predictor.model, cfg)
                c_attr = candidate_expl.attribute(tensor, target_class=target_class)
                
                c_report = self.evaluator.evaluate(
                    explainer=candidate_expl,
                    tensor=tensor,
                    predictor=self.predictor,
                    target_class=target_class,
                    reference_explainer=candidate_reference,
                    run_sanity_check=True
                )

                # Step 6: Apply Predefined Selection Rule
                # Eligibility checks: candidate must not regress on diagnosed primary defect
                is_eligible = True
                rejection_reason = None

                if FailureMode.INSTABILITY in failure_modes:
                    if c_report.stability.score < initial_report.stability.score:
                        is_eligible = False
                        rejection_reason = "Failed to improve stability over initial explainer."

                if FailureMode.UNFAITHFUL in failure_modes:
                    if c_report.faithfulness.score < initial_report.faithfulness.score:
                        is_eligible = False
                        rejection_reason = "Failed to improve causal faithfulness over initial explainer."

                if FailureMode.MODEL_INSENSITIVITY in failure_modes:
                    if c_report.sanity_check and c_report.sanity_check.status == "FAIL":
                        is_eligible = False
                        rejection_reason = "Failed parameter randomization sanity check."

                candidate_summaries.append(
                    CandidateEvaluationSummary(
                        name=candidate_expl.name,
                        composite_score=c_report.composite_score,
                        stability_score=c_report.stability.score,
                        faithfulness_score=c_report.faithfulness.score,
                        is_eligible=is_eligible,
                        rejection_reason=rejection_reason
                    )
                )

                if is_eligible and c_report.composite_score > best_composite_score:
                    best_composite_score = c_report.composite_score
                    best_candidate_explainer = candidate_expl
                    best_candidate_attr = c_attr
                    best_candidate_report = c_report

            except Exception as e:
                candidate_summaries.append(
                    CandidateEvaluationSummary(
                        name=cfg.get("label", "Unknown"),
                        composite_score=0.0,
                        stability_score=0.0,
                        faithfulness_score=0.0,
                        is_eligible=False,
                        rejection_reason=f"Runtime error during evaluation: {e}"
                    )
                )

        # Step 7: Determine outcome and compile report
        delta_composite = round(best_composite_score - initial_report.composite_score, 4)

        if best_candidate_explainer is not None and best_candidate_report is not None:
            delta_stab = round(best_candidate_report.stability.score - initial_report.stability.score, 4)
            delta_faith = round(best_candidate_report.faithfulness.score - initial_report.faithfulness.score, 4)

            if delta_composite >= self.repair_config.min_improvement_delta:
                outcome = "SUCCESS"
                rationale = (
                    f"Selected {best_candidate_explainer.name} which improved composite reliability "
                    f"from {initial_report.composite_score:.2f} to {best_composite_score:.2f} (Δ: +{delta_composite:.2f}), "
                    f"successfully remediating {', '.join(diagnosed_str_list)}."
                )
            elif delta_composite >= self.repair_config.partial_improvement_delta:
                outcome = "PARTIAL_IMPROVEMENT"
                rationale = (
                    f"Selected {best_candidate_explainer.name} which moderately improved composite reliability "
                    f"(Δ: +{delta_composite:.2f}), below the full success threshold of {self.repair_config.min_improvement_delta}."
                )
            else:
                outcome = "UNSUCCESSFUL"
                rationale = "Candidate improvements were negligible; retained original explanation."
                best_candidate_explainer = initial_explainer
                best_candidate_attr = initial_attr
                best_candidate_report = initial_report
        else:
            outcome = "UNSUCCESSFUL"
            best_candidate_explainer = initial_explainer
            best_candidate_attr = initial_attr
            best_candidate_report = initial_report
            delta_stab = 0.0
            delta_faith = 0.0
            rationale = (
                f"None of the {len(candidate_configs)} alternative candidates improved reliability on the diagnosed defects "
                f"({', '.join(diagnosed_str_list)}). Initial explanation retained honestly."
            )

        # Step 7b: Independent validation on the selected explanation
        val_selected = self.insertion_evaluator.evaluate(
            explainer=best_candidate_explainer,
            tensor=tensor,
            predictor=self.predictor,
            target_class=target_class,
            cached_attr_map=best_candidate_attr
        )
        delta_val = round(val_selected.score - val_initial.score, 4)

        if outcome == "SUCCESS":
            validation_outcome = "VALIDATED_IMPROVEMENT" if delta_val > 0 else "SELECTION_IMPROVED_ONLY"
        elif outcome == "PARTIAL_IMPROVEMENT":
            validation_outcome = "PARTIAL_VALIDATION" if delta_val > 0 else "SELECTION_IMPROVED_ONLY"
        else:
            validation_outcome = "VALIDATION_FAILED"

        # Step 8: Verify model integrity
        val_res = RepairValidator.verify_integrity(snapshot, self.predictor.model, tensor)

        return RepairResult(
            initial_explainer_name=initial_explainer.name,
            initial_attribution=initial_attr,
            initial_report=initial_report,
            selected_explainer_name=best_candidate_explainer.name,
            selected_attribution=best_candidate_attr,
            selected_report=best_candidate_report,
            outcome=outcome,
            diagnosed_failures=diagnosed_str_list,
            delta_composite=delta_composite,
            delta_stability=delta_stab,
            delta_faithfulness=delta_faith,
            candidates_evaluated=candidate_summaries,
            rationale=rationale,
            validation_status=val_res,
            independent_validation_initial=val_initial,
            independent_validation_selected=val_selected,
            delta_validation=delta_val,
            validation_outcome=validation_outcome
        )
