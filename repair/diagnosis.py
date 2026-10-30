"""
Explanation Failure Diagnosis Module.
Classifies empirical reliability flaws into actionable defect categories.
"""

from enum import Enum
from typing import List, Dict, Any
from reliability.reliability_evaluator import ReliabilityReport


class FailureMode(str, Enum):
    HEALTHY = "HEALTHY"
    INSTABILITY = "INSTABILITY"
    UNFAITHFUL = "UNFAITHFUL"
    CROSS_METHOD_DISAGREEMENT = "CROSS_METHOD_DISAGREEMENT"
    MODEL_INSENSITIVITY = "MODEL_INSENSITIVITY"


class DiagnosticEngine:
    """
    Parses reliability reports and determines primary and secondary failure modes.
    """
    @staticmethod
    def diagnose(report: ReliabilityReport) -> List[FailureMode]:
        failures: List[FailureMode] = []
        
        # 1. Sanity Check / Insensitivity takes precedence
        if report.sanity_check and report.sanity_check.status == "FAIL":
            failures.append(FailureMode.MODEL_INSENSITIVITY)

        # 2. Faithfulness defect
        if report.faithfulness.status in ["WARNING", "FAIL"]:
            failures.append(FailureMode.UNFAITHFUL)

        # 3. Instability defect
        if report.stability.status in ["WARNING", "FAIL"]:
            failures.append(FailureMode.INSTABILITY)

        # 4. Cross method disagreement
        if report.agreement and report.agreement.status in ["WARNING", "FAIL"]:
            failures.append(FailureMode.CROSS_METHOD_DISAGREEMENT)

        if not failures:
            failures.append(FailureMode.HEALTHY)

        return failures

    @staticmethod
    def get_repair_priorities(failures: List[FailureMode]) -> List[str]:
        """
        Returns prioritized strategies based on diagnosed defects.
        """
        strategies = []
        for f in failures:
            if f == FailureMode.MODEL_INSENSITIVITY:
                strategies.extend(["integrated_gradients", "smooth_gradcam"])
            elif f == FailureMode.INSTABILITY:
                strategies.extend(["smooth_gradcam", "integrated_gradients"])
            elif f == FailureMode.UNFAITHFUL:
                strategies.extend(["gradcam_layer3", "integrated_gradients"])
            elif f == FailureMode.CROSS_METHOD_DISAGREEMENT:
                strategies.extend(["integrated_gradients", "gradcam"])
        return list(dict.fromkeys(strategies))  # preserve order, unique
