"""
Report and Publication Artifact Generator.
Generates research figures and markdown synthesis from experimental results.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR, DOCS_DIR


def generate_visual_artifacts():
    csv_path = RESULTS_DIR / "experiments_raw.csv"
    json_path = RESULTS_DIR / "experiments_summary.json"

    if not csv_path.exists() or not json_path.exists():
        print("[ExplainRepair] Experiment data not found in results/. Run experiments first.")
        return

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        summary = json.load(f)

    # 1. Figure 1: Baseline Method Comparison
    fig, ax = plt.subplots(figsize=(7, 4.2))
    metrics = ["Composite", "Stability", "Faithfulness"]
    gc_vals = [
        summary["baseline_comparison"]["gradcam"]["composite_mean"],
        summary["baseline_comparison"]["gradcam"]["stability_mean"],
        summary["baseline_comparison"]["gradcam"]["faithfulness_mean"],
    ]
    ig_vals = [
        summary["baseline_comparison"]["integrated_gradients"]["composite_mean"],
        summary["baseline_comparison"]["integrated_gradients"]["stability_mean"],
        summary["baseline_comparison"]["integrated_gradients"]["faithfulness_mean"],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, gc_vals, width, label="Grad-CAM", color="#4a90e2", alpha=0.9)
    ax.bar(x + width/2, ig_vals, width, label="Integrated Gradients", color="#50e3c2", alpha=0.9)
    
    ax.set_ylabel("Reliability Metric Score [0, 1]", fontsize=10, fontweight="bold")
    ax.set_title("Experiment A: Baseline Reliability Comparison (CIFAR-10)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path1 = RESULTS_DIR / "baseline_comparison.png"
    fig.savefig(fig_path1, dpi=200)
    plt.close(fig)
    print(f"[ExplainRepair] Saved baseline comparison figure to {fig_path1}")

    # 2. Figure 2: Repair Outcome Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    outcomes = df["repair_outcome"].value_counts()
    colors = {"SUCCESS": "#2ca02c", "PARTIAL_IMPROVEMENT": "#ff7f0e", "UNSUCCESSFUL": "#d62728", "NO_REPAIR_NEEDED": "#1f77b4"}
    bar_colors = [colors.get(idx, "#888") for idx in outcomes.index]
    ax.bar(outcomes.index, outcomes.values, color=bar_colors, width=0.5, alpha=0.85)
    ax.set_ylabel("Number of Samples", fontsize=10, fontweight="bold")
    ax.set_title("Experiment D: Adaptive Repair Outcome Breakdown", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    fig_path2 = RESULTS_DIR / "repair_distribution.png"
    fig.savefig(fig_path2, dpi=200)
    plt.close(fig)
    print(f"[ExplainRepair] Saved repair distribution figure to {fig_path2}")

    # 3. Figure 3: Before-and-After Scatter Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    init_col = "initial_selection_score" if "initial_selection_score" in df.columns else "initial_score"
    rep_col = "repaired_selection_score" if "repaired_selection_score" in df.columns else "repaired_score"
    
    ax.scatter(df[init_col], df[rep_col], c="#2b5c8f", alpha=0.8, edgecolors="none", s=55, label="Test Samples")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Parity (No Change)")
    ax.set_xlabel("Initial Candidate Selection Score", fontsize=10, fontweight="bold")
    ax.set_ylabel("Repaired Candidate Selection Score", fontsize=10, fontweight="bold")
    ax.set_title("Closed-Loop Repair: Selection Score Trajectories", fontsize=11, fontweight="bold")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path3 = RESULTS_DIR / "repair_trajectories.png"
    fig.savefig(fig_path3, dpi=200)
    plt.close(fig)
    print(f"[ExplainRepair] Saved repair trajectories figure to {fig_path3}")

    # 4. Figure 4: Independent Validation Breakdown (Insertion Metric)
    if "validation_outcome" in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        val_counts = df["validation_outcome"].value_counts()
        v_colors = {
            "VALIDATED_IMPROVEMENT": "#2ca02c",
            "SELECTION_IMPROVED_ONLY": "#ff7f0e",
            "VALIDATION_FAILED": "#d62728",
            "NO_REPAIR_NEEDED": "#1f77b4",
            "INCONCLUSIVE": "#7f7f7f"
        }
        bar_colors = [v_colors.get(idx, "#888") for idx in val_counts.index]
        ax.bar(val_counts.index, val_counts.values, color=bar_colors, width=0.45, alpha=0.85)
        ax.set_ylabel("Number of Samples", fontsize=10, fontweight="bold")
        ax.set_title("Independent Validation Outcome Breakdown (Insertion Causal Metric)", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.xticks(rotation=20, ha="right", fontsize=8.5)
        plt.tight_layout()
        fig_path4 = RESULTS_DIR / "validation_outcomes.png"
        fig.savefig(fig_path4, dpi=200)
        plt.close(fig)
        print(f"[ExplainRepair] Saved independent validation figure to {fig_path4}")


if __name__ == "__main__":
    generate_visual_artifacts()
