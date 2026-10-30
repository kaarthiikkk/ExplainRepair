"""
Publication-quality visualization helpers for explanations,
reliability metrics, deletion curves, and repair comparisons.
"""

from typing import List, Optional, Tuple, Dict
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Streamlit & scripts
import matplotlib.pyplot as plt
import numpy as np

from utils.image_utils import overlay_heatmap


def plot_explanation_overview(
    rgb_image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    predicted_class: str,
    confidence: float,
    colormap_name: str = "jet"
) -> plt.Figure:
    """
    Plot the original image alongside multiple attribution heatmaps.
    attributions: Dict of {method_label: 2D attribution array}
    """
    n_methods = len(attributions)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(3.8 * (n_methods + 1), 3.8))
    
    if n_methods == 0:
        axes = [axes]
    
    # 1. Original Image
    axes[0].imshow(rgb_image)
    axes[0].set_title(f"Input\nPred: {predicted_class} ({confidence*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # Heatmaps
    for idx, (label, attr_map) in enumerate(attributions.items(), start=1):
        blended = overlay_heatmap(rgb_image, attr_map, alpha=0.55, colormap_name=colormap_name)
        axes[idx].imshow(blended)
        axes[idx].set_title(f"{label}", fontsize=10)
        axes[idx].axis("off")
        
    plt.tight_layout()
    return fig


def plot_deletion_curve(
    fractions: List[float],
    attr_scores: List[float],
    random_scores: List[float],
    auc_attr: float,
    auc_random: float,
    method_name: str = "Explainer"
) -> plt.Figure:
    """
    Plot target-class confidence drop as salient pixels are deleted.
    A faster drop (lower AUC of confidence curve, or higher Faithfulness AUC)
    indicates the explainer correctly identified decisive features.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    
    # Percentage deleted
    x_pct = [f * 100 for f in fractions]
    
    ax.plot(x_pct, attr_scores, marker="o", linewidth=2.2, color="#1f77b4", label=f"{method_name} Deletion (AUC: {auc_attr:.3f})")
    ax.plot(x_pct, random_scores, marker="s", linestyle="--", linewidth=1.8, color="#7f7f7f", label=f"Random Baseline (AUC: {auc_random:.3f})")
    
    # Highlight gap (Faithfulness margin)
    ax.fill_between(x_pct, attr_scores, random_scores, where=np.array(random_scores) >= np.array(attr_scores),
                    color="#2ca02c", alpha=0.18, label="Faithfulness Advantage")
    
    ax.set_xlabel("Percentage of Most Salient Pixels Masked (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Target Class Model Confidence", fontsize=10, fontweight="bold")
    ax.set_title("Faithfulness: Pixel Deletion Test", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    return fig


def plot_repair_comparison(
    rgb_image: np.ndarray,
    initial_name: str,
    initial_map: np.ndarray,
    initial_score: float,
    repaired_name: str,
    repaired_map: np.ndarray,
    repaired_score: float,
    repair_status: str
) -> plt.Figure:
    """
    Side-by-side comparison of explanation before and after adaptive repair.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    
    # Input
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # Before repair
    initial_blend = overlay_heatmap(rgb_image, initial_map, alpha=0.55)
    axes[1].imshow(initial_blend)
    axes[1].set_title(f"Initial: {initial_name}\nReliability: {initial_score:.2f}", fontsize=10, color="#d62728" if initial_score < 0.6 else "#333")
    axes[1].axis("off")
    
    # After repair
    status_color = "#2ca02c" if "SUCCESS" in repair_status else ("#ff7f0e" if "PARTIAL" in repair_status else "#d62728")
    repaired_blend = overlay_heatmap(rgb_image, repaired_map, alpha=0.55)
    axes[2].imshow(repaired_blend)
    axes[2].set_title(f"Repaired: {repaired_name}\nReliability: {repaired_score:.2f} [{repair_status}]", fontsize=10, color=status_color, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    return fig


def plot_metric_breakdown_bar(metrics_dict: Dict[str, float], thresholds_dict: Dict[str, float]) -> plt.Figure:
    """
    Horizontal bar chart showing reliability metrics against their warning thresholds.
    """
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    y_pos = np.arange(len(names))
    colors = []
    
    for name, val in zip(names, values):
        thresh = thresholds_dict.get(name, 0.5)
        if val >= thresh:
            colors.append("#2ca02c")  # Pass
        elif val >= thresh - 0.15:
            colors.append("#ff7f0e")  # Warning
        else:
            colors.append("#d62728")  # Fail
            
    bars = ax.barh(y_pos, values, color=colors, height=0.55, alpha=0.85)
    
    # Add threshold markers
    for idx, name in enumerate(names):
        thresh = thresholds_dict.get(name, 0.5)
        ax.plot([thresh, thresh], [idx - 0.32, idx + 0.32], color="#333", linestyle="--", linewidth=1.5)
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Reliability Score (Higher is Better, Range [0, 1])", fontsize=10, fontweight="bold")
    ax.set_title("Reliability Dimensions vs. Operational Thresholds", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2ca02c", lw=4, label="Healthy (>= Threshold)"),
        Line2D([0], [0], color="#ff7f0e", lw=4, label="Warning"),
        Line2D([0], [0], color="#d62728", lw=4, label="Defect / Fail"),
        Line2D([0], [0], color="#333", lw=1.5, linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_stress_test_comparison(
    orig_rgb: np.ndarray,
    pert_rgb: np.ndarray,
    orig_label: str,
    orig_conf: float,
    pert_label: str,
    pert_conf: float,
    orig_gc: np.ndarray,
    pert_gc: np.ndarray,
    orig_ig: np.ndarray,
    pert_ig: np.ndarray,
    gc_sim: float,
    ig_sim: float
) -> plt.Figure:
    """
    Side-by-side stress-test matrix comparing original and perturbed images
    alongside their corresponding Grad-CAM and Integrated Gradients heatmaps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))

    # Row 0: Original baseline
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title(f"Original Input\nPred: {orig_label} ({orig_conf*100:.1f}%)", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    gc_orig_overlay = overlay_heatmap(orig_rgb, orig_gc, alpha=0.55)
    axes[0, 1].imshow(gc_orig_overlay)
    axes[0, 1].set_title("Grad-CAM (Original)", fontsize=10)
    axes[0, 1].axis("off")

    ig_orig_overlay = overlay_heatmap(orig_rgb, orig_ig, alpha=0.55)
    axes[0, 2].imshow(ig_orig_overlay)
    axes[0, 2].set_title("Integrated Gradients (Original)", fontsize=10)
    axes[0, 2].axis("off")

    # Row 1: Perturbed stress-test
    axes[1, 0].imshow(pert_rgb)
    pert_title_color = "#2ca02c" if pert_label == orig_label else "#d62728"
    axes[1, 0].set_title(f"Perturbed Input\nPred: {pert_label} ({pert_conf*100:.1f}%)", fontsize=10, fontweight="bold", color=pert_title_color)
    axes[1, 0].axis("off")

    gc_pert_overlay = overlay_heatmap(pert_rgb, pert_gc, alpha=0.55)
    axes[1, 1].imshow(gc_pert_overlay)
    axes[1, 1].set_title(f"Grad-CAM (Perturbed)\nStability Cosine: {gc_sim:.3f}", fontsize=10)
    axes[1, 1].axis("off")

    ig_pert_overlay = overlay_heatmap(pert_rgb, pert_ig, alpha=0.55)
    axes[1, 2].imshow(ig_pert_overlay)
    axes[1, 2].set_title(f"Integrated Gradients (Perturbed)\nStability Cosine: {ig_sim:.3f}", fontsize=10)
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig

