"""
Mathematical and statistical metrics for attribution comparisons,
similarity, faithfulness, and rank correlation.
"""

from typing import Tuple, Union, Sequence
import numpy as np
from scipy import stats


def normalize_map(attribution: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize 2D attribution map to [0, 1].
    method: 'minmax' scales (x - min) / (max - min)
            'absmax' scales abs(x) / max(abs(x))
    """
    attr = attribution.astype(np.float64)
    if method == "minmax":
        min_v = np.min(attr)
        max_v = np.max(attr)
        if np.isclose(max_v, min_v):
            return np.zeros_like(attr)
        return (attr - min_v) / (max_v - min_v + 1e-12)
    elif method == "absmax":
        abs_v = np.abs(attr)
        max_v = np.max(abs_v)
        if np.isclose(max_v, 0.0):
            return np.zeros_like(attr)
        return abs_v / (max_v + 1e-12)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def cosine_similarity_2d(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two 2D attribution maps.
    Returns float in [-1.0, 1.0].
    """
    v1 = map1.flatten().astype(np.float64)
    v2 = map2.flatten().astype(np.float64)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if np.isclose(norm1, 0.0) or np.isclose(norm2, 0.0):
        return 0.0
    
    cos = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(cos, -1.0, 1.0))


# Alias for backward compatibility and clean importing
cosine_similarity = cosine_similarity_2d


def spearman_rank_correlation(map1: np.ndarray, map2: np.ndarray) -> float:
    """
    Calculate Spearman's rank correlation coefficient between two attribution maps.
    Returns float in [-1.0, 1.0].
    """
    v1 = map1.flatten()
    v2 = map2.flatten()
    
    if np.all(v1 == v1[0]) or np.all(v2 == v2[0]):
        return 0.0
    
    rho, _ = stats.spearmanr(v1, v2)
    if np.isnan(rho):
        return 0.0
    return float(np.clip(rho, -1.0, 1.0))


def iou_top_k(map1: np.ndarray, map2: np.ndarray, top_fraction: float = 0.20) -> float:
    """
    Calculate Intersection over Union (IoU / Jaccard Index) of the top_fraction
    most salient pixels between two attribution maps.
    Returns float in [0.0, 1.0].
    """
    v1 = map1.flatten()
    v2 = map2.flatten()
    
    k = max(1, int(len(v1) * top_fraction))
    
    # Indices of top k elements
    top1 = set(np.argpartition(v1, -k)[-k:])
    top2 = set(np.argpartition(v2, -k)[-k:])
    
    intersection = len(top1.intersection(top2))
    union = len(top1.union(top2))
    
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_auc(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute Area Under the Curve (AUC) using the trapezoidal rule.
    Supports both NumPy 2.0+ (np.trapezoid) and legacy (np.trapz).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    
    if len(x_arr) < 2:
        return 0.0
    
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_arr, x_arr))
    elif hasattr(np, "trapz"):
        return float(getattr(np, "trapz")(y_arr, x_arr))
    else:
        from scipy.integrate import trapezoid
        return float(trapezoid(y_arr, x_arr))
