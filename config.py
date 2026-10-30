"""
ExplainRepair: Central Configuration Module
Provides typed dataclasses for paths, model settings, explainer parameters,
reliability metric weights/thresholds, and repair strategies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_default_device() -> torch.device:
    """Detect available hardware accelerator: MPS (Apple Silicon), CUDA, or CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    architecture: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = True
    checkpoint_name: str = "resnet18_cifar10.pt"
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 5  # Quick start default
    seed: int = 42

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / self.checkpoint_name


@dataclass
class ExplainerConfig:
    default_method: str = "gradcam"  # 'gradcam' or 'integrated_gradients'
    gradcam_layer: str = "layer4"    # Target convolutional layer for ResNet-18
    ig_steps: int = 50               # Approximation steps for Integrated Gradients
    ig_baseline: str = "black"       # 'black', 'white', 'blur', 'random'
    smooth_grad_samples: int = 10    # Noise samples for SmoothGrad
    smooth_grad_stdevs: float = 0.15 # Noise level for SmoothGrad


@dataclass
class ReliabilityConfig:
    # Stability test parameters
    stability_samples: int = 5
    perturbation_types: List[str] = field(
        default_factory=lambda: ["gaussian_noise", "slight_blur", "brightness_jitter"]
    )
    noise_std: float = 0.05
    blur_kernel_size: int = 3
    brightness_delta: float = 0.10

    # Faithfulness test parameters (pixel/patch deletion)
    deletion_fractions: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]
    )
    deletion_mode: Literal["patch", "pixel"] = "pixel"
    patch_size: int = 4
    mask_strategy: Literal["zero", "mean", "blur"] = "mean"

    # Sanity check parameters (parameter randomization)
    sanity_randomize_layers: List[str] = field(
        default_factory=lambda: ["fc", "layer4.1.conv2"]
    )

    # Thresholds for failure diagnosis (configurable experimental defaults)
    stability_warning_threshold: float = 0.60    # Score in [0, 1]; <0.60 triggers instability warning
    stability_fail_threshold: float = 0.40
    faithfulness_warning_threshold: float = 0.50 # Normalized drop AUC; <0.50 triggers unfaithful warning
    faithfulness_fail_threshold: float = 0.35
    agreement_warning_threshold: float = 0.35    # Spatial IoU/Cosine; <0.35 triggers disagreement warning
    sanity_similarity_max: float = 0.65          # Attribution correlation with randomized model; >0.65 triggers insensitivity fail

    # Weights for composite reliability score (sum = 1.0)
    weight_stability: float = 0.35
    weight_faithfulness: float = 0.45
    weight_agreement: float = 0.10
    weight_sanity: float = 0.10


@dataclass
class RepairConfig:
    max_candidates: int = 5
    min_improvement_delta: float = 0.05  # Minimum composite gain to declare SUCCESS
    partial_improvement_delta: float = 0.01
    candidate_pool: List[Dict[str, str]] = field(
        default_factory=lambda: [
            {"method": "gradcam", "layer": "layer4", "label": "Grad-CAM (Layer 4)"},
            {"method": "gradcam", "layer": "layer3", "label": "Grad-CAM (Layer 3 - High Spatial Resolution)"},
            {"method": "integrated_gradients", "baseline": "black", "steps": "50", "label": "Integrated Gradients (Black Baseline, 50 Steps)"},
            {"method": "integrated_gradients", "baseline": "blur", "steps": "50", "label": "Integrated Gradients (Blurred Baseline, 50 Steps)"},
            {"method": "integrated_gradients", "baseline": "white", "steps": "50", "label": "Integrated Gradients (White Baseline, 50 Steps)"},
            {"method": "smooth_gradcam", "layer": "layer4", "label": "Smoothed Grad-CAM (Noise Tunnel)"},
        ]
    )


@dataclass
class ExperimentConfig:
    sample_size: int = 50
    seed: int = 42
    results_dir: Path = RESULTS_DIR
    save_raw: bool = True
    save_aggregates: bool = True
    batch_size: int = 16


@dataclass
class AppConfig:
    title: str = "ExplainRepair"
    subtitle: str = "An Adaptive Framework for Evaluating and Repairing Unreliable AI Explanations"
    model: ModelConfig = field(default_factory=ModelConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    device: torch.device = field(default_factory=get_default_device)


# Singleton instance
default_config = AppConfig()
