"""
Image loading, normalization, transformation, and perturbation utilities.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance

from config import CIFAR10_MEAN, CIFAR10_STD, CIFAR10_CLASSES


def get_cifar10_transforms(train: bool = False) -> transforms.Compose:
    """Returns standard torchvision transforms for CIFAR-10."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def preprocess_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (32, 32)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess a PIL image or numpy array into:
    1. A normalized torch.Tensor of shape [1, 3, H, W]
    2. A clean RGB numpy array [H, W, 3] in range [0, 1] for visualization.
    """
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # Resize to CIFAR-10 resolution
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    rgb_array = np.array(pil_resized, dtype=np.float32) / 255.0

    # Convert to normalized tensor
    transform = get_cifar10_transforms(train=False)
    tensor = transform(pil_resized).unsqueeze(0)  # Shape [1, 3, 32, 32]
    return tensor, rgb_array


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor [1, 3, H, W] or [3, H, W]
    back to an RGB numpy array [H, W, 3] with values in [0, 1].
    """
    t = tensor.detach().cpu().clone()
    if t.ndim == 4:
        t = t.squeeze(0)
    
    # Reverse normalization: (t * std) + mean
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    
    # Convert [3, H, W] -> [H, W, 3]
    return t.permute(1, 2, 0).numpy()


def overlay_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> np.ndarray:
    """
    Overlay attribution heatmap onto an RGB image.
    rgb_image: [H, W, 3] with values in [0, 1]
    heatmap: [H, W] with values in [0, 1]
    Returns blended RGB image [H, W, 3] in [0, 1].
    """
    import matplotlib
    import matplotlib.cm as cm

    # Ensure heatmap is 2D and normalized
    heatmap_norm = np.clip(heatmap, 0.0, 1.0)
    
    # Apply colormap (compatible with Matplotlib 3.8+)
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps.get(colormap_name, matplotlib.colormaps["jet"])
    elif hasattr(cm, "get_cmap"):
        cmap = cm.get_cmap(colormap_name)
    else:
        cmap = cm.jet
    colored_heatmap = cmap(heatmap_norm)[..., :3]  # Drop alpha channel -> [H, W, 3]
    
    # Resize heatmap if shape mismatch
    if colored_heatmap.shape[:2] != rgb_image.shape[:2]:
        from PIL import Image as PILImage
        h_img = PILImage.fromarray((colored_heatmap * 255).astype(np.uint8))
        h_img = h_img.resize((rgb_image.shape[1], rgb_image.shape[0]), PILImage.Resampling.BILINEAR)
        colored_heatmap = np.array(h_img, dtype=np.float32) / 255.0

    blended = (1.0 - alpha) * rgb_image + alpha * colored_heatmap
    return np.clip(blended, 0.0, 1.0)


def apply_perturbation(
    tensor: torch.Tensor,
    perturbation_type: str,
    magnitude: float = 0.05,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply a controlled, reproducible perturbation to an image tensor [1, 3, H, W].
    Supported types: 'gaussian_noise', 'slight_blur', 'brightness_jitter'.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    perturbed = tensor.clone()

    if perturbation_type == "gaussian_noise":
        noise = torch.randn_like(perturbed) * magnitude
        perturbed = perturbed + noise
    elif perturbation_type == "slight_blur":
        # Convert to numpy, apply PIL Gaussian blur, convert back
        rgb = denormalize_tensor(perturbed)
        pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
        # Radius corresponds to magnitude
        radius = max(0.5, magnitude * 5.0)
        blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        perturbed, _ = preprocess_image(blurred_pil, target_size=(perturbed.shape[2], perturbed.shape[3]))
        perturbed = perturbed.to(tensor.device)
    elif perturbation_type == "brightness_jitter":
        rgb = denormalize_tensor(perturbed)
        pil_img = Image.fromarray((rgb * 255).astype(np.uint8))
        enhancer = ImageEnhance.Brightness(pil_img)
        # Factor: 1.0 + magnitude or 1.0 - magnitude
        factor = 1.0 + magnitude
        bright_pil = enhancer.enhance(factor)
        perturbed, _ = preprocess_image(bright_pil, target_size=(perturbed.shape[2], perturbed.shape[3]))
        perturbed = perturbed.to(tensor.device)
    return perturbed


def apply_custom_perturbation(
    tensor: torch.Tensor,
    noise_std: float = 0.0,
    blur_radius: float = 0.0,
    brightness_factor: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Apply simultaneous or individual controlled perturbations (noise, blur, brightness)
    to a normalized image tensor [1, 3, H, W].
    Returns (perturbed_tensor, perturbed_rgb_array).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 1. Denormalize to PIL Image for spatial/color transformations
    rgb = denormalize_tensor(tensor)
    pil_img = Image.fromarray((rgb * 255).astype(np.uint8))

    # 2. Brightness adjustment
    if abs(brightness_factor - 1.0) > 1e-3:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(max(0.1, float(brightness_factor)))

    # 3. Gaussian Blur
    if blur_radius > 0.01:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    # 4. Convert back to normalized tensor
    perturbed_tensor, perturbed_rgb = preprocess_image(pil_img, target_size=(tensor.shape[2], tensor.shape[3]))
    perturbed_tensor = perturbed_tensor.to(tensor.device)

    # 5. Add additive Gaussian noise in tensor space if specified
    if noise_std > 1e-4:
        noise = torch.randn_like(perturbed_tensor) * float(noise_std)
        perturbed_tensor = perturbed_tensor + noise
        perturbed_rgb = denormalize_tensor(perturbed_tensor)

    return perturbed_tensor, perturbed_rgb

