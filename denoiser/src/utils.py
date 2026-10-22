import os
import cv2
import numpy as np
import torch
from PIL import Image

def load_image_rgb(path: str) -> np.ndarray:
    """
    Loads an image from disk and returns an RGB uint8 array [H, W, 3].
    Safely strips alpha transparency if RGBA.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")
    
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.uint8)

def add_gaussian_noise(img_rgb: np.ndarray, sigma: float = 25.0, seed: int = 42) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to an RGB uint8 image.
    Args:
        img_rgb: uint8 array [H, W, 3] in range [0, 255]
        sigma: noise standard deviation on [0, 255] scale (e.g. 15, 25, 50)
        seed: random seed for reproducibility
    Returns:
        noisy_img: uint8 array [H, W, 3] in range [0, 255]
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img_rgb.shape)
    noisy = img_rgb.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).round().astype(np.uint8)
    return noisy

def np2tensor(img_rgb: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts uint8 [H, W, C] in [0, 255] to float32 [1, C, H, W] in [0.0, 1.0]."""
    img_f = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).to(device)
    return tensor

def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    """Converts float32 [1, C, H, W] or [C, H, W] in [0.0, 1.0] to uint8 [H, W, C] in [0, 255]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img_np = tensor.detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = np.clip(img_np * 255.0, 0, 255).round().astype(np.uint8)
    return img_np
