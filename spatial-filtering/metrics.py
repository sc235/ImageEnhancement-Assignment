import numpy as np
import cv2

def calculate_stats(image: np.ndarray) -> dict:
    """Calculates Mean, Standard Deviation (Contrast), and Shannon Entropy."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    
    # Shannon Entropy calculation
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy_val = float(-np.sum(prob * np.log2(prob)))
    
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "entropy": round(entropy_val, 2)
    }

def calculate_mse(reference: np.ndarray, target: np.ndarray) -> float:
    """Calculates Mean Squared Error (MSE) between reference and target image."""
    ref_g = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    tgt_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    
    mse = np.mean((ref_g.astype(np.float64) - tgt_g.astype(np.float64)) ** 2)
    return round(float(mse), 2)

def calculate_psnr(reference: np.ndarray, target: np.ndarray) -> str:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = calculate_mse(reference, target)
    if mse == 0:
        return "inf"
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return f"{psnr:.2f}"
