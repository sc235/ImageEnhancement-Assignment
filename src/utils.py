import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_stats(img):
    """
    Computes statistical properties of an image channel/grayscale image:
    Mean, Standard Deviation (Contrast), Range [min, max], and Entropy (in bits).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    min_val, max_val = np.min(gray), np.max(gray)
    
    # Entropy calculation: H = -sum(p * log2(p))
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob = hist / float(gray.size)
    prob = prob[prob > 0]
    entropy_val = -np.sum(prob * np.log2(prob))
    
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": int(min_val),
        "max": int(max_val),
        "entropy": round(entropy_val, 2)
    }

def calculate_psnr(original, processed):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB between two images."""
    if len(original.shape) == 3 and len(processed.shape) == 3:
        orig_g = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_g = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        orig_g = original
        proc_g = processed
        
    mse = np.mean((orig_g.astype(np.float64) - proc_g.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(psnr, 2)

def save_comparison_figure(image_dict, title, filename):
    """
    Saves a grid figure displaying images and their corresponding intensity histograms.
    image_dict format: {"Label": image_bgr_or_gray, ...}
    """
    n = len(image_dict)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    if n == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i, (label, img) in enumerate(image_dict.items()):
        # Display image
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img_rgb)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
            gray = img
            
        stats = calculate_stats(gray)
        axes[0, i].set_title(f"{label}\nMean:{stats['mean']} Std:{stats['std']}\nRange:[{stats['min']},{stats['max']}] H:{stats['entropy']}b", fontsize=9)
        axes[0, i].axis('off')
        
        # Display histogram
        axes[1, i].hist(gray.ravel(), bins=256, range=(0, 256), color='navy', alpha=0.7)
        axes[1, i].set_xlim([0, 255])
        axes[1, i].set_xlabel("Intensity")
        if i == 0:
            axes[1, i].set_ylabel("Pixel Count")
        axes[1, i].grid(True, linestyle='--', alpha=0.5)
        
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
