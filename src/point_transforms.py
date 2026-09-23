import cv2
import numpy as np

def apply_lut(img, f):
    """
    Applies a point transformation T(r) using a 256-entry lookup table (LUT)
    for high computational efficiency (Slide 12 & 28).
    """
    lut = np.array([np.clip(f(r), 0, 255) for r in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def transform_negative(img):
    """s = 255 - r (Slide 13)"""
    return apply_lut(img, lambda r: 255 - r)

def transform_linear(img, alpha=1.5, beta=-30):
    """s = alpha * r + beta, with clipping [0, 255] (Slide 14)"""
    return apply_lut(img, lambda r: alpha * r + beta)

def transform_minmax_stretch(img):
    """Ordinary Min-Max Contrast Stretch (Slide 16)"""
    r_min, r_max = int(np.min(img)), int(np.max(img))
    if r_max == r_min:
        return img.copy()
    scale = 255.0 / (r_max - r_min)
    return apply_lut(img, lambda r: (int(r) - r_min) * scale)

def transform_percentile_stretch(img, p_low=1, p_high=99):
    """Robust Percentile Contrast Stretch (Slide 18-19 & 28)"""
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi == lo:
        return img.copy()
    img_float = img.astype(np.float32)
    stretched = np.clip((img_float - lo) * 255.0 / (hi - lo), 0, 255)
    return stretched.astype(np.uint8)

def transform_log(img):
    """
    Logarithmic Transform: s = c * log(1 + r) (Slide 20-21)
    Expands dark tones and compresses bright tones.
    c = 255 / log(256) ~ 105.9 for natural log (or log10).
    """
    c = 255.0 / np.log(256.0)
    return apply_lut(img, lambda r: c * np.log(1.0 + r))

def transform_gamma(img, gamma=0.5):
    """
    Gamma Correction (Power Law): s = 255 * (r / 255)^gamma (Slide 22-24)
    gamma < 1: brightens shadows and expands dark tones.
    gamma > 1: darkens over-exposed highlights.
    """
    return apply_lut(img, lambda r: 255.0 * ((r / 255.0) ** gamma))

def transform_threshold(img, T=128):
    """Binarization / Thresholding: s = 255 if r > T else 0 (Slide 25)"""
    return apply_lut(img, lambda r: 255 if r > T else 0)

def extract_bit_planes(img):
    """
    Bit-Plane Slicing: Decomposes 8-bit image into 8 binary images (Slide 26-27).
    Returns a dictionary of bit plane images {"Bit 7": img_bit7, ...}
    """
    planes = {}
    # Convert to grayscale if image is color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    for bit in range(8):
        # Extract bit using bitwise AND and scale to 0 or 255 for visual rendering
        plane = np.bitwise_and(gray, 1 << bit)
        plane_visual = ((plane > 0) * 255).astype(np.uint8)
        planes[f"Bit {bit} (val {1<<bit})"] = plane_visual
        
    return planes
