import cv2
import numpy as np

def apply_mean_filter(img, ksize=5):
    """Mean (Box) Spatial Filter (Slide 3)"""
    return cv2.blur(img, (ksize, ksize))

def apply_gaussian_filter(img, ksize=5, sigma=1.0):
    """Gaussian Spatial Filter (Slide 3)"""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_median_filter(img, ksize=5):
    """Median Spatial Filter (Slide 3) - optimal for Salt-and-Pepper noise"""
    return cv2.medianBlur(img, ksize)

def apply_unsharp_masking(img, ksize=5, sigma=1.0, amount=1.5):
    """Unsharp Masking Sharpening Filter (Slide 3): I_sharp = I + amount * (I - I_blur)"""
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sharp = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def apply_laplacian_sharpening(img):
    """Laplacian Sharpening Filter (Slide 3)"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
