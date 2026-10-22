import numpy as np
import cv2

def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE) between two RGB images [0, 255]."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10((max_val ** 2) / mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) according to Wang et al. (2004).
    Supports grayscale and 3-channel RGB uint8 arrays [0, 255].
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 3:
        ssims = []
        for c in range(img1.shape[2]):
            channel1 = img1[:, :, c]
            channel2 = img2[:, :, c]
            ssims.append(_ssim_single_channel(channel1, channel2, C1, C2))
        return float(np.mean(ssims))
    else:
        return float(_ssim_single_channel(img1, img2, C1, C2))

def _ssim_single_channel(ch1: np.ndarray, ch2: np.ndarray, C1: float, C2: float) -> float:
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(ch1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(ch2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(ch1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(ch2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(ch1 * ch2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))
