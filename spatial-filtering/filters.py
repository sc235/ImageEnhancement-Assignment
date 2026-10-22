import numpy as np
import cv2

# =====================================================================
# 1. MEAN FILTER (BOX FILTER) - LINEAR SMOOTHING
# =====================================================================
def create_mean_kernel(kernel_size: int) -> np.ndarray:
    """Creates a normalized square mean (box) kernel of size (ksize x ksize)."""
    return np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

def apply_mean_filter_scratch(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Manual implementation of Mean (Box) filter using 2D spatial correlation."""
    kernel = create_mean_kernel(kernel_size)
    return custom_convolve2d(image, kernel)

def apply_mean_filter_cv(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """OpenCV implementation of Mean Filter."""
    return cv2.blur(image, (kernel_size, kernel_size))

# =====================================================================
# 2. GAUSSIAN FILTER - WEIGHTED LINEAR SMOOTHING
# =====================================================================
def create_gaussian_kernel_2d(kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Generates a 2D Gaussian Kernel centered at (0,0).
    Formula: h(x,y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    """
    pad = kernel_size // 2
    x, y = np.mgrid[-pad:pad+1, -pad:pad+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return g / g.sum()

def apply_gaussian_filter_scratch(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Manual 2D Gaussian filter using constructed 2D Gaussian kernel."""
    kernel = create_gaussian_kernel_2d(kernel_size, sigma)
    return custom_convolve2d(image, kernel)

def apply_gaussian_filter_cv(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """OpenCV Gaussian Blur implementation."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

# =====================================================================
# 3. MEDIAN FILTER - NON-LINEAR ORDER-STATISTIC FILTER
# =====================================================================
def apply_median_filter_scratch(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Manual implementation of Median Filter.
    Ranks neighboring pixel intensities in ascending order and picks the middle (median) value.
    """
    if image.ndim == 3:
        channels = [apply_median_filter_scratch(image[:, :, c], kernel_size) for c in range(3)]
        return np.stack(channels, axis=2)
        
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.uint8)
    
    for r in range(h):
        for c in range(w):
            window = padded[r:r+kernel_size, c:c+kernel_size]
            output[r, c] = np.median(window)
            
    return output

def apply_median_filter_cv(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """OpenCV Median Blur implementation."""
    return cv2.medianBlur(image, kernel_size)

# =====================================================================
# 4. SOBEL FILTER - 1ST DERIVATIVE EDGE DETECTION
# =====================================================================
def apply_sobel_filter(image: np.ndarray, ksize: int = 3):
    """
    Applies 1st derivative Sobel operator in X and Y directions.
    
    Returns:
        Gx (np.ndarray): Horizontal gradient response.
        Gy (np.ndarray): Vertical gradient response.
        magnitude (np.ndarray): Edge magnitude M = sqrt(Gx^2 + Gy^2), normalized to [0, 255].
        direction (np.ndarray): Edge orientation angle theta = arctan2(Gy, Gx) in degrees.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    
    # Compute 1st derivatives with 64-bit float to handle negative values and prevent overflow
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Gradient Magnitude and Orientation Angle
    magnitude = np.hypot(Gx, Gy)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    angle = np.arctan2(Gy, Gx) * (180.0 / np.pi) # [-180, 180] degrees
    
    # Convert Gx and Gy to displayable uint8 (absolute value scaling)
    Gx_abs = cv2.convertScaleAbs(Gx)
    Gy_abs = cv2.convertScaleAbs(Gy)
    
    return Gx_abs, Gy_abs, magnitude_norm, angle

# =====================================================================
# 5. LAPLACIAN FILTER - 2ND DERIVATIVE SHARPENING
# =====================================================================
def apply_laplacian_filter(image: np.ndarray, ksize: int = 3, alpha: float = 1.0):
    """
    Applies 2nd derivative Laplacian operator and performs image sharpening:
    g(x,y) = f(x,y) - alpha * Laplacian(f(x,y))
    
    Returns:
        laplacian_norm (np.ndarray): Displayable 8-bit Laplacian response map.
        sharpened (np.ndarray): Sharpened image result (uint8).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    
    # 2nd derivative calculation
    laplacian_raw = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    
    # Displayable laplacian response map
    laplacian_norm = cv2.convertScaleAbs(laplacian_raw)
    
    # Sharpening: subtract 2nd derivative (when center is negative)
    sharpened_float = gray.astype(np.float64) - alpha * laplacian_raw
    sharpened = np.clip(sharpened_float, 0, 255).astype(np.uint8)
    
    return laplacian_norm, sharpened

# =====================================================================
# HELPER: CUSTOM CONVOLUTION (REPLICATE PADDING)
# =====================================================================
def custom_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Executes 2D spatial convolution/correlation with border replicate padding."""
    if image.ndim == 3:
        channels = [custom_convolve2d(image[:, :, c], kernel) for c in range(3)]
        return np.stack(channels, axis=2)
        
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='edge').astype(np.float32)
    h, w = image.shape
    kh, kw = kernel.shape
    
    # Flip kernel for standard convolution (symmetric kernels like mean/Gaussian stay identical)
    k_flipped = np.flipud(np.fliplr(kernel))
    
    output = np.zeros((h, w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            region = padded[r:r+kh, c:c+kw]
            output[r, c] = np.sum(region * k_flipped)
            
    return np.clip(output, 0, 255).astype(np.uint8)
