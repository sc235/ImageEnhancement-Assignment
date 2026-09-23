import cv2
import numpy as np

def infect_low_contrast(img, target_min=20, target_max=70):
    """
    Infects the image by crushing its dynamic range into a narrow band [target_min, target_max].
    This simulates low-contrast / muddy imaging conditions (like Image I in Slide 5).
    """
    img_float = img.astype(np.float32)
    orig_min, orig_max = np.min(img_float), np.max(img_float)
    if orig_max == orig_min:
        return img.copy()
    
    # Scale range from [orig_min, orig_max] to [target_min, target_max]
    scaled = (img_float - orig_min) / (orig_max - orig_min) * (target_max - target_min) + target_min
    return np.clip(scaled, 0, 255).astype(np.uint8)

def infect_low_light(img, factor=0.15):
    """
    Infects the image by severe darkening (low-light / underexposed condition).
    Details hide in dark shadows, requiring Log transform or Gamma < 1.
    """
    darkened = img.astype(np.float32) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)

def infect_salt_and_pepper(img, amount=0.04):
    """
    Infects the image with Salt-and-Pepper impulse noise.
    """
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    
    # Salt (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    noisy[tuple(coords)] = 255
    
    # Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    noisy[tuple(coords)] = 0
    return noisy

def infect_gaussian_noise(img, mean=0, sigma=25):
    """
    Infects the image with Gaussian sensor noise.
    """
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def infect_overexposed(img, shift=100, scale=1.2):
    """
    Infects the image by over-exposing / washing out highlights with clipping.
    """
    washed = img.astype(np.float32) * scale + shift
    return np.clip(washed, 0, 255).astype(np.uint8)
