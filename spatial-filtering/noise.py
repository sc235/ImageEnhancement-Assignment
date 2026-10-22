import numpy as np
import cv2

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25.0) -> np.ndarray:
    """
    Adds Gaussian additive noise to an image.
    
    Parameters:
        image (np.ndarray): Input grayscale or BGR image (uint8).
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.
        
    Returns:
        np.ndarray: Noisy image (uint8, clipped to [0, 255]).
    """
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
    """
    Adds Salt-and-Pepper (impulse) noise to an image.
    
    Parameters:
        image (np.ndarray): Input grayscale or BGR image (uint8).
        salt_prob (float): Probability of white pixels (salt).
        pepper_prob (float): Probability of black pixels (pepper).
        
    Returns:
        np.ndarray: Noisy image (uint8).
    """
    noisy = image.copy()
    
    # Salt noise (White pixels = 255)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    if image.ndim == 2:
        noisy[tuple(coords)] = 255
    else:
        noisy[tuple(coords)] = [255, 255, 255]

    # Pepper noise (Black pixels = 0)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    if image.ndim == 2:
        noisy[tuple(coords)] = 0
    else:
        noisy[tuple(coords)] = [0, 0, 0]

    return noisy
