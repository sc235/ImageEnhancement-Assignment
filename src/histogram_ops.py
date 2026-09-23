import cv2
import numpy as np

def global_histogram_equalization(img):
    """
    Global Histogram Equalization (Slide 30-34).
    Uses CDF to flatten the intensity distribution and maximize overall contrast.
    If image is RGB/BGR, converts to YCrCb or Lab to avoid color distortion.
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast-Limited Adaptive Histogram Equalization (CLAHE) (Slide 40-45).
    Crucial Rule from Slide 43:
    'NEVER apply to R, G, B separately. Convert to Lab, enhance L channel ONLY, and convert back.'
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        enhanced_lab = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)
