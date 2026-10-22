import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from noise import add_gaussian_noise, add_salt_and_pepper_noise
from filters import (
    apply_mean_filter_cv, apply_gaussian_filter_cv, apply_median_filter_cv,
    apply_sobel_filter, apply_laplacian_filter
)
from metrics import calculate_stats, calculate_mse, calculate_psnr

def main():
    print("=" * 80)
    print("   SPATIAL FILTERING DEMONSTRATION & QUANTITATIVE EVALUATION   ")
    print("   Target Filters: Mean, Gaussian, Median, Sobel, Laplacian    ")
    print("=" * 80)
    
    # 1. Load Original Image
    img_path = "../original_image.png"
    if not os.path.exists(img_path):
        img_path = "original_image.png" # Fallback if run inside root
        
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Could not load original image from path: {img_path}")
        
    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    
    print(f"\n[0] Original Image Loaded: {img_path}")
    print(f"    Dimensions: {orig_bgr.shape[1]}x{orig_bgr.shape[0]} | Channels: {orig_bgr.shape[2]}")
    orig_stats = calculate_stats(orig_gray)
    print(f"    Stats -> Mean: {orig_stats['mean']} | Std (Contrast): {orig_stats['std']} | Entropy: {orig_stats['entropy']} bits")
    
    # Output directory creation
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save original grayscale reference
    cv2.imwrite(os.path.join(out_dir, "0_original_gray.png"), orig_gray)
    
    # -----------------------------------------------------------------
    # STEP 1: GENERATE NOISY TEST IMAGES
    # -----------------------------------------------------------------
    print("\n[STEP 1] Injecting Noise for Smoothing Filter Evaluation...")
    gaussian_noisy = add_gaussian_noise(orig_gray, mean=0, sigma=25.0)
    sp_noisy = add_salt_and_pepper_noise(orig_gray, salt_prob=0.03, pepper_prob=0.03)
    
    cv2.imwrite(os.path.join(out_dir, "noisy_gaussian.png"), gaussian_noisy)
    cv2.imwrite(os.path.join(out_dir, "noisy_salt_pepper.png"), sp_noisy)
    
    # -----------------------------------------------------------------
    # STEP 2: APPLY SMOOTHING FILTERS (MEAN, GAUSSIAN, MEDIAN)
    # -----------------------------------------------------------------
    print("\n[STEP 2] Applying Smoothing Filters...")
    
    # A. Mean Filter (Box Filter)
    mean_3x3 = apply_mean_filter_cv(gaussian_noisy, kernel_size=3)
    mean_5x5 = apply_mean_filter_cv(gaussian_noisy, kernel_size=5)
    
    # B. Gaussian Filter (Weighted Low-Pass)
    gauss_5x5 = apply_gaussian_filter_cv(gaussian_noisy, kernel_size=5, sigma=1.5)
    
    # C. Median Filter (Order-Statistic Non-Linear)
    median_sp_3x3 = apply_median_filter_cv(sp_noisy, kernel_size=3)
    mean_sp_3x3 = apply_mean_filter_cv(sp_noisy, kernel_size=3) # To compare against median
    
    # Plotting Smoothing Results on Gaussian Noise
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(gaussian_noisy, cmap='gray')
    axes[0, 0].set_title("Gaussian Noise Input (\u03c3 = 25)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mean_3x3, cmap='gray')
    axes[0, 1].set_title("Mean Filter (3x3 Box)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mean_5x5, cmap='gray')
    axes[1, 0].set_title("Mean Filter (5x5 Box)", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(gauss_5x5, cmap='gray')
    axes[1, 1].set_title("Gaussian Filter (5x5, \u03c3=1.5)", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle("1. Smoothing Filters on Additive Gaussian Noise", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_smoothing_gaussian_noise.png"), dpi=150)
    plt.show()
    plt.close()
    
    # Plotting Salt-and-Pepper Comparison (Median vs Mean)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sp_noisy, cmap='gray')
    axes[0].set_title("Salt-and-Pepper Noise (6% Density)", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mean_sp_3x3, cmap='gray')
    axes[1].set_title("Mean Filter (3x3) - Blurs Noise spikes", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(median_sp_3x3, cmap='gray')
    axes[2].set_title("Median Filter (3x3) - Cleans Noise Perfectly!", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle("2. Salt-and-Pepper Noise Removal: Median Filter vs. Mean Filter", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_smoothing_salt_pepper.png"), dpi=150)
    plt.show()
    plt.close()
    
    # -----------------------------------------------------------------
    # STEP 3: APPLY SOBEL FILTER (1ST DERIVATIVE EDGE DETECTION)
    # -----------------------------------------------------------------
    print("\n[STEP 3] Applying Sobel Filter (1st Derivative Edge Detection)...")
    Gx, Gy, mag_sobel, angle_sobel = apply_sobel_filter(orig_gray, ksize=3)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(Gx, cmap='gray')
    axes[0, 0].set_title("Sobel Gx (Horizontal Derivative / Vertical Edges)", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(Gy, cmap='gray')
    axes[0, 1].set_title("Sobel Gy (Vertical Derivative / Horizontal Edges)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(mag_sobel, cmap='gray')
    axes[1, 0].set_title("Gradient Magnitude M = sqrt(Gx\u00b2 + Gy\u00b2)", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    im = axes[1, 1].imshow(angle_sobel, cmap='twilight')
    axes[1, 1].set_title("Gradient Orientation Angle \u03b8 = arctan(Gy/Gx)", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle("3. Sobel 1st Derivative Edge Detection Decomposition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_sobel_edge_detection.png"), dpi=150)
    plt.show()
    plt.close()
    
    # -----------------------------------------------------------------
    # STEP 4: APPLY LAPLACIAN FILTER (2ND DERIVATIVE SHARPENING)
    # -----------------------------------------------------------------
    print("\n[STEP 4] Applying Laplacian Filter (2nd Derivative Sharpening)...")
    laplacian_map, laplacian_sharpened = apply_laplacian_filter(orig_gray, ksize=3, alpha=0.7)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_gray, cmap='gray')
    axes[0].set_title("Original Grayscale Reference", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(laplacian_map, cmap='gray')
    axes[1].set_title("Laplacian Response |\u2207\u00b2 f| (2nd Derivative)", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(laplacian_sharpened, cmap='gray')
    axes[2].set_title("Sharpened Image: g = f - \u03b1\u2207\u00b2f", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle("4. Laplacian 2nd Derivative & Image Sharpening", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_laplacian_sharpening.png"), dpi=150)
    plt.show()
    plt.close()
    
    # -----------------------------------------------------------------
    # STEP 5: MASTER COLLAGE - ALL 5 FILTERS SIDE-BY-SIDE
    # -----------------------------------------------------------------
    print("\n[STEP 5] Generating Master Comparison Collage...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(orig_gray, cmap='gray')
    axes[0, 0].set_title("0. Original Grayscale Reference", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mean_3x3, cmap='gray')
    axes[0, 1].set_title("1. Mean Filter (3x3 Box)", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gauss_5x5, cmap='gray')
    axes[0, 2].set_title("2. Gaussian Filter (5x5, \u03c3=1.5)", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(median_sp_3x3, cmap='gray')
    axes[1, 0].set_title("3. Median Filter (3x3, Salt & Pepper)", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mag_sobel, cmap='gray')
    axes[1, 1].set_title("4. Sobel Edge Magnitude", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(laplacian_sharpened, cmap='gray')
    axes[1, 2].set_title("5. Laplacian Sharpened Result", fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle("Master Comparison: 5 Essential Spatial Filtering Techniques", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_all_5_filters_summary.png"), dpi=150)
    plt.show()
    plt.close()
    
    # -----------------------------------------------------------------
    # STEP 6: QUANTITATIVE EVALUATION TABLE
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("   SPATIAL FILTERING QUANTITATIVE EVALUATION REPORT   ")
    print("=" * 80)
    print(f"{'Filter / Method':<33} | {'Mean':<6} | {'Std (Contrast)':<14} | {'Entropy':<8} | {'MSE':<8} | {'PSNR (dB)':<9}")
    print("-" * 80)
    
    eval_list = [
        ("0. Original Reference", orig_gray, orig_gray),
        ("Infected: Gaussian Noise", gaussian_noisy, orig_gray),
        ("Infected: Salt-and-Pepper Noise", sp_noisy, orig_gray),
        ("1. Mean Filter (3x3 Box)", mean_3x3, orig_gray),
        ("1. Mean Filter (5x5 Box)", mean_5x5, orig_gray),
        ("2. Gaussian Filter (5x5, sigma=1.5)", gauss_5x5, orig_gray),
        ("3. Mean Filter on S&P Noise", mean_sp_3x3, orig_gray),
        ("3. Median Filter on S&P Noise", median_sp_3x3, orig_gray),
        ("4. Sobel Gradient Magnitude", mag_sobel, orig_gray),
        ("5. Laplacian Sharpened", laplacian_sharpened, orig_gray),
    ]
    
    for name, img, ref in eval_list:
        st = calculate_stats(img)
        mse_val = calculate_mse(ref, img)
        psnr_val = calculate_psnr(ref, img)
        print(f"{name:<33} | {st['mean']:<6} | {st['std']:<14} | {st['entropy']:<8} | {mse_val:<8} | {psnr_val:<9}")
        
    print("=" * 80)
    print(f"\nExecution Complete! Visual output grids saved to '{out_dir}/' directory.")

if __name__ == "__main__":
    main()
