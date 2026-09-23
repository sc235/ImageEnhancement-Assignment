import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.degradation import (
    infect_low_contrast, infect_low_light, 
    infect_salt_and_pepper, infect_gaussian_noise, infect_overexposed
)
from src.point_transforms import (
    transform_negative, transform_linear, transform_minmax_stretch,
    transform_percentile_stretch, transform_log, transform_gamma,
    transform_threshold, extract_bit_planes
)
from src.histogram_ops import global_histogram_equalization, apply_clahe
from src.spatial_ops import (
    apply_mean_filter, apply_gaussian_filter, apply_median_filter,
    apply_unsharp_masking, apply_laplacian_sharpening
)
from src.utils import calculate_stats, calculate_psnr, save_comparison_figure

def main():
    print("=" * 60)
    print(" DIGITAL IMAGE ENHANCEMENT ASSIGNMENT: FULL PIPELINE ")
    print("=" * 60)
    
    # 1. Load Original Image
    img_path = "original_image.png"
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
        
    print(f"\n[1] Original Image Loaded: {img_path} | Shape: {orig_img.shape}")
    orig_stats = calculate_stats(orig_img)
    print(f"    Stats -> Mean: {orig_stats['mean']}, Std: {orig_stats['std']}, Range: [{orig_stats['min']}, {orig_stats['max']}], Entropy: {orig_stats['entropy']} bits")
    
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/0_original.png", orig_img)
    
    # -------------------------------------------------------------
    # STEP 1: INFECTION / DEGRADATION PIPELINE
    # -------------------------------------------------------------
    print("\n[STEP 1] Infecting (degrading) original image...")
    inf_low_contrast = infect_low_contrast(orig_img, target_min=25, target_max=75)
    inf_low_light = infect_low_light(orig_img, factor=0.15)
    inf_salt_pepper = infect_salt_and_pepper(orig_img, amount=0.04)
    inf_overexposed = infect_overexposed(orig_img, shift=110, scale=1.2)
    
    save_comparison_figure({
        "Original": orig_img,
        "Infected: Low Contrast": inf_low_contrast,
        "Infected: Low Light": inf_low_light,
        "Infected: S&P Noise": inf_salt_pepper,
        "Infected: Overexposed": inf_overexposed
    }, "Step 1: Image Infection (Forward Degradation Model)", "step1_infections.png")
    
    # -------------------------------------------------------------
    # STEP 2: ENHANCING LOW CONTRAST (Point Transforms & Stretching)
    # -------------------------------------------------------------
    print("\n[STEP 2] Applying Point Operations & Contrast Stretching...")
    minmax_stretched = transform_minmax_stretch(inf_low_contrast)
    percentile_stretched = transform_percentile_stretch(inf_low_contrast, 1, 99)
    linear_boost = transform_linear(inf_low_contrast, alpha=1.8, beta=-40)
    
    save_comparison_figure({
        "Low-Contrast Input": inf_low_contrast,
        "Linear Boost (a=1.8, b=-40)": linear_boost,
        "Min-Max Stretch": minmax_stretched,
        "Percentile Stretch (1-99%)": percentile_stretched
    }, "Step 2: Low-Contrast Enhancement via Stretching", "step2_contrast_stretching.png")
    
    # -------------------------------------------------------------
    # STEP 3: ENHANCING LOW LIGHT (Log Transform & Gamma Correction)
    # -------------------------------------------------------------
    print("\n[STEP 3] Rescuing Shadow Details (Log & Gamma Correction)...")
    log_enhanced = transform_log(inf_low_light)
    gamma_04 = transform_gamma(inf_low_light, gamma=0.4)
    gamma_20 = transform_gamma(inf_overexposed, gamma=2.0)
    
    save_comparison_figure({
        "Low-Light Input": inf_low_light,
        "Log Transform s=c log(1+r)": log_enhanced,
        "Gamma Correction (γ=0.4)": gamma_04,
        "Overexposed Input": inf_overexposed,
        "Gamma Correction (γ=2.0)": gamma_20
    }, "Step 3: Non-Linear Point Transforms (Log & Gamma)", "step3_log_gamma.png")
    
    # -------------------------------------------------------------
    # STEP 4: HISTOGRAM EQUALIZATION & CLAHE
    # -------------------------------------------------------------
    print("\n[STEP 4] Applying Global Histogram Equalization & CLAHE...")
    he_enhanced = global_histogram_equalization(inf_low_contrast)
    clahe_enhanced = apply_clahe(inf_low_contrast, clip_limit=2.5, tile_grid_size=(8, 8))
    
    save_comparison_figure({
        "Low-Contrast Input": inf_low_contrast,
        "Global Equalization (HE)": he_enhanced,
        "CLAHE (Lab L-channel)": clahe_enhanced,
        "Original Target": orig_img
    }, "Step 4: Histogram Equalization vs CLAHE", "step4_histogram_clahe.png")
    
    # -------------------------------------------------------------
    # STEP 5: SPATIAL OPERATIONS (Denoising Noise Infection)
    # -------------------------------------------------------------
    print("\n[STEP 5] Applying Spatial Neighborhood Filters...")
    mean_denoised = apply_mean_filter(inf_salt_pepper, ksize=5)
    gauss_denoised = apply_gaussian_filter(inf_salt_pepper, ksize=5, sigma=1.2)
    median_denoised = apply_median_filter(inf_salt_pepper, ksize=5)
    unsharp_sharpened = apply_unsharp_masking(median_denoised, ksize=5, sigma=1.0, amount=1.5)
    
    save_comparison_figure({
        "S&P Noise Input": inf_salt_pepper,
        "Mean Filter (5x5)": mean_denoised,
        "Gaussian Filter (5x5)": gauss_denoised,
        "Median Filter (5x5)": median_denoised,
        "Median + Unsharp Mask": unsharp_sharpened
    }, "Step 5: Denoising & Sharpening Spatial Filters", "step5_spatial_filtering.png")
    
    # -------------------------------------------------------------
    # STEP 6: BIT-PLANE SLICING & THRESHOLDING
    # -------------------------------------------------------------
    print("\n[STEP 6] Performing Bit-Plane Slicing & Thresholding...")
    thresholded = transform_threshold(orig_img, T=128)
    bit_planes = extract_bit_planes(orig_img)
    
    # Save Bit-Plane Slicing Figure
    n_planes = len(bit_planes)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for idx, (b_name, b_img) in enumerate(bit_planes.items()):
        row, col = idx // 4, idx % 4
        axes[row, col].imshow(b_img, cmap='gray')
        axes[row, col].set_title(b_name, fontsize=10)
        axes[row, col].axis('off')
    plt.suptitle("Bit-Plane Slicing: 8 Binary Decomposition Planes (Slide 26-27)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/step6_bit_planes.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # METRICS SUMMARY REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" PERFORMANCE COMPARISON & METRICS TABLE ")
    print("=" * 70)
    print(f"{'Method/Image':<30} | {'Mean':<6} | {'Std (Contrast)':<14} | {'Entropy':<8} | {'PSNR (dB)':<9}")
    print("-" * 70)
    
    eval_list = [
        ("Original Reference", orig_img, orig_img),
        ("Infected: Low Contrast", inf_low_contrast, orig_img),
        ("Linear Boost (a=1.8, b=-40)", linear_boost, orig_img),
        ("Percentile Stretch (1-99%)", percentile_stretched, orig_img),
        ("Global Equalization (HE)", he_enhanced, orig_img),
        ("CLAHE (Lab L-channel)", clahe_enhanced, orig_img),
        ("Infected: Low Light", inf_low_light, orig_img),
        ("Log Transform", log_enhanced, orig_img),
        ("Gamma Correction (gamma=0.4)", gamma_04, orig_img),
        ("Infected: S&P Noise", inf_salt_pepper, orig_img),
        ("Mean Denoised (5x5)", mean_denoised, orig_img),
        ("Gaussian Denoised (5x5)", gauss_denoised, orig_img),
        ("Median Denoised (5x5)", median_denoised, orig_img),
    ]
    
    for name, img, ref in eval_list:
        st = calculate_stats(img)
        psnr_val = calculate_psnr(ref, img)
        print(f"{name:<30} | {st['mean']:<6} | {st['std']:<14} | {st['entropy']:<8} | {psnr_val:<9}")
    print("=" * 70)
    print("\nPipeline execution complete! All output plots saved to outputs/ directory.")

if __name__ == "__main__":
    main()
