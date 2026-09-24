import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.degradation import (
    infect_low_contrast, infect_low_light, infect_overexposed
)
from src.point_transforms import (
    transform_negative, transform_linear, transform_minmax_stretch,
    transform_percentile_stretch, transform_log, transform_gamma,
    transform_threshold, extract_bit_planes
)
from src.histogram_ops import global_histogram_equalization
from src.utils import calculate_stats, calculate_psnr, save_comparison_figure

def main():
    print("=" * 70)
    print(" DIGITAL IMAGE ENHANCEMENT: SECTION 2.1 POINT OPERATIONS s = T(r) ")
    print("=" * 70)
    
    # 1. Load Original Image
    img_path = "original_image.png"
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
        
    print(f"\n[0] Original Image Loaded: {img_path} | Shape: {orig_img.shape}")
    orig_stats = calculate_stats(orig_img)
    print(f"    Stats -> Mean: {orig_stats['mean']}, Std: {orig_stats['std']}, Range: [{orig_stats['min']}, {orig_stats['max']}], Entropy: {orig_stats['entropy']} bits")
    
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/0_original.png", orig_img)
    
    # -------------------------------------------------------------
    # STEP 1: INFECTION / DEGRADATION PIPELINE (FOR POINT TRANSFORMS)
    # -------------------------------------------------------------
    print("\n[STEP 1] Degrading (infecting) original image for point operation tests...")
    inf_low_contrast = infect_low_contrast(orig_img, target_min=25, target_max=75)
    inf_low_light = infect_low_light(orig_img, factor=0.15)
    inf_overexposed = infect_overexposed(orig_img, shift=110, scale=1.2)
    
    save_comparison_figure({
        "Original Reference": orig_img,
        "Low Contrast Infection": inf_low_contrast,
        "Low Light Infection": inf_low_light,
        "Overexposed Infection": inf_overexposed
    }, "Step 1: Point Operation Infection (Degradation States)", "step1_infections.png")
    
    # -------------------------------------------------------------
    # STEP 2: POINT TRANSFORM 1 & 2 — NEGATIVE & LINEAR BRIGHTNESS/CONTRAST
    # -------------------------------------------------------------
    print("\n[STEP 2] Point Transforms: Image Negative & Linear (Alpha, Beta)...")
    neg_img = transform_negative(orig_img)
    linear_contrast = transform_linear(inf_low_contrast, alpha=1.8, beta=-40)
    linear_bright = transform_linear(orig_img, alpha=1.0, beta=50)
    
    save_comparison_figure({
        "Original Reference": orig_img,
        "Negative s = 255 - r": neg_img,
        "Low Contrast Input": inf_low_contrast,
        "Linear s = 1.8r - 40": linear_contrast
    }, "Step 2: Negative & Linear Point Transformations", "step2_negative_linear.png")
    
    # -------------------------------------------------------------
    # STEP 3: POINT TRANSFORM 3 — CONTRAST STRETCHING
    # -------------------------------------------------------------
    print("\n[STEP 3] Point Transforms: Min-Max & Percentile Contrast Stretching...")
    minmax_stretched = transform_minmax_stretch(inf_low_contrast)
    percentile_stretched = transform_percentile_stretch(inf_low_contrast, p_low=1, p_high=99)
    
    save_comparison_figure({
        "Low-Contrast Input": inf_low_contrast,
        "Min-Max Stretch": minmax_stretched,
        "Percentile Stretch (1-99%)": percentile_stretched,
        "Original Reference": orig_img
    }, "Step 3: Contrast Stretching Point Transformations", "step3_contrast_stretching.png")
    
    # -------------------------------------------------------------
    # STEP 4: POINT TRANSFORM 4 & 5 — LOG TRANSFORM & GAMMA CORRECTION
    # -------------------------------------------------------------
    print("\n[STEP 4] Point Transforms: Log Transform & Gamma Correction...")
    log_enhanced = transform_log(inf_low_light)
    gamma_04 = transform_gamma(inf_low_light, gamma=0.4)
    gamma_20 = transform_gamma(inf_overexposed, gamma=2.0)
    
    save_comparison_figure({
        "Low-Light Input": inf_low_light,
        "Log s = c log(1+r)": log_enhanced,
        "Gamma (gamma = 0.4)": gamma_04,
        "Overexposed Input": inf_overexposed,
        "Gamma (gamma = 2.0)": gamma_20
    }, "Step 4: Log & Gamma Point Transformations", "step4_log_gamma.png")
    
    # -------------------------------------------------------------
    # STEP 5: POINT TRANSFORM 6 & 7 — THRESHOLDING & HISTOGRAM EQUALIZATION
    # -------------------------------------------------------------
    print("\n[STEP 5] Point Transforms: Thresholding & Global Histogram Equalization...")
    thresh_128 = transform_threshold(orig_img, T=128)
    he_enhanced = global_histogram_equalization(inf_low_contrast)
    
    save_comparison_figure({
        "Original Reference": orig_img,
        "Thresholding (T=128)": thresh_128,
        "Low Contrast Input": inf_low_contrast,
        "Global Equalization s=round((L-1)cdf(r))": he_enhanced
    }, "Step 5: Thresholding & Histogram Equalization Point Operations", "step5_threshold_equalization.png")
    
    # -------------------------------------------------------------
    # STEP 6: BIT-PLANE SLICING (8 BINARY PLANES)
    # -------------------------------------------------------------
    print("\n[STEP 6] Point Transforms: Bit-Plane Slicing (Bits 7 to 0)...")
    bit_planes = extract_bit_planes(orig_img)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for idx, (b_name, b_img) in enumerate(bit_planes.items()):
        row, col = idx // 4, idx % 4
        axes[row, col].imshow(b_img, cmap='gray')
        axes[row, col].set_title(b_name, fontsize=10)
        axes[row, col].axis('off')
    plt.suptitle("Step 6: Bit-Plane Slicing (Decomposing Pixels into 8 Binary Planes)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/step6_bit_planes.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # METRICS SUMMARY REPORT FOR ALL 7 POINT OPERATIONS
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" POINT OPERATIONS QUANTITATIVE EVALUATION TABLE ")
    print("=" * 75)
    print(f"{'Point Operation / Method':<32} | {'Mean':<6} | {'Std (Contrast)':<14} | {'Entropy':<8} | {'PSNR (dB)':<9}")
    print("-" * 75)
    
    eval_list = [
        ("Original Reference", orig_img, orig_img),
        ("Infected: Low Contrast", inf_low_contrast, orig_img),
        ("Infected: Low Light", inf_low_light, orig_img),
        ("Infected: Overexposed", inf_overexposed, orig_img),
        ("1. Image Negative (s=255-r)", neg_img, orig_img),
        ("2. Linear Boost (a=1.8, b=-40)", linear_contrast, orig_img),
        ("3. Min-Max Contrast Stretch", minmax_stretched, orig_img),
        ("3. Percentile Stretch (1-99%)", percentile_stretched, orig_img),
        ("4. Log Transform s=clog(1+r)", log_enhanced, orig_img),
        ("5. Gamma Correction (gamma=0.4)", gamma_04, orig_img),
        ("5. Gamma Correction (gamma=2.0)", gamma_20, orig_img),
        ("6. Thresholding (T=128)", thresh_128, orig_img),
        ("7. Global Histogram Equalization", he_enhanced, orig_img),
    ]
    
    for name, img, ref in eval_list:
        st = calculate_stats(img)
        psnr_val = calculate_psnr(ref, img)
        print(f"{name:<32} | {st['mean']:<6} | {st['std']:<14} | {st['entropy']:<8} | {psnr_val:<9}")
    print("=" * 75)
    print("\nExecution complete! Output plots saved to outputs/ directory.")

if __name__ == "__main__":
    main()
