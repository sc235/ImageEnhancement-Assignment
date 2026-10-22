import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

# Ensure denoiser directory is in python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.utils import load_image_rgb, add_gaussian_noise, np2tensor, tensor2np
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse
from src.models import load_dncnn, load_ffdnet

def main():
    print("=" * 75)
    print("   LEARNED DENOISERS MODULE — TECHNIQUE #2: FFDNet (2018)")
    print("   Evaluating on User Image: denoiser/input_image.png")
    print("=" * 75)

    # 1. Paths
    input_path = os.path.join(current_dir, "input_image.png")
    ffdnet_weights = os.path.join(current_dir, "models", "ffdnet_color.pth")
    dncnn_weights = os.path.join(current_dir, "models", "dncnn_color_blind.pth")
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Image
    img_clean = load_image_rgb(input_path)
    H, W, C = img_clean.shape
    print(f"\n[+] Loaded original image: {W}x{H} pixels, {C} color channels.")

    # 3. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading FFDNet on device: {device}...")
    ffdnet = load_ffdnet(ffdnet_weights, device=device)
    total_ffdnet_params = sum(p.numel() for p in ffdnet.parameters())
    print(f"[+] FFDNet successfully loaded ({total_ffdnet_params:,} parameters).")

    print(f"[+] Loading DnCNN for comparative benchmark...")
    dncnn = load_dncnn(dncnn_weights, device=device)

    # =========================================================================
    # EXPERIMENT 1: Quantitative Benchmark Across Noise Levels (sigma=15, 25, 50)
    # =========================================================================
    print("\n" + "=" * 75)
    print("   EXPERIMENT 1: Standard AWGN Benchmark (FFDNet vs DnCNN)")
    print("=" * 75)
    print(f"{'Condition':<25} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'MSE':<10} | {'Runtime (s)':<10}")
    print("-" * 75)

    sigma_levels = [15, 25, 50]
    exp1_results = []

    for sigma in sigma_levels:
        noisy = add_gaussian_noise(img_clean, sigma=sigma, seed=42)
        psnr_noisy = calculate_psnr(img_clean, noisy)
        ssim_noisy = calculate_ssim(img_clean, noisy)
        mse_noisy = calculate_mse(img_clean, noisy)
        print(f"{f'Noisy (sigma={sigma})':<25} | {psnr_noisy:<10.2f} | {ssim_noisy:<8.4f} | {mse_noisy:<10.2f} | {'-':<10}")

        # Run DnCNN
        t_in = np2tensor(noisy, device=device)
        t0 = time.time()
        with torch.no_grad():
            t_dncnn = dncnn(t_in)
        time_dncnn = time.time() - t0
        out_dncnn = tensor2np(t_dncnn)
        psnr_dncnn = calculate_psnr(img_clean, out_dncnn)
        ssim_dncnn = calculate_ssim(img_clean, out_dncnn)
        mse_dncnn = calculate_mse(img_clean, out_dncnn)
        print(f"{'  -> DnCNN (2017)':<25} | {psnr_dncnn:<10.2f} | {ssim_dncnn:<8.4f} | {mse_dncnn:<10.2f} | {time_dncnn:<10.3f}")

        # Run FFDNet (sigma normalized to [0, 1] as sigma / 255.0)
        t0 = time.time()
        with torch.no_grad():
            t_ffdnet = ffdnet(t_in, sigma / 255.0)
        time_ffdnet = time.time() - t0
        out_ffdnet = tensor2np(t_ffdnet)
        psnr_ffdnet = calculate_psnr(img_clean, out_ffdnet)
        ssim_ffdnet = calculate_ssim(img_clean, out_ffdnet)
        mse_ffdnet = calculate_mse(img_clean, out_ffdnet)
        speedup = time_dncnn / max(time_ffdnet, 1e-4)
        print(f"{f'  -> FFDNet (2018)':<25} | {psnr_ffdnet:<10.2f} | {ssim_ffdnet:<8.4f} | {mse_ffdnet:<10.2f} | {time_ffdnet:<10.3f} ({speedup:.1f}x faster)")
        print("-" * 75)

        exp1_results.append({
            'sigma': sigma,
            'noisy': (noisy, psnr_noisy, ssim_noisy),
            'dncnn': (out_dncnn, psnr_dncnn, ssim_dncnn, time_dncnn),
            'ffdnet': (out_ffdnet, psnr_ffdnet, ssim_ffdnet, time_ffdnet)
        })

    # =========================================================================
    # EXPERIMENT 2: Controllability Analysis (Varying Input Sigma on Fixed Noise)
    # =========================================================================
    print("\n" + "=" * 75)
    print("   EXPERIMENT 2: Noise Level Controllability on Fixed Noise (sigma=25)")
    print("=" * 75)
    noisy_25 = exp1_results[1]['noisy'][0]
    t_in_25 = np2tensor(noisy_25, device=device)

    tested_input_sigmas = [10, 25, 50, 75]
    controllability_results = {}
    print(f"{'Injected Sigma_in':<25} | {'Behavior':<20} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 75)

    for s_in in tested_input_sigmas:
        with torch.no_grad():
            out_ctrl = tensor2np(ffdnet(t_in_25, s_in / 255.0))
        p = calculate_psnr(img_clean, out_ctrl)
        s = calculate_ssim(img_clean, out_ctrl)
        behavior = "Under-smoothed" if s_in < 25 else ("Matched (Optimal)" if s_in == 25 else "Over-smoothed")
        print(f"{s_in:<25} | {behavior:<20} | {p:<10.2f} | {s:<8.4f}")
        controllability_results[s_in] = (out_ctrl, p, s, behavior)

    # =========================================================================
    # EXPERIMENT 3: Spatially Variant Denoising (Non-Uniform Noise Field)
    # =========================================================================
    print("\n" + "=" * 75)
    print("   EXPERIMENT 3: Spatially-Variant Noise Map Denoising")
    print("=" * 75)
    # Create gradient noise: Top half sigma=10, Bottom half sigma=45
    rng = np.random.default_rng(42)
    sigma_spatial = np.zeros((H, W), dtype=np.float32)
    for row in range(H):
        ratio = row / float(H)
        sigma_spatial[row, :] = 10.0 + ratio * 35.0  # smoothly from 10 to 45

    noise_var = rng.normal(0, 1.0, img_clean.shape) * np.expand_dims(sigma_spatial, 2)
    noisy_spatial = np.clip(img_clean.astype(np.float64) + noise_var, 0, 255).round().astype(np.uint8)

    # Pass 2D noise level map to FFDNet
    t_spatial_in = np2tensor(noisy_spatial, device=device)
    t_sigma_map = torch.from_numpy(sigma_spatial.astype(np.float32) / 255.0).to(device)

    with torch.no_grad():
        out_spatial = tensor2np(ffdnet(t_spatial_in, t_sigma_map))

    psnr_spatial_noisy = calculate_psnr(img_clean, noisy_spatial)
    psnr_spatial_denoised = calculate_psnr(img_clean, out_spatial)
    ssim_spatial_denoised = calculate_ssim(img_clean, out_spatial)
    print(f"Spatially Noisy Input PSNR: {psnr_spatial_noisy:.2f} dB")
    print(f"FFDNet Spatially Adapted Denoised PSNR: {psnr_spatial_denoised:.2f} dB | SSIM: {ssim_spatial_denoised:.4f}")

    # =========================================================================
    # GENERATE VISUAL FIGURES
    # =========================================================================
    print("\n[+] Generating visual evaluation figures for FFDNet...")

    # Figure 4: Head-to-Head Comparison: Ground Truth, Noisy (sigma=25), DnCNN, FFDNet
    s25_data = exp1_results[1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(img_clean)
    axes[0].set_title("Ground Truth Clean Image\n(PSNR: inf dB)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(s25_data['noisy'][0])
    axes[1].set_title(f"Noisy Input (sigma=25)\nPSNR: {s25_data['noisy'][1]:.2f} dB", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(s25_data['dncnn'][0])
    axes[2].set_title(f"DnCNN (2017) [Runtime: {s25_data['dncnn'][3]:.2f}s]\nPSNR: {s25_data['dncnn'][1]:.2f} dB | SSIM: {s25_data['dncnn'][2]:.4f}", fontsize=11, color="blue")
    axes[2].axis("off")

    axes[3].imshow(s25_data['ffdnet'][0])
    axes[3].set_title(f"FFDNet (2018) [Runtime: {s25_data['ffdnet'][3]:.2f}s]\nPSNR: {s25_data['ffdnet'][1]:.2f} dB | SSIM: {s25_data['ffdnet'][2]:.4f}", fontsize=11, color="green", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "4_ffdnet_vs_dncnn_sigma25.png")
    plt.savefig(fig4_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig4_path}")

    # Figure 5: Controllability (Under vs Optimal vs Over-smoothing)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for i, s_val in enumerate(tested_input_sigmas):
        img_out, p, s, beh = controllability_results[s_val]
        axes[i].imshow(img_out)
        color = "red" if s_val < 25 else ("green" if s_val == 25 else "orange")
        axes[i].set_title(f"sigma_in = {s_val} ({beh})\nPSNR: {p:.2f} dB | SSIM: {s:.4f}", fontsize=11, color=color, fontweight="bold" if s_val==25 else "normal")
        axes[i].axis("off")

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "5_ffdnet_controllability_sigma.png")
    plt.savefig(fig5_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig5_path}")

    # Figure 6: Spatially Variant Denoising
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(noisy_spatial)
    axes[0].set_title(f"Spatially Noisy Input\n(sigma varies 10 -> 45 from Top to Bottom)\nPSNR: {psnr_spatial_noisy:.2f} dB", fontsize=11, color="red")
    axes[0].axis("off")

    # Show noise map visualization
    im_map = axes[1].imshow(sigma_spatial, cmap="hot")
    axes[1].set_title("Input Noise Level Map sigma(x,y)\n[Range: 10.0 to 45.0]", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im_map, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(out_spatial)
    axes[2].set_title(f"FFDNet Spatially Adapted Output\nPSNR: {psnr_spatial_denoised:.2f} dB | SSIM: {ssim_spatial_denoised:.4f}", fontsize=11, color="green", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "6_ffdnet_spatially_variant.png")
    plt.savefig(fig6_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig6_path}")

    print("\n[+] FFDNet evaluation successfully completed!")

if __name__ == "__main__":
    main()
