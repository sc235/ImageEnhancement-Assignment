import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from src.utils import load_image_rgb, add_gaussian_noise, np2tensor, tensor2np
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse
from src.models import load_dncnn

def main():
    print("=" * 70)
    print("   LEARNED DENOISERS MODULE — TECHNIQUE #1: DnCNN (2017)")
    print("   Evaluating on User Image: denoiser/input_image.png")
    print("=" * 70)

    # 1. Paths
    input_path = os.path.join(os.path.dirname(__file__), "input_image.png")
    model_path = os.path.join(os.path.dirname(__file__), "models", "dncnn_color_blind.pth")
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Image
    img_clean = load_image_rgb(input_path)
    H, W, C = img_clean.shape
    print(f"\n[+] Loaded original image: {W}x{H} pixels, {C} color channels.")

    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading pre-trained DnCNN (Color Blind) on device: {device}...")
    dncnn = load_dncnn(model_path, device=device)
    total_params = sum(p.numel() for p in dncnn.parameters())
    print(f"[+] DnCNN Model successfully loaded ({total_params:,} parameters).")

    # 4. Noise Levels to Test
    sigma_levels = [15, 25, 50]
    results = []

    print("\n" + "-" * 70)
    print(f"{'Condition':<25} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'MSE':<10} | {'Time (s)':<8}")
    print("-" * 70)

    # Reference baseline
    psnr_ref = calculate_psnr(img_clean, img_clean)
    ssim_ref = calculate_ssim(img_clean, img_clean)
    mse_ref = calculate_mse(img_clean, img_clean)
    print(f"{'0. Clean Reference':<25} | {psnr_ref:<10.2f} | {ssim_ref:<8.4f} | {mse_ref:<10.2f} | {'-':<8}")

    denoised_samples = {}

    for sigma in sigma_levels:
        # Add AWGN
        noisy = add_gaussian_noise(img_clean, sigma=sigma, seed=42)
        psnr_noisy = calculate_psnr(img_clean, noisy)
        ssim_noisy = calculate_ssim(img_clean, noisy)
        mse_noisy = calculate_mse(img_clean, noisy)
        print(f"{f'Noisy (sigma={sigma})':<25} | {psnr_noisy:<10.2f} | {ssim_noisy:<8.4f} | {mse_noisy:<10.2f} | {'-':<8}")

        # Run Classical Gaussian Blur (5x5, sigma=1.5)
        t0 = time.time()
        classic_gauss = cv2.GaussianBlur(noisy, (5, 5), 1.5)
        t_gauss = time.time() - t0
        psnr_gauss = calculate_psnr(img_clean, classic_gauss)
        ssim_gauss = calculate_ssim(img_clean, classic_gauss)
        mse_gauss = calculate_mse(img_clean, classic_gauss)
        print(f"{f'  -> Gaussian Blur (5x5)':<25} | {psnr_gauss:<10.2f} | {ssim_gauss:<8.4f} | {mse_gauss:<10.2f} | {t_gauss:<8.4f}")

        # Run Classical Non-Local Means (SOTA classical patch filter)
        t0 = time.time()
        classic_nlm = cv2.fastNlMeansDenoisingColored(noisy, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
        t_nlm = time.time() - t0
        psnr_nlm = calculate_psnr(img_clean, classic_nlm)
        ssim_nlm = calculate_ssim(img_clean, classic_nlm)
        mse_nlm = calculate_mse(img_clean, classic_nlm)
        print(f"{f'  -> Non-Local Means (NLM)':<25} | {psnr_nlm:<10.2f} | {ssim_nlm:<8.4f} | {mse_nlm:<10.2f} | {t_nlm:<8.4f}")

        # Run DnCNN
        tensor_in = np2tensor(noisy, device=device)
        t0 = time.time()
        with torch.no_grad():
            tensor_out = dncnn(tensor_in)
        t_dncnn = time.time() - t0
        dncnn_out = tensor2np(tensor_out)

        psnr_dncnn = calculate_psnr(img_clean, dncnn_out)
        ssim_dncnn = calculate_ssim(img_clean, dncnn_out)
        mse_dncnn = calculate_mse(img_clean, dncnn_out)
        print(f"{f'  -> DnCNN (Learned)':<25} | {psnr_dncnn:<10.2f} | {ssim_dncnn:<8.4f} | {mse_dncnn:<10.2f} | {t_dncnn:<8.4f}")
        print("-" * 70)

        results.append({
            'sigma': sigma,
            'noisy': (noisy, psnr_noisy, ssim_noisy),
            'gauss': (classic_gauss, psnr_gauss, ssim_gauss),
            'nlm': (classic_nlm, psnr_nlm, ssim_nlm),
            'dncnn': (dncnn_out, psnr_dncnn, ssim_dncnn, t_dncnn)
        })
        denoised_samples[sigma] = dncnn_out

    # 5. Generate Figures
    print("\n[+] Generating visual evaluation figures...")

    # Figure 1: Full-view Comparison for sigma=25
    s25 = results[1]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    axes[0].imshow(img_clean)
    axes[0].set_title("Ground Truth Clean Image\n(PSNR: inf dB | SSIM: 1.000)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(s25['noisy'][0])
    axes[1].set_title(f"Noisy Input (sigma=25)\nPSNR: {s25['noisy'][1]:.2f} dB | SSIM: {s25['noisy'][2]:.4f}", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(s25['nlm'][0])
    axes[2].set_title(f"Classical Non-Local Means (NLM)\nPSNR: {s25['nlm'][1]:.2f} dB | SSIM: {s25['nlm'][2]:.4f}", fontsize=11, color="orange")
    axes[2].axis("off")

    axes[3].imshow(s25['dncnn'][0])
    axes[3].set_title(f"DnCNN (Learned Residual CNN)\nPSNR: {s25['dncnn'][1]:.2f} dB | SSIM: {s25['dncnn'][2]:.4f}", fontsize=11, color="green", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "1_dncnn_full_comparison_sigma25.png")
    plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig1_path}")

    # Figure 2: Zoomed-In Patch Comparison (Focusing on the person and illuminated grass)
    # The person is around center: y in [550, 750], x in [300, 480]
    y1, y2, x1, x2 = 540, 740, 300, 480
    crop_clean = img_clean[y1:y2, x1:x2]
    crop_noisy = s25['noisy'][0][y1:y2, x1:x2]
    crop_nlm   = s25['nlm'][0][y1:y2, x1:x2]
    crop_dncnn = s25['dncnn'][0][y1:y2, x1:x2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(crop_clean)
    axes[0].set_title("Ground Truth Crop", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(crop_noisy)
    axes[1].set_title("Noisy (sigma=25)", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(crop_nlm)
    axes[2].set_title("Classical NLM", fontsize=11, color="orange")
    axes[2].axis("off")

    axes[3].imshow(crop_dncnn)
    axes[3].set_title("DnCNN Denoised", fontsize=11, color="green", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "2_dncnn_crop_detail_sigma25.png")
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig2_path}")

    # Figure 3: DnCNN Performance Across Multiple Noise Levels (15, 25, 50)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for idx, r in enumerate(results):
        sigma = r['sigma']
        # Noisy row
        axes[0, idx].imshow(r['noisy'][0])
        axes[0, idx].set_title(f"Noisy Input (sigma={sigma})\nPSNR: {r['noisy'][1]:.2f} dB", fontsize=11, color="red")
        axes[0, idx].axis("off")

        # DnCNN row
        axes[1, idx].imshow(r['dncnn'][0])
        axes[1, idx].set_title(f"DnCNN Denoised (sigma={sigma})\nPSNR: {r['dncnn'][1]:.2f} dB | SSIM: {r['dncnn'][2]:.4f}", fontsize=11, color="green", fontweight="bold")
        axes[1, idx].axis("off")

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "3_dncnn_multi_sigma_levels.png")
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig3_path}")

    print("\n[+] DnCNN evaluation successfully completed!")

if __name__ == "__main__":
    main()
