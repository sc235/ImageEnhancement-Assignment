import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.utils import load_image_rgb, add_gaussian_noise, np2tensor, tensor2np
from src.metrics import calculate_psnr, calculate_ssim, calculate_mse

# -------------------------------------------------------------------------
# Compact Single-Image Blind-Spot / Self-Supervised Network
# -------------------------------------------------------------------------
class SingleImageDenoiser(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1), # Self2Self dropout component
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(features, out_channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


def create_n2v_blind_spots(patch: torch.Tensor, mask_ratio: float = 0.02):
    """
    Implements Noise2Void (N2V) Blind-Spot Masking.
    Randomly selects `mask_ratio` pixels and replaces their values with
    a random neighbor, forcing the network to predict center pixel from context.
    """
    B, C, H, W = patch.shape
    blind_patch = patch.clone()
    mask = torch.rand((B, 1, H, W), device=patch.device) < mask_ratio

    # Neighbor offsets: (-1, 0, 1) excluding (0,0)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    for dy, dx in offsets:
        # Rolled neighbor values
        neighbor = torch.roll(patch, shifts=(dy, dx), dims=(-2, -1))
        # Where masked, randomly replace
        replace_cond = mask & (torch.rand((B, 1, H, W), device=patch.device) < (1.0 / len(offsets)))
        blind_patch = torch.where(replace_cond.expand_as(patch), neighbor, blind_patch)

    return blind_patch, mask


def main():
    print("=" * 75)
    print("   LEARNED DENOISERS MODULE — TECHNIQUE #4: Noise2Void & Self2Self")
    print("   Evaluating on User Image: denoiser/input_image.png")
    print("=" * 75)

    input_path = os.path.join(current_dir, "input_image.png")
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    img_clean = load_image_rgb(input_path)
    H, W, C = img_clean.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sigma = 25.0
    print(f"[+] Creating single noisy test image (sigma={sigma})...")
    noisy_single = add_gaussian_noise(img_clean, sigma=sigma, seed=42)
    psnr_noisy = calculate_psnr(img_clean, noisy_single)
    ssim_noisy = calculate_ssim(img_clean, noisy_single)
    print(f"[+] Noisy Input Quality: PSNR = {psnr_noisy:.2f} dB | SSIM = {ssim_noisy:.4f}")

    print("\n[+] THE NOISE2VOID / SELF2SELF PARADIGM:")
    print("    - Training Set: ZERO external images. ZERO clean targets. ZERO pairs.")
    print("    - Principle: Noise is pixel-wise independent (uncorrelated).")
    print("    - Mechanism: Blind-spot masking replaces target pixels with neighbors.")
    print("      The network cannot copy the noise, so it predicts the underlying signal E[x|context]!")

    # Setup training exclusively on this single noisy image!
    model = SingleImageDenoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    noisy_tensor = np2tensor(noisy_single, device=device)
    num_steps = 70
    patch_size = 128
    batch_size = 4

    print(f"\n[+] Self-training network directly on the single noisy image ({num_steps} iterations)...")
    t0 = time.time()
    for step in range(1, num_steps + 1):
        patches = []
        for _ in range(batch_size):
            y_start = np.random.randint(0, H - patch_size)
            x_start = np.random.randint(0, W - patch_size)
            patches.append(noisy_tensor[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size])
        batch_patch = torch.cat(patches, dim=0)

        # Apply Noise2Void Blind-Spot Masking
        blind_patch, mask = create_n2v_blind_spots(batch_patch, mask_ratio=0.03)

        optimizer.zero_grad()
        pred = model(blind_patch)

        # Loss evaluated ONLY at the blind-spot masked pixels!
        mask_expanded = mask.expand_as(batch_patch)
        loss = criterion(pred[mask_expanded], batch_patch[mask_expanded])
        loss.backward()
        optimizer.step()

        if step % 15 == 0 or step == num_steps:
            print(f"    Step [{step:2d}/{num_steps}] | Blind-Spot Loss: {loss.item():.5f}")

    train_time = time.time() - t0
    print(f"[+] Self-training finished in {train_time:.2f} seconds.")

    # Self2Self Inference: Ensemble averaging across multiple forward passes (Monte Carlo Dropout)
    print("\n[+] Performing Self2Self Monte Carlo Ensemble Inference (N=10)...")
    model.train() # Keep dropout active for Self2Self ensemble
    ensemble_preds = []
    with torch.no_grad():
        for _ in range(10):
            out_i = model(noisy_tensor)
            ensemble_preds.append(out_i)
        
        # Average the predictions
        avg_pred = torch.stack(ensemble_preds, dim=0).mean(dim=0)
        denoised_n2v = tensor2np(avg_pred)

    psnr_n2v = calculate_psnr(img_clean, denoised_n2v)
    ssim_n2v = calculate_ssim(img_clean, denoised_n2v)
    mse_n2v = calculate_mse(img_clean, denoised_n2v)

    print("\n" + "=" * 75)
    print("   QUANTITATIVE RESULTS: Noise2Void / Self2Self (Single Image Denoising)")
    print("=" * 75)
    print(f"{'Condition / Method':<35} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'External Dataset?'}")
    print("-" * 75)
    print(f"{'Noisy Observation':<35} | {psnr_noisy:<10.2f} | {ssim_noisy:<8.4f} | —")
    print(f"{'Noise2Void / Self2Self (Self-trained)':<35} | {psnr_n2v:<10.2f} | {ssim_n2v:<8.4f} | ZERO (Trained only on test image!)")
    print("-" * 75)
    print(f"[+] Restored Gain: +{psnr_n2v - psnr_noisy:.2f} dB without any training dataset!")

    # Visual Figures
    print("\n[+] Generating visual evaluation figures for Noise2Void / Self2Self...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_clean)
    axes[0].set_title("Ground Truth Clean Image\n[Never seen anywhere!]", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(noisy_single)
    axes[1].set_title(f"Single Noisy Input (sigma=25)\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(denoised_n2v)
    axes[2].set_title(f"Noise2Void / Self2Self Output\nPSNR: {psnr_n2v:.2f} dB | SSIM: {ssim_n2v:.4f}\n(Zero Dataset / Self-Trained)", fontsize=11, color="green", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig9_path = os.path.join(output_dir, "9_noise2void_self2self.png")
    plt.savefig(fig9_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig9_path}")

    # Zoomed Patch
    y1, y2, x1, x2 = 540, 740, 300, 480
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img_clean[y1:y2, x1:x2])
    axes[0].set_title("Ground Truth Crop", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(noisy_single[y1:y2, x1:x2])
    axes[1].set_title("Single Noisy Crop", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(denoised_n2v[y1:y2, x1:x2])
    axes[2].set_title("N2V / Self2Self Denoised Crop", fontsize=11, color="green", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig10_path = os.path.join(output_dir, "10_noise2void_crop.png")
    plt.savefig(fig10_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig10_path}")

    print("\n[+] Noise2Void / Self2Self evaluation successfully completed!")

if __name__ == "__main__":
    main()
