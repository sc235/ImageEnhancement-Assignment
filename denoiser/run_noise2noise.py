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
# Lightweight U-Net Architecture for Noise2Noise
# -------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = int((4 - h % 4) % 4)
        pad_w = int((4 - w % 4) % 4)
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        if pad_h > 0 or pad_w > 0:
            out = out[..., :h, :w]
        return torch.clamp(out, 0.0, 1.0)


def main():
    print("=" * 75)
    print("   LEARNED DENOISERS MODULE — TECHNIQUE #3: Noise2Noise (2018)")
    print("   Evaluating on User Image: denoiser/input_image.png")
    print("=" * 75)

    input_path = os.path.join(current_dir, "input_image.png")
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    img_clean = load_image_rgb(input_path)
    H, W, C = img_clean.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loaded original image: {W}x{H} pixels, {C} color channels.")

    # 1. Theoretical Concept
    print("\n[+] THE NOISE2NOISE THEOREM:")
    print("    Standard Supervised (Noise2Clean): min E[ || f(y1) - x ||^2 ] -> f*(y1) = E[ x | y1 ]")
    print("    Noise2Noise:                      min E[ || f(y1) - y2 ||^2 ]")
    print("    Since y2 = x + v2 and E[ v2 | x ] = 0,")
    print("    E[ y2 | y1 ] = E[ x + v2 | y1 ] = E[ x | y1 ] + 0 = E[ x | y1 ]")
    print("    Conclusion: The optimal denoiser is mathematically IDENTICAL without clean targets!")

    # 2. Setup Noise Level and Pair Generation
    sigma = 25.0
    print(f"\n[+] Synthesizing independent noisy pairs (sigma={sigma})...")
    # Fixed test images
    noisy_test1 = add_gaussian_noise(img_clean, sigma=sigma, seed=100)
    noisy_test2 = add_gaussian_noise(img_clean, sigma=sigma, seed=200)

    # 3. Instantiate Models
    torch.manual_seed(42)
    model_n2c = MiniUNet().to(device)  # Noise2Clean (Supervised)
    torch.manual_seed(42)
    model_n2n = MiniUNet().to(device)  # Noise2Noise (Unsupervised Target)

    opt_n2c = optim.Adam(model_n2c.parameters(), lr=1e-3)
    opt_n2n = optim.Adam(model_n2n.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 4. Training Loop: 100 iterations on random patches from user image
    patch_size = 128
    num_iters = 80
    batch_size = 4
    img_clean_f = img_clean.astype(np.float32) / 255.0

    print(f"[+] Training Noise2Clean vs Noise2Noise on {num_iters} patch iterations...")
    t0 = time.time()
    for it in range(1, num_iters + 1):
        # Sample random patches
        patches_clean = []
        for _ in range(batch_size):
            y_start = np.random.randint(0, H - patch_size)
            x_start = np.random.randint(0, W - patch_size)
            patch = img_clean_f[y_start:y_start+patch_size, x_start:x_start+patch_size]
            patches_clean.append(patch)
        clean_batch = np.stack(patches_clean, axis=0) # [B, P, P, 3]

        # Generate independent noises v1, v2
        noise1 = np.random.normal(0, sigma / 255.0, clean_batch.shape).astype(np.float32)
        noise2 = np.random.normal(0, sigma / 255.0, clean_batch.shape).astype(np.float32)

        y1_batch = np.clip(clean_batch + noise1, 0.0, 1.0)
        y2_batch = np.clip(clean_batch + noise2, 0.0, 1.0)

        t_clean = torch.from_numpy(clean_batch.transpose(0, 3, 1, 2)).to(device)
        t_y1 = torch.from_numpy(y1_batch.transpose(0, 3, 1, 2)).to(device)
        t_y2 = torch.from_numpy(y2_batch.transpose(0, 3, 1, 2)).to(device)

        # Train Noise2Clean: input y1 -> target clean x
        opt_n2c.zero_grad()
        loss_n2c = criterion(model_n2c(t_y1), t_clean)
        loss_n2c.backward()
        opt_n2c.step()

        # Train Noise2Noise: input y1 -> target noisy y2
        opt_n2n.zero_grad()
        loss_n2n = criterion(model_n2n(t_y1), t_y2)
        loss_n2n.backward()
        opt_n2n.step()

        if it % 20 == 0 or it == num_iters:
            print(f"    Iter [{it:2d}/{num_iters}] | N2C Loss: {loss_n2c.item():.5f} | N2N Loss: {loss_n2n.item():.5f}")

    train_time = time.time() - t0
    print(f"[+] Training completed in {train_time:.2f} seconds.")

    # 5. Full-Resolution Evaluation on Unseen Test Noise
    model_n2c.eval()
    model_n2n.eval()

    t_test_in = np2tensor(noisy_test1, device=device)
    with torch.no_grad():
        out_n2c = tensor2np(model_n2c(t_test_in))
        out_n2n = tensor2np(model_n2n(t_test_in))

    # Metrics
    psnr_noisy = calculate_psnr(img_clean, noisy_test1)
    ssim_noisy = calculate_ssim(img_clean, noisy_test1)

    psnr_n2c = calculate_psnr(img_clean, out_n2c)
    ssim_n2c = calculate_ssim(img_clean, out_n2c)

    psnr_n2n = calculate_psnr(img_clean, out_n2n)
    ssim_n2n = calculate_ssim(img_clean, out_n2n)

    print("\n" + "=" * 75)
    print("   QUANTITATIVE COMPARISON: Supervised (N2C) vs Unsupervised (N2N)")
    print("=" * 75)
    print(f"{'Method / Training Target':<35} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Clean Target Needed?'}")
    print("-" * 75)
    print(f"{'Noisy Input y1':<35} | {psnr_noisy:<10.2f} | {ssim_noisy:<8.4f} | —")
    print(f"{'Noise2Clean (Supervised Target x)':<35} | {psnr_n2c:<10.2f} | {ssim_n2c:<8.4f} | YES")
    print(f"{'Noise2Noise (Noisy Target y2)':<35} | {psnr_n2n:<10.2f} | {ssim_n2n:<8.4f} | NO (Zero clean images!)")
    print("-" * 75)
    diff_psnr = abs(psnr_n2c - psnr_n2n)
    print(f"[+] Difference between N2C and N2N: Only {diff_psnr:.2f} dB!")

    # 6. Generate Figures
    print("\n[+] Generating visual evaluation figures for Noise2Noise...")
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    axes[0].imshow(img_clean)
    axes[0].set_title("Ground Truth (Clean x)\n[Never seen by N2N!]", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(noisy_test1)
    axes[1].set_title(f"Input Image (y1)\nPSNR: {psnr_noisy:.2f} dB", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(noisy_test2)
    axes[2].set_title("Target Image for N2N (y2)\n[Notice: Completely Noisy!]", fontsize=11, color="purple")
    axes[2].axis("off")

    axes[3].imshow(out_n2c)
    axes[3].set_title(f"Noise2Clean Output (Supervised)\nPSNR: {psnr_n2c:.2f} dB | SSIM: {ssim_n2c:.4f}", fontsize=11, color="blue")
    axes[3].axis("off")

    axes[4].imshow(out_n2n)
    axes[4].set_title(f"Noise2Noise Output (Unsupervised)\nPSNR: {psnr_n2n:.2f} dB | SSIM: {ssim_n2n:.4f}", fontsize=11, color="green", fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    fig7_path = os.path.join(output_dir, "7_noise2noise_comparison.png")
    plt.savefig(fig7_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig7_path}")

    # Zoomed-in patch comparison
    y1, y2, x1, x2 = 540, 740, 300, 480
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img_clean[y1:y2, x1:x2])
    axes[0].set_title("Ground Truth Crop", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(noisy_test1[y1:y2, x1:x2])
    axes[1].set_title("Noisy Input (y1)", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(out_n2c[y1:y2, x1:x2])
    axes[2].set_title(f"Noise2Clean Crop ({psnr_n2c:.2f} dB)", fontsize=11, color="blue")
    axes[2].axis("off")

    axes[3].imshow(out_n2n[y1:y2, x1:x2])
    axes[3].set_title(f"Noise2Noise Crop ({psnr_n2n:.2f} dB)", fontsize=11, color="green", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig8_path = os.path.join(output_dir, "8_noise2noise_crop_detail.png")
    plt.savefig(fig8_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig8_path}")

    print("\n[+] Noise2Noise evaluation successfully completed!")

if __name__ == "__main__":
    main()
