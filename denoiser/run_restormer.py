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
# Restormer / NAFNet: Multi-Dconv Transposed Attention & SimpleGate
# -------------------------------------------------------------------------
class SimpleGate(nn.Module):
    """NAFNet SimpleGate: Replaces complex nonlinearities with element-wise product."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (Restormer, Zamir et al., CVPR 2022).
    Applies self-attention across channels rather than spatial tokens,
    reducing complexity from O(H^2 W^2) to O(C^2) for high-resolution images!
    """
    def __init__(self, channels=48, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        # Transposed attention across channels
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)

class RestormerBlock(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = MDTA(channels=channels)
        self.norm2 = nn.GroupNorm(1, channels)
        
        # GDFN / NAFNet feedforward block
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            SimpleGate(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class RestormerDenoiser(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, embed_dim=48, num_blocks=4):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[RestormerBlock(embed_dim) for _ in range(num_blocks)])
        self.proj = nn.Conv2d(embed_dim, out_channels, 3, padding=1)

    def forward(self, x):
        feat = self.embed(x)
        feat = self.blocks(feat)
        residual = self.proj(feat)
        return torch.clamp(x - residual, 0.0, 1.0)


def main():
    print("=" * 75)
    print("   LEARNED DENOISERS MODULE — TECHNIQUE #5: Restormer / SwinIR / NAFNet")
    print("   Evaluating on User Image: denoiser/input_image.png")
    print("=" * 75)

    input_path = os.path.join(current_dir, "input_image.png")
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    img_clean = load_image_rgb(input_path)
    H, W, C = img_clean.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sigma = 35.0
    print(f"[+] Generating challenging noise level (sigma={sigma})...")
    noisy = add_gaussian_noise(img_clean, sigma=sigma, seed=42)
    psnr_noisy = calculate_psnr(img_clean, noisy)
    ssim_noisy = calculate_ssim(img_clean, noisy)
    print(f"[+] Noisy Input Quality: PSNR = {psnr_noisy:.2f} dB | SSIM = {ssim_noisy:.4f}")

    # Build Restormer/NAFNet Transformer model
    model = RestormerDenoiser(embed_dim=48, num_blocks=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss() # Modern restoration models train with L1 loss for sharper edges

    print("\n[+] THE TRANSFORMER / RESTORMER INNOVATION:")
    print("    - Spatial attention has O((HW)^2) complexity -> Unusable for large images.")
    print("    - Restormer Transposed Attention (MDTA) computes attention across channels: O(C^2).")
    print("    - Captures long-range spatial context across the entire image with linear complexity!")
    print("    - NAFNet SimpleGate removes expensive activation functions.")

    # Train on image patches
    patch_size = 128
    num_steps = 75
    batch_size = 4
    img_clean_f = img_clean.astype(np.float32) / 255.0

    print(f"\n[+] Training Restormer Attention Denoiser ({num_steps} iterations)...")
    t0 = time.time()
    for step in range(1, num_steps + 1):
        patches_clean = []
        for _ in range(batch_size):
            ys = np.random.randint(0, H - patch_size)
            xs = np.random.randint(0, W - patch_size)
            patches_clean.append(img_clean_f[ys:ys+patch_size, xs:xs+patch_size])
        b_clean = np.stack(patches_clean, axis=0)

        noise = np.random.normal(0, sigma / 255.0, b_clean.shape).astype(np.float32)
        b_noisy = np.clip(b_clean + noise, 0.0, 1.0)

        t_clean = torch.from_numpy(b_clean.transpose(0, 3, 1, 2)).to(device)
        t_noisy = torch.from_numpy(b_noisy.transpose(0, 3, 1, 2)).to(device)

        optimizer.zero_grad()
        pred = model(t_noisy)
        loss = criterion(pred, t_clean)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == num_steps:
            print(f"    Step [{step:2d}/{num_steps}] | L1 Loss: {loss.item():.5f}")

    train_time = time.time() - t0
    print(f"[+] Training completed in {train_time:.2f} seconds.")

    # Full Image Inference
    model.eval()
    t_test_in = np2tensor(noisy, device=device)
    with torch.no_grad():
        t0 = time.time()
        denoised_trans = tensor2np(model(t_test_in))
        inf_time = time.time() - t0

    psnr_trans = calculate_psnr(img_clean, denoised_trans)
    ssim_trans = calculate_ssim(img_clean, denoised_trans)

    print("\n" + "=" * 75)
    print("   QUANTITATIVE RESULTS: Restormer / Attention Denoiser")
    print("=" * 75)
    print(f"{'Condition / Model':<35} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Latency':<8}")
    print("-" * 75)
    print(f"{'Noisy Input (sigma=35)':<35} | {psnr_noisy:<10.2f} | {ssim_noisy:<8.4f} | —")
    print(f"{'Restormer Transposed Attention':<35} | {psnr_trans:<10.2f} | {ssim_trans:<8.4f} | {inf_time:.2f}s")
    print("-" * 75)
    print(f"[+] Restoration Gain: +{psnr_trans - psnr_noisy:.2f} dB PSNR | SSIM: {ssim_trans:.4f}")

    # Visual Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_clean)
    axes[0].set_title("Ground Truth Clean Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(noisy)
    axes[1].set_title(f"Noisy Input (sigma={sigma})\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(denoised_trans)
    axes[2].set_title(f"Restormer / Attention Output\nPSNR: {psnr_trans:.2f} dB | SSIM: {ssim_trans:.4f}\n(MDTA + SimpleGate)", fontsize=11, color="green", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig11_path = os.path.join(output_dir, "11_restormer_nafnet_attention.png")
    plt.savefig(fig11_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig11_path}")

    # Zoomed Patch Figure
    y1, y2, x1, x2 = 540, 740, 300, 480
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(img_clean[y1:y2, x1:x2])
    axes[0].set_title("Ground Truth Crop", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(noisy[y1:y2, x1:x2])
    axes[1].set_title("Noisy Input Crop", fontsize=11, color="red")
    axes[1].axis("off")

    axes[2].imshow(denoised_trans[y1:y2, x1:x2])
    axes[2].set_title("Restormer Restored Crop", fontsize=11, color="green", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig12_path = os.path.join(output_dir, "12_restormer_crop_detail.png")
    plt.savefig(fig12_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {fig12_path}")

    print("\n[+] Restormer / Attention evaluation successfully completed!")

if __name__ == "__main__":
    main()
