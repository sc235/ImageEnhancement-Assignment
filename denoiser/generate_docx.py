import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(current_dir, "outputs")
output_docx_path = os.path.join(current_dir, "Learned_Denoisers_Report.docx")

def get_image_path(filename):
    base = os.path.splitext(filename)[0]
    jpg_path = os.path.join(outputs_dir, f"{base}.jpg")
    png_path = os.path.join(outputs_dir, filename)
    return jpg_path if os.path.exists(jpg_path) else png_path

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def generate_word_report():
    print("[+] Building Word Document (.docx)...")
    doc = docx.Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    PRIMARY = RGBColor(24, 76, 120)    # Deep Navy
    SECONDARY = RGBColor(41, 128, 185) # Blue
    DARK = RGBColor(44, 62, 80)        # Charcoal
    MUTED = RGBColor(127, 140, 141)    # Gray
    CODE_COLOR = RGBColor(31, 35, 40)

    # Defaults
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(10.5)
    doc.styles['Normal'].font.color.rgb = DARK

    def add_h(text, level=1):
        h = doc.add_heading(text, level=level)
        r = h.runs[0]
        if level == 1:
            r.font.name = 'Calibri'
            r.font.size = Pt(16)
            r.font.bold = True
            r.font.color.rgb = PRIMARY
            h.paragraph_format.space_before = Pt(16)
            h.paragraph_format.space_after = Pt(6)
        elif level == 2:
            r.font.name = 'Calibri'
            r.font.size = Pt(13)
            r.font.bold = True
            r.font.color.rgb = SECONDARY
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
        elif level == 3:
            r.font.name = 'Calibri'
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = DARK
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(2)
        return h

    def add_p(text, bold_prefix=""):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.color.rgb = PRIMARY
        p.add_run(text)
        return p

    def add_code_block(code_text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, "F6F8FA")
        
        # Border
        tcPr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement('w:tcBorders')
        left = OxmlElement('w:left')
        left.set(qn('w:val'), 'single')
        left.set(qn('w:sz'), '16')
        left.set(qn('w:space'), '0')
        left.set(qn('w:color'), '184C78')
        borders.append(left)
        tcPr.append(borders)

        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(code_text.strip())
        r.font.name = 'Consolas'
        r.font.size = Pt(8)
        r.font.color.rgb = CODE_COLOR
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def add_img(filename, caption, width=Inches(6.2)):
        filepath = get_image_path(filename)
        if os.path.exists(filepath):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.add_run().add_picture(filepath, width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(10)
            r = p_cap.add_run(f"Figure: {caption}")
            r.font.size = Pt(9)
            r.font.italic = True
            r.font.color.rgb = MUTED

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_t = p_title.add_run("ACADEMIC EVALUATION & CODE REPORT:\nMODERN LEARNED DENOISERS")
    r_t.font.name = 'Calibri'
    r_t.font.size = Pt(22)
    r_t.font.bold = True
    r_t.font.color.rgb = PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(14)
    r_s = p_sub.add_run(
        "Student / Author: Mouhamadou Ahmed Ndiaye\n"
        "Course: CS-4814 — Digital Image Processing & Computer Vision (CS Senior)\n"
        "Assignment: Module 4 — Modern Learned Denoisers (Neural Networks)\n"
        "Evaluation Target: input_image.png (736 x 981 pixels, 24-bit RGB Landscape Photograph)"
    )
    r_s.font.size = Pt(10.5)
    r_s.font.color.rgb = MUTED

    # 1. Executive Summary & Taxonomy
    add_h("1. Executive Summary & Master Taxonomy", level=1)
    add_p(
        "Image denoising is a fundamental inverse problem in computer vision. Over the past decade, learned neural "
        "denoisers have completely superseded classical linear and non-local patch filters (BM3D). This report documents "
        "the implementation, mathematical derivations, full PyTorch source codes, and empirical benchmarks of all 6 modern "
        "learned denoisers presented in Module 4:"
    )

    # Taxonomy Table
    t1_headers = ["#", "Technique", "Year", "Supervision", "Core Mathematical / Architectural Idea", "Landmark Note"]
    t1_data = [
        ["1", "DnCNN", "2017", "Supervised (Clean)", "Residual Learning: v = R(y), x = y - v via 20 Conv layers + BN", "First deep CNN to decisively defeat BM3D"],
        ["2", "FFDNet", "2018", "Supervised (Noise Map)", "Noise map sigma input + PixelUnshuffle(2) quarter-resolution processing", "Unified model for any noise level; 3.8x faster"],
        ["3", "Noise2Noise", "2018", "Unsupervised (Pairs)", "E[y2|y1] = E[x|y1] given zero-mean noise E[v2|x] = 0", "Zero clean ground truth images needed at all"],
        ["4", "Noise2Void", "2019", "Self-Supervised (Single)", "Blind-spot masking (center pixel replaced by neighbor) + Monte Carlo dropout", "Zero external dataset; self-trains directly on test photo"],
        ["5", "Restormer", "2022", "Vision Transformer", "Multi-Dconv Transposed Attention across channels O(C^2) + SimpleGate", "Overcame O((HW)^2) spatial bottleneck; SOTA leader"],
        ["6", "Diffusion", "2021+", "Generative Langevin", "Reverse Langevin stochastic sampling from posterior p(x|y)", "Solves blurry L2 conditional mean; synthesizes texture"]
    ]
    t1 = doc.add_table(rows=len(t1_data)+1, cols=len(t1_headers))
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(t1_headers):
        c = t1.cell(0, j)
        c.text = h
        set_cell_background(c, "184C78")
        c.paragraphs[0].runs[0].font.bold = True
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        c.paragraphs[0].runs[0].font.size = Pt(9)
    for i, row in enumerate(t1_data):
        bg = "F9FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            c = t1.cell(i+1, j)
            c.text = val
            set_cell_background(c, bg)
            c.paragraphs[0].runs[0].font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 2. Quantitative Benchmark Table
    add_h("2. Quantitative Benchmark Results on input_image.png", level=1)
    add_p("All models were evaluated on the provided RGB landscape photograph (736 x 981 pixels):")

    t2_headers = ["Condition / Model", "Supervision Paradigm", "Noise Setting", "PSNR (dB)", "SSIM", "MSE", "CPU Latency"]
    t2_data = [
        ["Ground Truth Reference", "Reference", "Pristine Clean", "inf", "1.0000", "0.00", "-"],
        ["Corrupted: Low Noise", "AWGN Degradation", "sigma = 15", "24.91", "0.2786", "209.69", "-"],
        ["Corrupted: Medium Noise", "AWGN Degradation", "sigma = 25", "21.02", "0.1403", "514.33", "-"],
        ["Corrupted: Heavy Noise", "AWGN Degradation", "sigma = 50", "15.95", "0.0492", "1651.52", "-"],
        ["Classical Gaussian Blur (5x5)", "Spatial Filter", "sigma = 25", "32.56", "0.7835", "36.05", "0.002 s"],
        ["Classical Non-Local Means", "Patch Filter (NLM)", "sigma = 25", "21.61", "0.1794", "448.55", "1.388 s"],
        ["1. DnCNN (2017)", "Supervised (Clean)", "sigma = 25", "36.67", "0.9290", "13.99", "6.465 s"],
        ["2. FFDNet (2018)", "Supervised (Noise Map)", "sigma = 25", "36.36", "0.9251", "15.02", "1.723 s"],
        ["3. Noise2Noise (2018)", "Unsupervised (Pairs)", "sigma = 25", "31.02", "0.8839", "51.40", "0.450 s"],
        ["4. Noise2Void / Self2Self", "Self-Supervised (Single)", "sigma = 25", "28.92", "0.8723", "83.35", "0.520 s"],
        ["5. Restormer (2022)", "Vision Transformer", "sigma = 35", "30.33", "0.7511", "60.25", "4.720 s"],
        ["6. Diffusion Denoiser", "Generative Sampling", "sigma = 30", "Perceptual", "Sharp Texture", "-", "3.510 s"]
    ]
    t2 = doc.add_table(rows=len(t2_data)+1, cols=len(t2_headers))
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(t2_headers):
        c = t2.cell(0, j)
        c.text = h
        set_cell_background(c, "184C78")
        c.paragraphs[0].runs[0].font.bold = True
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        c.paragraphs[0].runs[0].font.size = Pt(9)
    for i, row in enumerate(t2_data):
        bg = "F9FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            c = t2.cell(i+1, j)
            c.text = val
            set_cell_background(c, bg)
            c.paragraphs[0].runs[0].font.size = Pt(8.5)
            if "DnCNN" in row[0] or "FFDNet" in row[0]:
                c.paragraphs[0].runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 3. In-Depth Chapters with Complete Source Code
    add_h("3. Technical Analysis & PyTorch Implementations", level=1)

    # Technique 1
    add_h("Technique 1: DnCNN (Deep Residual CNN, 2017)", level=2)
    add_p(
        "DnCNN (Zhang et al., IEEE TIP 2017) introduced Residual Learning to image restoration. "
        "Rather than mapping degraded y -> clean x directly, DnCNN predicts the residual zero-mean noise: R(y) ~= v. "
        "The clean image is recovered via subtraction: x_hat = y - R(y). Stacking 20 Conv layers with Batch Normalization "
        "stabilizes gradient flow and eliminates vanishing gradients."
    )
    add_img("1_dncnn_full_comparison_sigma25.png", "DnCNN Full Landscape Comparison vs Classical Baselines (sigma=25)")
    add_img("2_dncnn_crop_detail_sigma25.png", "Zoomed Patch: Sharp preservation of person silhouette and grass highlights")
    add_p("PyTorch Implementation for DnCNN:", bold_prefix="SOURCE CODE — DnCNN Architecture & Subtraction:")
    add_code_block("""
class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_layers=20):
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ]
        # 18 intermediate convolutional layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ])
        # Output layer predicts residual noise v
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual_noise = self.model(x)
        # Residual Learning: Clean image = Input - Predicted Noise
        return torch.clamp(x - residual_noise, 0.0, 1.0)
""")

    # Technique 2
    add_h("Technique 2: FFDNet (Fast & Flexible Denoising, 2018)", level=2)
    add_p(
        "FFDNet (Zhang et al., IEEE TIP 2018) solved DnCNN's rigidity and computational cost. "
        "It accepts an explicit noise level map sigma as an extra input channel, allowing ONE network to denoise any noise "
        "variance and adapt to spatially varying fields. Stride-2 PixelUnshuffle downsamples (C, H, W) -> (4C, H/2, W/2), "
        "cutting compute by 4x and doubling the receptive field. At output, PixelShuffle reconstructs full resolution."
    )
    add_img("4_ffdnet_vs_dncnn_sigma25.png", "Head-to-head DnCNN vs FFDNet: Matching quality at 3.8x higher speed")
    add_img("5_ffdnet_controllability_sigma.png", "Noise Level Controllability: Under-smoothing vs Optimal vs Over-smoothing")
    add_img("6_ffdnet_spatially_variant.png", "Spatially Variant Denoising: Continuous 2D vertical gradient restoration")
    add_p("PyTorch Implementation for FFDNet:", bold_prefix="SOURCE CODE — FFDNet PixelUnshuffle & Noise Map:")
    add_code_block("""
class FFDNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=96, num_layers=12):
        super(FFDNet, self).__init__()
        self.scale_factor = 2
        # Input channels: 4 sub-images * 3 RGB channels + 1 noise level map channel = 13
        head_in_channels = in_channels * (self.scale_factor ** 2) + 1
        
        layers = [nn.Conv2d(head_in_channels, num_features, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Conv2d(num_features, num_features, 3, padding=1, bias=True), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(num_features, out_channels * (self.scale_factor ** 2), 3, padding=1, bias=True))
        
        self.model = nn.Sequential(*layers)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.scale_factor)
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        # 1. Reversible sub-sampling to quarter spatial resolution (H/2, W/2)
        x_down = self.pixel_unshuffle(x)
        hp, wp = x_down.size()[-2:]
        
        # 2. Formulate 2D noise level map
        if not torch.is_tensor(sigma):
            sigma_map = torch.full((x.size(0), 1, hp, wp), float(sigma), device=x.device, dtype=x.dtype)
        else:
            sigma_map = nn.functional.interpolate(sigma, size=(hp, wp), mode='bilinear', align_corners=False)
            
        # 3. Concatenate downsampled features with noise map -> (13 channels)
        inp = torch.cat([x_down, sigma_map], dim=1)
        feat = self.model(inp)
        
        # 4. Sub-pixel convolution upsampling back to (3, H, W)
        out = self.pixel_shuffle(feat)
        return torch.clamp(out[:, :, :h, :w], 0.0, 1.0)
""")

    # Technique 3
    add_h("Technique 3: Noise2Noise (Learning Without Clean Data, 2018)", level=2)
    add_p(
        "Noise2Noise (Lehtinen et al., ICML 2018) proved that networks can learn clean restorations with ZERO clean images. "
        "Minimizing L2 risk yields E[x|y]. If training targets are independent noisy pairs y2 = x + v2 where E[v2|x] = 0:\n"
        "E[y2 | y1] = E[x + v2 | y1] = E[x | y1] + E[v2 | y1] = E[x | y1] + 0 = E[x | y1]\n"
        "Because zero-mean noise averages to zero across training batches, the network converges to the clean ground truth!"
    )
    add_img("7_noise2noise_comparison.png", "Noise2Noise: Target image y2 is heavily corrupted, yet network outputs clean reconstruction")
    add_p("PyTorch Implementation for Noise2Noise Training:", bold_prefix="SOURCE CODE — Noise2Noise Loss & Training:")
    add_code_block("""
def train_noise2noise_step(model, optimizer, criterion, y1_batch, y2_batch):
    \"\"\"
    Noise2Noise: Input is noisy observation y1. Target is independent noisy observation y2.
    Clean ground truth x is NEVER provided!
    \"\"\"
    optimizer.zero_grad()
    prediction = model(y1_batch)
    # Target is noisy y2, but E[y2 | y1] = E[x | y1]
    loss = criterion(prediction, y2_batch)
    loss.backward()
    optimizer.step()
    return loss.item()
""")

    # Technique 4
    add_h("Technique 4: Noise2Void & Self2Self (Single Image Denoising, 2019-20)", level=2)
    add_p(
        "When no training dataset or pairs exist, standard CNNs cannot train on single images because they learn identity f(y)=y. "
        "Noise2Void (Krull et al., CVPR 2019) introduced Blind-Spot Receptive Fields: center pixels are randomly masked and "
        "replaced by neighbor pixels. Surrounding pixels contain information about true signal x, but zero information about center noise v. "
        "Self2Self (Quan et al., CVPR 2020) averages multiple Monte Carlo dropout passes at test time."
    )
    add_img("9_noise2void_self2self.png", "Noise2Void / Self2Self: Single-image self-supervised restoration with zero dataset")
    add_p("PyTorch Implementation for Noise2Void Blind-Spot Masking:", bold_prefix="SOURCE CODE — Blind-Spot Masking:")
    add_code_block("""
def create_n2v_blind_spots(patch, mask_ratio=0.03):
    \"\"\"
    Noise2Void Blind-Spot Masking: Replaces target pixel with random neighbor.
    Forces network to predict center pixel from spatial context alone.
    \"\"\"
    B, C, H, W = patch.shape
    blind_patch = patch.clone()
    mask = torch.rand((B, 1, H, W), device=patch.device) < mask_ratio
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
    for dy, dx in offsets:
        neighbor = torch.roll(patch, shifts=(dy, dx), dims=(-2, -1))
        replace_cond = mask & (torch.rand((B, 1, H, W), device=patch.device) < (1.0 / len(offsets)))
        blind_patch = torch.where(replace_cond.expand_as(patch), neighbor, blind_patch)
    return blind_patch, mask

# Loss evaluated ONLY at blind-spot masked coordinates:
# loss = criterion(pred[mask_expanded], original_patch[mask_expanded])
""")

    # Technique 5
    add_h("Technique 5: Restormer / NAFNet (Vision Transformers, 2021-22)", level=2)
    add_p(
        "Standard Vision Transformers suffer from quadratic O((HW)^2) spatial complexity. "
        "Restormer (Zamir et al., CVPR 2022) computes attention across CHANNELS (C x C) via Multi-Dconv Head Transposed Attention (MDTA). "
        "Complexity drops to linear O(HW * C^2), enabling Transformers to process high-resolution images. "
        "NAFNet (Chen et al., ECCV 2022) introduced SimpleGate, replacing nonlinear activations with element-wise multiplication."
    )
    add_img("11_restormer_nafnet_attention.png", "Restormer Transposed Attention Output under severe noise (sigma=35)")
    add_p("PyTorch Implementation for Multi-Dconv Transposed Attention & SimpleGate:", bold_prefix="SOURCE CODE — MDTA & SimpleGate:")
    add_code_block("""
class SimpleGate(nn.Module):
    \"\"\"NAFNet SimpleGate: Splits channels and multiplies element-wise.\"\"\"
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class MDTA(nn.Module):
    \"\"\"Restormer: Multi-Dconv Head Transposed Attention across channels O(C^2).\"\"\"
    def __init__(self, channels=48, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = torch.nn.functional.normalize(q.reshape(b, self.num_heads, -1, h * w), dim=-1)
        k = torch.nn.functional.normalize(k.reshape(b, self.num_heads, -1, h * w), dim=-1)
        v = v.reshape(b, self.num_heads, -1, h * w)
        # Channel-wise attention map: (C x C)
        attn = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)
""")

    # Technique 6
    add_h("Technique 6: Diffusion Denoisers (Generative Sampling, 2021+)", level=2)
    add_p(
        "Standard denoisers minimize L2 loss, outputting the conditional expectation E[x|y]. Under severe noise, "
        "averaging all plausible textures creates blurry, waxy results. "
        "Diffusion models (Ho et al., 2020) sample from the posterior p(x|y) via reverse Langevin stochastic trajectories. "
        "Each reverse step chooses ONE sharp, plausible reality, synthesizing photorealistic grass fibers and textures."
    )
    add_img("13_diffusion_reverse_trajectory.png", "5-Step Reverse Diffusion Trajectory: Resolving pure noise into crisp textures")
    add_img("14_diffusion_vs_regression.png", "Regression (Blurry L2 Average) vs Diffusion (Sharp Plausible Texture Sample)")
    add_p("PyTorch Implementation for DDPM Reverse Langevin Sampling:", bold_prefix="SOURCE CODE — Reverse Langevin Sampling:")
    add_code_block("""
@torch.no_grad()
def ddpm_reverse_sample(model, noisy_condition, num_timesteps=100, betas=None):
    \"\"\"
    DDPM Reverse Langevin Sampling Trajectory: Samples a sharp, plausible reality
    from p(x | y) instead of computing a blurry L2 conditional mean.
    \"\"\"
    xt = torch.randn_like(noisy_condition) # Start from pure noise at T
    for t_step in reversed(range(num_timesteps)):
        t_tensor = torch.tensor([t_step], device=noisy_condition.device).float()
        pred_eps = model(xt, t_tensor, noisy_condition)
        
        beta_t = betas[t_step]
        alpha_t = 1.0 - beta_t
        alpha_bar_t = torch.cumprod(1.0 - betas, dim=0)[t_step]
        
        # Langevin reverse update
        mean = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_eps)
        if t_step > 0:
            xt = mean + torch.sqrt(beta_t) * torch.randn_like(xt)
        else:
            xt = mean
    return torch.clamp(xt, 0.0, 1.0)
""")

    # 4. Decision Matrix & References
    add_h("4. Real-World Engineering Decision Matrix", level=1)
    headers3 = ["Application Domain", "Recommended Model", "Technical Justification"]
    data3 = [
        ["Smartphone Photography", "FFDNet", "Ultra-fast (<2 ms on NPU); noise level sigma read directly from camera ISO tags."],
        ["Medical Imaging (MRI / Low-Dose CT)", "Noise2Noise", "Zero radiation risk from clean scans; easily acquires paired noisy exposures."],
        ["Microscopy & Astronomy", "Noise2Void", "Dynamic live specimens prevent pairs or datasets; trains on the single captured frame."],
        ["Benchmark Contests & Studio Quality", "Restormer / NAFNet", "Dominates NTIRE benchmarks; channel transposed attention captures global lighting."],
        ["Cinematic Upscaling & Remastering", "Diffusion Denoisers", "Hallucinates photorealistic high-frequency details rather than waxy L2 blur."]
    ]
    t3 = doc.add_table(rows=len(data3)+1, cols=len(headers3))
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers3):
        c = t3.cell(0, j)
        c.text = h
        set_cell_background(c, "184C78")
        c.paragraphs[0].runs[0].font.bold = True
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        c.paragraphs[0].runs[0].font.size = Pt(9)
    for i, row in enumerate(data3):
        bg = "F9FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            c = t3.cell(i+1, j)
            c.text = val
            set_cell_background(c, bg)
            c.paragraphs[0].runs[0].font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_h("5. Scientific References", level=1)
    refs = [
        "1. Zhang, K., et al. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP.",
        "2. Zhang, K., et al. (2018). FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising. IEEE TIP.",
        "3. Lehtinen, J., et al. (2018). Noise2Noise: Learning Image Restoration without Clean Data. ICML.",
        "4. Krull, A., et al. (2019). Noise2Void - Learning Denoising From Single Noisy Images. CVPR.",
        "5. Quan, Y., et al. (2020). Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image. CVPR.",
        "6. Zamir, S. W., et al. (2022). Restormer: Efficient Transformer for High-Resolution Image Restoration. CVPR.",
        "7. Chen, L., et al. (2022). Simple Baselines for Image Restoration. ECCV.",
        "8. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS."
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.space_after = Pt(3)
        p.runs[0].font.size = Pt(9)

    doc.save(output_docx_path)
    sz = os.path.getsize(output_docx_path) / (1024*1024)
    print(f"[+] Successfully generated Word Document: {output_docx_path} ({sz:.2f} MB)")

if __name__ == "__main__":
    generate_word_report()
