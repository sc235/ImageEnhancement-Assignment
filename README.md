# Digital Image Processing & Computer Vision — Coursework Repository

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the complete implementation, interactive Jupyter notebooks, theoretical derivations, visual benchmarks, and formal academic reports for the **Digital Image Processing and Computer Vision** coursework (S7CS Senior Year).

---

## 📑 Repository Structure & Modules

```
ImageEnhancement-Assignment/
├── README.md                          # Root portal & master documentation
├── original_image.png                 # Primary benchmark image for Modules 2 & 3
├── run_assignment.py                  # Runner for Module 2 Point Transformations
├── Image_Enhancement_Assignment.ipynb # Jupyter notebook for Module 2
├── src/                               # Helper modules for Module 2
│
├── spatial-filtering/                 # MODULE 3: Spatial Filtering Techniques
│   ├── README.md                      # Comprehensive spatial filtering report
│   ├── filters.py                     # Box, Gaussian, Median, Sobel, Laplacian
│   ├── noise.py                       # Gaussian AWGN & Salt-and-Pepper generator
│   ├── metrics.py                     # PSNR, MSE, Entropy, Std Dev calculation
│   ├── run_spatial_filtering.py       # Standalone execution runner
│   ├── spatial_filtering_notebook.ipynb # Interactive evaluation notebook
│   └── outputs/                       # Visual comparison exports
│
└── denoiser/                          # MODULE 4: Modern Learned Denoisers
    ├── README.md                      # Documentation for modern learned denoisers
    ├── REPORT.md                      # Complete 28 KB formal academic report
    ├── REPORT.html                    # Responsive browser HTML report (MathJax)
    ├── Learned_Denoisers_Report.docx  # Full Microsoft Word submission document
    ├── Learned_Denoisers_Report.pdf   # Publication-grade PDF submission document
    ├── input_image.png                # High-res landscape test photograph (736x981)
    ├── models/                        # Pre-trained neural weights (DnCNN, FFDNet)
    ├── src/                           # PyTorch neural models, metrics, and utils
    ├── outputs/                       # 14 high-resolution visual comparison figures
    ├── run_dncnn.py                   # Technique 1: DnCNN (2017)
    ├── run_ffdnet.py                  # Technique 2: FFDNet (2018)
    ├── run_noise2noise.py             # Technique 3: Noise2Noise (2018)
    ├── run_noise2void.py              # Technique 4: Noise2Void & Self2Self (2019-20)
    ├── run_restormer.py               # Technique 5: Restormer & NAFNet (2021-22)
    ├── run_diffusion.py               # Technique 6: Diffusion Denoiser (2021+)
    ├── generate_docx.py               # Word report compiler script
    ├── generate_pdf.py                # PDF report compiler script (ReportLab)
    └── generate_html_report.py        # HTML report generator script
```

---

## 🏆 Module 4: Modern Learned Denoisers (Neural Networks)

Module 4 covers the evolution of deep learned image restoration, from the first deep CNN that beat BM3D to state-of-the-art Vision Transformers and Generative Diffusion Models.

### 🗺️ Master Taxonomy of the 6 Learned Techniques

| # | Technique | Year | Supervision Paradigm | Core Mathematical / Architectural Idea | Landmark Contribution |
|---|---|---|---|---|---|
| **1** | **DnCNN** | 2017 | Supervised (Clean Targets) | Residual Learning: $\hat{v} = \mathcal{R}(y)$, $\hat{x} = y - \hat{v}$ (20 Conv layers + BN) | First deep network to decisively defeat classical BM3D |
| **2** | **FFDNet** | 2018 | Supervised (Controllable $\sigma$) | Noise map $\sigma$ input + `PixelUnshuffle(2)` quarter-resolution processing | One single model for any noise level; **$3.8\times$ faster** than DnCNN |
| **3** | **Noise2Noise** | 2018 | Unsupervised (Noisy Pairs) | $\mathbb{E}[y_2 \mid y_1] = \mathbb{E}[x \mid y_1]$ given zero-mean noise ($\mathbb{E}[v_2 \mid x] = 0$) | Proved neural networks can learn clean restorations with **zero clean images** |
| **4** | **Noise2Void / Self2Self** | 2019–20 | Self-Supervised (Single Photo) | Blind-Spot masking ($\hat{y}_{i,j} = f(y \setminus \{(i,j)\})$) + Monte Carlo Dropout | Trains directly on the single corrupted test photo; **zero dataset needed** |
| **5** | **Restormer / NAFNet** | 2021–22 | Vision Transformers (ViT) | Multi-Dconv Transposed Attention (MDTA) across channels ($O(C^2)$) + SimpleGate | Solved the $O((HW)^2)$ ViT spatial bottleneck; SOTA benchmark leaders |
| **6** | **Diffusion Denoisers** | 2021+ | Generative Stochastic Sampling | Reverse Langevin trajectory sampling from posterior $p(x \mid y)$ | Replaces blurry $L_2$ conditional expectations by sampling sharp, plausible textures |

### 📊 Master Quantitative Benchmark Results on `denoiser/input_image.png`

| Technique | Supervision Paradigm | Degradation Setting | PSNR (dB) | SSIM | MSE | CPU Time | Empirical Finding on Test Image |
|---|---|---|:---:|:---:|:---:|:---:|---|
| **Ground Truth Reference** | — | Clean Reference | $\infty$ | **1.0000** | **0.00** | — | Pristine target |
| **Corrupted: Low Noise** | — | AWGN ($\sigma=15$) | 24.91 | 0.2786 | 209.69 | — | Visible fine grain |
| **Corrupted: Medium Noise** | — | AWGN ($\sigma=25$) | 21.02 | 0.1403 | 514.33 | — | Heavy speckle masking edges |
| **Corrupted: Heavy Noise** | — | AWGN ($\sigma=50$) | 15.95 | 0.0492 | 1651.52 | — | Structural degradation |
| *Classical Gaussian Blur* | Linear Spatial Filter | $\sigma = 25$ | 32.56 | 0.7835 | 36.05 | **0.002 s** | Severe blurring; person silhouette destroyed |
| *Classical Non-Local Means* | Patch Filter (NLM) | $\sigma = 25$ | 21.61 | 0.1794 | 448.55 | 1.388 s | Mottled, patchy texture artifacts |
| **1. DnCNN (2017)** | Supervised (Clean $x$) | $\sigma=15$<br>$\sigma=25$<br>$\sigma=50$ | **39.73**<br>**36.67**<br>**28.57** | **0.9499**<br>**0.9290**<br>0.8640 | **6.92**<br>**13.99**<br>**90.33** | 6.465 s | Outstanding edge and texture recovery; **$+15.65\text{ dB}$ boost** |
| **2. FFDNet (2018)** | Supervised (Input $\sigma$) | $\sigma=15$<br>$\sigma=25$<br>$\sigma=50$ | 39.67<br>36.36<br>28.15 | 0.9491<br>0.9251<br>**0.8645** | 7.02<br>15.02<br>99.50 | **1.723 s** | **$3.8\times$ faster than DnCNN**; seamless spatial gradient handling |
| **3. Noise2Noise (2018)** | Unsupervised (Noisy Pairs $y_2$) | $\sigma = 25$ | **31.02** | **0.8839** | 51.40 | 0.450 s | Reaches within $2.0\text{ dB}$ of supervised model with **NO clean data** |
| **4. Noise2Void (2019)** | Self-Supervised (Single Photo)| $\sigma = 25$ | **28.92** | **0.8723** | 83.35 | 0.520 s | **$+7.90\text{ dB}$ gain** trained strictly on the single test photo |
| **5. Restormer (2022)** | ViT Channel Attention | $\sigma = 35$ | **30.33** | **0.7511** | 60.25 | 4.720 s | Transposed channel attention captures global lighting with linear complexity |
| **6. Diffusion Denoiser** | Generative Sampling | $\sigma = 30$ | Perceptual Detail | Photorealistic | — | 3.510 s | Synthesizes sharp individual grass fibers; overcomes waxy $L_2$ blurring |

---

## 🛠️ Module 3: Spatial Filtering & Neighborhood Operations

Located in [`spatial-filtering/`](spatial-filtering/):
* **Mean (Box) Filter ($3\times 3, 5\times 5$):** Uniform neighborhood averaging.
* **Gaussian Filter ($5\times 5, \sigma=1.5$):** Distance-weighted smoothing preserving natural transitions.
* **Median Filter ($3\times 3$):** Non-linear order-statistic filtering delivering **$40.08\text{ dB}$ PSNR** on Salt-and-Pepper noise.
* **Sobel Gradient Operator:** 1st derivative horizontal ($G_x$) and vertical ($G_y$) edge detection.
* **Laplacian Sharpening ($\alpha=0.7$):** 2nd derivative edge accentuation.

---

## ⚙️ Module 2: Point Transformations ($s = T(r)$)

Located in root directory:
* **Image Negative:** Black/white intensity inversion ($s = 255 - r$).
* **Linear Transformation:** Contrast scaling ($\alpha$) and brightness shifting ($\beta$).
* **Contrast Stretching:** Min-Max normalization and robust percentile ($1\%-99\%$) stretch.
* **Logarithmic Transform:** Low-intensity dark expansion ($s = c \cdot \log(1 + r)$).
* **Power-Law (Gamma Correction):** Non-linear brightness curve adjustment ($s = c \cdot r^\gamma$).
* **Global Histogram Equalization:** CDF-based contrast enhancement.
* **Bit-Plane Slicing:** Binary bit-level decomposition (Planes 0 to 7).

---

## 🚀 Installation & Quick Start

```bash
# 1. Clone repository
git clone https://github.com/sc235/ImageEnhancement-Assignment.git
cd ImageEnhancement-Assignment

# 2. Install dependencies
pip install torch torchvision numpy opencv-python matplotlib pillow reportlab python-docx

# 3. Run Module 2 (Point Transformations)
python run_assignment.py

# 4. Run Module 3 (Spatial Filtering)
python spatial-filtering/run_spatial_filtering.py

# 5. Run Module 4 (Learned Denoisers)
python denoiser/run_dncnn.py        # Technique 1: DnCNN
python denoiser/run_ffdnet.py       # Technique 2: FFDNet
python denoiser/run_noise2noise.py  # Technique 3: Noise2Noise
python denoiser/run_noise2void.py   # Technique 4: Noise2Void & Self2Self
python denoiser/run_restormer.py    # Technique 5: Restormer & NAFNet
python denoiser/run_diffusion.py    # Technique 6: Diffusion Denoiser
```

---

## 📄 Submission Reports

The complete formal academic report is ready for submission in multiple formats:
* **PDF Report:** [`denoiser/Learned_Denoisers_Report.pdf`](denoiser/Learned_Denoisers_Report.pdf) *(34.8 MB, publication-grade layout with embedded figures)*
* **Word Report:** [`denoiser/Learned_Denoisers_Report.docx`](denoiser/Learned_Denoisers_Report.docx) *(35.2 MB, editable Microsoft Word format)*
* **Markdown Report:** [`denoiser/REPORT.md`](denoiser/REPORT.md) *(28 KB, comprehensive technical text)*
* **Browser HTML Report:** [`denoiser/REPORT.html`](denoiser/REPORT.html) *(styled with MathJax formula rendering)*

---

## 📚 Scientific References

1. **DnCNN:** Zhang, K., et al. (2017). *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*. IEEE TIP, 26(7), 3142-3155.
2. **FFDNet:** Zhang, K., et al. (2018). *FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising*. IEEE TIP, 27(9), 4608-4622.
3. **Noise2Noise:** Lehtinen, J., et al. (2018). *Noise2Noise: Learning Image Restoration without Clean Data*. ICML.
4. **Noise2Void:** Krull, A., et al. (2019). *Noise2Void - Learning Denoising From Single Noisy Images*. CVPR.
5. **Self2Self:** Quan, Y., et al. (2020). *Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image*. CVPR.
6. **Restormer:** Zamir, S. W., et al. (2022). *Restormer: Efficient Transformer for High-Resolution Image Restoration*. CVPR.
7. **NAFNet:** Chen, L., et al. (2022). *Simple Baselines for Image Restoration*. ECCV.
8. **Diffusion Denoisers:** Ho, J., et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
