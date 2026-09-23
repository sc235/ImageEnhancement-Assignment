# Digital Image Enhancement & Restoration Assignment

This repository contains the complete implementation and report for the **Digital Image Enhancement Assignment** (Module 2: Classical Image Enhancement, Dr. Guindo).

## 📌 Features & Implemented Techniques

1. **Forward Degradation / Infection Models (`src/degradation.py`)**:
   - Low-Contrast range squashing $[25, 75]$
   - Low-Light underexposure ($0.15 \times \text{img}$)
   - Impulse Salt-and-Pepper noise ($4\%$)
   - Over-exposure shift ($+110$) with highlight clipping at $255$

2. **Point Transformations ($s = T(r)$) (`src/point_transforms.py`)**:
   - Image Negative ($s = 255 - r$)
   - Linear Contrast & Brightness adjustment
   - Min-Max & Robust Percentile Stretching ($1\%-99\%$)
   - Logarithmic Transform ($s = c \log(1+r)$)
   - Gamma Correction ($\gamma = 0.4$ shadows, $\gamma = 2.0$ highlights)
   - Thresholding & 8 Binary Bit-Plane Slicing

3. **Histogram Equalization & CLAHE (`src/histogram_ops.py`)**:
   - Global Histogram Equalization (HE) using Cumulative Distribution Functions (CDF)
   - Contrast-Limited Adaptive Histogram Equalization (CLAHE) operating on the $L$ (Lightness) channel in **$Lab$ color space** to preserve natural color fidelity.

4. **Spatial Neighborhood Filtering (`src/spatial_ops.py`)**:
   - Mean ($5\times5$), Gaussian ($5\times5$), and Median ($5\times5$) filters.
   - Unsharp Masking & Laplacian Sharpening.

5. **Quantitative Evaluation (`src/utils.py`)**:
   - Mean Intensity, Standard Deviation ($\sigma$, Contrast measure), Entropy (in bits), and Peak Signal-to-Noise Ratio (PSNR in dB).

---

## 🛠️ Installation & Execution

```bash
# Clone the repository
git clone https://github.com/sc235/ImageEnhancement-Assignment.git
cd ImageEnhancement-Assignment

# Install dependencies
pip install opencv-python numpy matplotlib pillow

# Run the complete assignment pipeline
python run_assignment.py
```

All generated output comparison figures and histograms are automatically saved to the `outputs/` folder.

---

## 📊 Summary Results Table

| Method / Image State | Mean | Standard Dev ($\sigma$) | Entropy (bits) | PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **Original Reference** | 112.16 | 62.01 | 7.76 | $\infty$ |
| **Infected: Low Contrast** | 46.52 | 12.14 | 5.41 | 9.81 |
| **Percentile Stretch ($1\%-99\%$)** | 118.30 | **70.39** | **7.76** | **27.74** |
| **Global Equalization (HE)** | 130.45 | 73.22 | 5.41 | 21.00 |
| **CLAHE ($Lab$ $L$-channel)** | 69.70 | 29.91 | 6.73 | 13.30 |
| **Infected: Low Light** | 16.36 | 9.23 | 5.02 | 7.35 |
| **Log Transform** | 122.02 | 30.77 | 6.53 | **17.05** |
| **Gamma Correction ($\gamma=0.4$)** | 80.24 | 21.73 | 6.19 | 13.80 |
| **Infected: S&P Noise** | 113.46 | 72.51 | 7.50 | 14.54 |
| **Median Denoised ($5 \times 5$)** | 112.09 | **61.81** | **7.75** | **36.27** |
