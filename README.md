# Digital Image Enhancement — Section 2.1: Point Transformations ($s = T(r)$)

This repository contains the complete implementation, interactive Jupyter Notebook, and quantitative evaluation for **Section 2.1: Point Transformations** from Module 2 (Classical Image Enhancement, Dr. Guindo).

---

## 📌 What is a Point Transformation? (Slide 11)

A **point operation** processes **one pixel at a time** using a mapping function:
$$s = T(r)$$
- $r$ is the input intensity value, $s$ is the output value.
- **Pixel position and neighboring pixels do NOT enter the formula.**

---

## 🛠️ The 7 Core Point Transformations Implemented

1. **Image Negative** ($s = 255 - r$): Inverts black and white. Used in mammograms/X-rays to reveal bright specks inside dark fields.
2. **Linear Transform** ($s = \alpha \cdot r + \beta$): $\alpha$ adjusts contrast; $\beta$ adjusts brightness (with clipping $[0, 255]$).
3. **Contrast Stretching**:
   - Ordinary Min-Max Stretch ($s = 255 \frac{r - r_{\min}}{r_{\max} - r_{\min}}$)
   - Robust Percentile Stretch ($1\%-99\%$) to prevent sensor hot pixels from disabling the stretch.
4. **Log Transform** ($s = c \cdot \log(1 + r)$): Expands low-intensity dark tones while compressing highlights.
5. **Gamma Correction (Power Law)** ($s = 255 (r \div 255)^\gamma$):
   - $\gamma < 1$ (e.g. $\gamma=0.4$): Brightens shadow details.
   - $\gamma > 1$ (e.g. $\gamma=2.0$): Darkens over-exposed highlights.
6. **Thresholding** ($s = 255$ if $r > T$, else $0$): Binarizes an image into a mask.
7. **Histogram Equalization** ($s = \text{round}((L - 1) \cdot \text{cdf}(r))$): Automatic contrast enhancement using the CDF.
8. **Bit-Plane Slicing**: Decomposes 8-bit pixels into 8 binary planes (Bits 7 to 0).

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/sc235/ImageEnhancement-Assignment.git
cd ImageEnhancement-Assignment

# Install dependencies
pip install opencv-python numpy matplotlib pillow

# Run python script
python run_assignment.py

# Or launch Jupyter Notebook
jupyter notebook Image_Enhancement_Assignment.ipynb
```

---

## 📊 Section 2.1 Point Operations Results Table

| Point Operation / Method | Mean | Standard Dev ($\sigma$, Contrast) | Entropy (bits) | PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **Original Reference** | 112.16 | 62.01 | 7.76 | $\infty$ |
| **Infected: Low Contrast** | 46.52 | 12.14 | 5.41 | 9.81 |
| **Infected: Low Light** | 16.36 | 9.23 | 5.02 | 7.35 |
| **Infected: Overexposed** | 216.03 | 42.33 | 5.25 | 7.48 |
| **1. Image Negative ($s=255-r$)** | 142.84 | 62.01 | 7.76 | 6.00 |
| **2. Linear Boost ($\alpha=1.8, \beta=-40$)** | 43.30 | 21.89 | 6.22 | 10.10 |
| **3. Min-Max Contrast Stretch** | 109.20 | 62.04 | 7.65 | **38.22** |
| **3. Percentile Stretch ($1\%-99\%$)** | 118.30 | **70.39** | **7.76** | **27.74** |
| **4. Log Transform** | 122.02 | 30.77 | 6.53 | **17.05** |
| **5. Gamma Correction ($\gamma=0.4$)** | 80.24 | 21.73 | 6.19 | 13.80 |
| **5. Gamma Correction ($\gamma=2.0$)** | 189.96 | 67.91 | 5.63 | 9.88 |
| **6. Thresholding ($T=128$)** | 103.77 | 121.62 | 1.46 | 11.08 |
| **7. Global Histogram Equalization** | 130.45 | 73.22 | 5.41 | 21.00 |
