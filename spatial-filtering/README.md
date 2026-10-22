# Digital Image Processing — Spatial Filtering Module

This folder contains the complete implementation, interactive Jupyter notebook, visual comparisons, and quantitative benchmark evaluation for **Spatial Filtering Techniques** operating on [`original_image.png`](../original_image.png).

---

## 📌 The 5 Core Spatial Filters Implemented

### 1. Mean Filter (Box Filter) — Linear Smoothing
* **Formula:** $g(x, y) = \frac{1}{m^2} \sum_{s} \sum_{t} f(x+s, y+t)$
* **Kernel ($3 \times 3$):** $\frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$
* **Role:** Averages neighboring pixel intensities to smooth uniform noise.

### 2. Gaussian Filter — Weighted Linear Smoothing
* **Formula:** $h(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$
* **Role:** Assigns higher weights to pixels near the center, delivering smooth natural blurring without boxy edge artifacts.

### 3. Median Filter — Non-Linear Order-Statistic Filter
* **Formula:** $g(x, y) = \text{median}\{f(s, t) \mid (s, t) \in S_{xy}\}$
* **Role:** Sorts local neighborhood pixels and picks the median. **Outstanding performance against Salt-and-Pepper noise** while preserving sharp edges.

### 4. Sobel Filter — 1st Derivative Edge Detection
* **Kernels:**
  $$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
* **Gradient Magnitude:** $M(x, y) = \sqrt{G_x^2 + G_y^2}$
* **Role:** Measures intensity rate of change to extract horizontal and vertical edge boundaries.

### 5. Laplacian Filter — 2nd Derivative Sharpening
* **Formula:** $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$
* **Kernel ($3 \times 3$):** $\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 0 \\ 0 & 1 & 0 \end{bmatrix}$
* **Sharpened Output:** $g(x, y) = f(x, y) - \alpha \cdot \nabla^2 f(x, y)$
* **Role:** Enhances high-frequency spatial details and sharpens subtle edge boundaries.

---

## 📊 Quantitative Evaluation Results Table

Evaluated on [`original_image.png`](../original_image.png) ($736 \times 920$):

| Spatial Filtering Method | Mean | Std Dev (Contrast) | Entropy (bits) | MSE | PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Original Reference** | 112.16 | 62.01 | 7.76 | 0.00 | $\infty$ |
| **Infected: Gaussian Noise ($\sigma=25$)** | 111.92 | 66.12 | 7.86 | 594.92 | 20.39 |
| **Infected: Salt-and-Pepper Noise (6%)** | 112.90 | 67.63 | 7.69 | 1183.40 | 17.40 |
| **1. Mean Filter ($3 \times 3$ Box)** | 111.92 | 61.78 | 7.76 | 74.66 | **29.40** |
| **1. Mean Filter ($5 \times 5$ Box)** | 111.92 | 61.16 | 7.73 | 44.30 | **31.67** |
| **2. Gaussian Filter ($5 \times 5, \sigma=1.5$)** | 111.92 | 61.31 | 7.73 | 43.98 | **31.70** |
| **3. Mean Filter on S&P Noise** | 112.90 | 59.23 | 7.74 | 152.06 | 26.31 |
| **3. Median Filter on S&P Noise** | 112.14 | 61.91 | 7.75 | **6.39** | **40.08** |
| **4. Sobel Gradient Magnitude** | 10.88 | 12.92 | 4.81 | 14384.74 | 6.55 |
| **5. Laplacian Sharpened ($\alpha=0.7$)** | 112.08 | **65.95** | **7.85** | 303.40 | 23.31 |

---

## 🖼️ Visual Generated Outputs

All generated figures are saved in the [`outputs/`](outputs/) folder:

1. [`1_smoothing_gaussian_noise.png`](outputs/1_smoothing_gaussian_noise.png) — Comparing Box Filter vs. Gaussian Filter on additive noise.
2. [`2_smoothing_salt_pepper.png`](outputs/2_smoothing_salt_pepper.png) — Demonstrating Median Filter's superiority over Mean Filter on impulse noise.
3. [`3_sobel_edge_detection.png`](outputs/3_sobel_edge_detection.png) — Displaying $G_x, G_y$, Magnitude, and Orientation Angle.
4. [`4_laplacian_sharpening.png`](outputs/4_laplacian_sharpening.png) — Displaying Laplacian 2D derivative response map and sharpened result.
5. [`5_all_5_filters_summary.png`](outputs/5_all_5_filters_summary.png) — Master collage comparing all 5 filters.

---

## 🚀 How to Run

```bash
# Navigate to the spatial-filtering directory
cd spatial-filtering

# Run the Python evaluation pipeline
python run_spatial_filtering.py

# Or open the interactive Jupyter Notebook
jupyter notebook spatial_filtering_notebook.ipynb
```
