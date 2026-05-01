"""
IV.   Principal Component Analysis
This homework project allows to implement and understand the merits of the PCA on multispectral images.  
This document shows the results of applying PCA algorithm over multispectral satellite Images. 
In particular, we selected an 8-band image (i.e. coastal, blue, green, yellow, red, red edge, NIR1 and NIR2); 
see below their respective wavelength intervals, in which we can see different geographic properties like 
a river, forest (trees), building structures, green space, etc. This image is 16-bit depth and its location 
is Adelaide, Australia. Please see below a representation of the RGB channels of the 8-band image.
Spectral bands considered include:
Coastal band [400–450 nm],
Blue [450–510 nm],
Green (510–580 nm)
Yellow [585–625 nm],
Red [630–690 nm]
Red-edge [705–745 nm]
Near-infrared 1, NIR1 [770–895 nm]
Near-infrared 2, NIR2; [860–1040 nm]
Given these 8-band image, construct a matrix of, where    and  and  are 
the corresponding width and height of the input image, with  representing the 8 dimensions or wavebands. 
Hence the rows of this matrix correspond to observations and columns correspond to variables.
Compute the covariance matrix
Determine the eigenvalues and eigenvectors (principal components) of the covariance matrix.
Through the inverse PCA, reconstruct the original image by keeping
each of the Principal Component (PC) starting from the first to the eight PC and see what you get in 
each of these reconstructed mages.
Reconstruct the input image from the eigenvectors of the highest 4 eigenvalues
Reconstruct the input image from the eigenvectors of the lowest 4 eigenvalues
Subtract the original image from the reconstructed image you obtained in (b)
Subtract the original image from the reconstructed image you obtained in (c)
Provide your assessment or concluding remarks on these reconstructed images and on the subtractions performed
"""
"""
IV. Principal Component Analysis on Multispectral Satellite Images
==================================================================
8-band image: Coastal, Blue, Green, Yellow, Red, Red-edge, NIR1, NIR2
Location: Adelaide, Australia | Bit depth: 16-bit
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

IMAGE_PATH = r"C:\Users\danie\EEL6821-Computer-Vision-Assignments\Assignment 4 Images\14NOV27004452-M2AS-054191978040_01_P001.TIF"

BAND_NAMES = ["Coastal", "Blue", "Green", "Yellow", "Red", "Red-edge", "NIR1", "NIR2"]

OUTPUT_DIR = "pca_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with rasterio.open(IMAGE_PATH) as src:
    data = src.read().astype(np.float64)

image = np.transpose(data, (1, 2, 0))
H, W, num_bands = image.shape
N = H * W

print(f"Image shape: {H} x {W} x {num_bands}")
print(f"Observations (N): {N:,}  |  Variables (p): {num_bands}")

X = image.reshape(N, num_bands)

mean_vec  = X.mean(axis=0)
X_centred = X - mean_vec

C = (X_centred.T @ X_centred) / (N - 1)

print("\nCovariance Matrix (8x8):")
print(np.array2string(C, precision=2, suppress_small=True))

eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(C)

idx = np.argsort(eigenvalues_raw)[::-1]
eigenvalues  = eigenvalues_raw[idx]
eigenvectors = eigenvectors_raw[:, idx]

explained = eigenvalues / eigenvalues.sum() * 100
cumulative = np.cumsum(explained)

print(f"\n{'PC':<5} {'Eigenvalue':>14} {'Explained%':>12} {'Cumulative%':>13}")
print("-" * 46)
for i in range(num_bands):
    print(f"PC{i+1:<3} {eigenvalues[i]:>14.2f} {explained[i]:>11.2f}% {cumulative[i]:>12.2f}%")

Z = X_centred @ eigenvectors

def reconstruct(Z, eigenvectors, mean_vec, pc_indices):
    Z_sub = Z[:, pc_indices]
    V_sub = eigenvectors[:, pc_indices]
    X_rec = Z_sub @ V_sub.T + mean_vec
    return X_rec.reshape(H, W, num_bands)

def to_display(arr):
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

def make_rgb(img):
    return to_display(np.stack([img[:,:,4], img[:,:,2], img[:,:,1]], axis=-1))


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for k in range(1, num_bands + 1):
    rec = reconstruct(Z, eigenvectors, mean_vec, list(range(k)))
    axes[k-1].imshow(make_rgb(rec))
    axes[k-1].set_title(f"PC1–PC{k}  (cum. {cumulative[k-1]:.1f}%)", fontsize=9)
    axes[k-1].axis("off")

fig.suptitle("Inverse PCA: Cumulative Reconstructions (PC1 → PC8)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_cumulative_reconstructions.png"), dpi=150)
plt.show()

rec_top4 = reconstruct(Z, eigenvectors, mean_vec, [0, 1, 2, 3])

rec_bot4 = reconstruct(Z, eigenvectors, mean_vec, [4, 5, 6, 7])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(make_rgb(image))
axes[0].set_title("Original Image", fontsize=11)
axes[0].axis("off")

axes[1].imshow(make_rgb(rec_top4))
axes[1].set_title(f"Top-4 PCs  (PC1–PC4)\nCum. variance: {cumulative[3]:.1f}%", fontsize=11)
axes[1].axis("off")

axes[2].imshow(make_rgb(rec_bot4))
axes[2].set_title("Bottom-4 PCs  (PC5–PC8)\nResidual variance", fontsize=11)
axes[2].axis("off")

fig.suptitle("Top-4 vs Bottom-4 PCs Reconstructions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_top4_vs_bot4.png"), dpi=150)
plt.show()

diff_top4 = image - rec_top4
diff_bot4 = image - rec_bot4

fig, axes = plt.subplots(2, 8, figsize=(24, 6))

for b in range(num_bands):
    vmax = np.percentile(np.abs(diff_top4[:,:,b]), 99)
    axes[0, b].imshow(diff_top4[:,:,b], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, b].set_title(BAND_NAMES[b], fontsize=8)
    axes[0, b].axis("off")

    vmax = np.percentile(np.abs(diff_bot4[:,:,b]), 99)
    axes[1, b].imshow(diff_bot4[:,:,b], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1, b].set_title(BAND_NAMES[b], fontsize=8)
    axes[1, b].axis("off")

axes[0, 0].set_ylabel("Original − Top-4", fontsize=9)
axes[1, 0].set_ylabel("Original − Bottom-4", fontsize=9)

fig.suptitle("Difference Images per Band  (Original − Reconstruction)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_difference_images.png"), dpi=150)
plt.show()

def rmse(orig, rec):
    return np.sqrt(((orig - rec) ** 2).mean(axis=(0, 1)))

rmse_top4 = rmse(image, rec_top4)
rmse_bot4 = rmse(image, rec_bot4)

print(f"\n{'Band':<12} {'RMSE Top-4':>12} {'RMSE Bot-4':>12}")
print("-" * 38)
for b in range(num_bands):
    print(f"{BAND_NAMES[b]:<12} {rmse_top4[b]:>12.2f} {rmse_bot4[b]:>12.2f}")
print(f"{'MEAN':<12} {rmse_top4.mean():>12.2f} {rmse_bot4.mean():>12.2f}")

print(f"\nResults saved to: {os.path.abspath(OUTPUT_DIR)}")