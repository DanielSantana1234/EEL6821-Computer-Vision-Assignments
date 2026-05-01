"""
VII. Log-Polar Plots
Using images that have been rotated or translated, determine their log-polar plot
Determine what happens to the log-polar plot under these transformations
Could you quantify (or find a way to gauge) what happened in the log-polar plots to express 
the degree of rotation and the extent of translation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from skimage.transform import warp_polar
from skimage.io import imread
from skimage.color import rgb2gray

IMAGE_PATH = r"C:\Users\danie\EEL6821-Computer-Vision-Assignments\Images\Assignment 1 Images\Street.jpg"

img = imread(IMAGE_PATH)
if img.ndim == 3:
    img = rgb2gray(img)
img = img.astype(np.float64)
img /= img.max()

ROTATION_ANGLE = 30.0
SHIFT_DY, SHIFT_DX = 20, 15

rotated    = rotate(img,  angle=ROTATION_ANGLE, reshape=False)
translated = shift(img,   shift=(SHIFT_DY, SHIFT_DX))

def log_polar(image):
    return warp_polar(image, scaling='log')

lp_orig  = log_polar(img)
lp_rot   = log_polar(rotated)
lp_trans = log_polar(translated)

def estimate_rotation(lp1, lp2):
    F1, F2 = np.fft.fft2(lp1), np.fft.fft2(lp2)
    corr = np.fft.ifft2((F1 * np.conj(F2)) / (np.abs(F1 * np.conj(F2)) + 1e-10)).real
    row_shift = np.argmax(corr.max(axis=1))
    if row_shift > lp1.shape[0] // 2:
        row_shift -= lp1.shape[0]
    return row_shift * (360.0 / lp1.shape[0])

def estimate_translation(lp1, lp2):
    p1, p2 = lp1.mean(axis=0), lp2.mean(axis=0)
    corr = np.fft.ifft(np.fft.fft(p1) * np.conj(np.fft.fft(p2))).real
    s = np.argmax(corr)
    if s > len(p1) // 2:
        s -= len(p1)
    return s

est_rot   = estimate_rotation(lp_orig, lp_rot)
est_trans = estimate_translation(lp_orig, lp_trans)

print(f"Rotation    — applied: {ROTATION_ANGLE}°       | estimated: {abs(est_rot):.1f}°")
print(f"Translation — applied: dy={SHIFT_DY} dx={SHIFT_DX} | radial shift: {est_trans} bins")

fig, axes = plt.subplots(2, 3, figsize=(13, 12))
fig.suptitle("Log-Polar Analysis: Rotation & Translation", fontsize=14, fontweight='bold')

labels  = ["Original", f"Rotated {ROTATION_ANGLE}°", f"Translated dy={SHIFT_DY} dx={SHIFT_DX}"]
images  = [img, rotated, translated]
lp_imgs = [lp_orig, lp_rot, lp_trans]
colors  = ['steelblue', 'tomato', 'seagreen']

for ax, image, title in zip(axes[0], images, labels):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for ax, lp, title in zip(axes[1], lp_imgs, labels):
    ax.imshow(lp, cmap='inferno', aspect='auto', origin='lower', extent=[0, 1, 0, 360])
    ax.set_title(f"LP: {title}")
    ax.set_xlabel("log(radius)")
    ax.set_ylabel("Angle (°)")

plt.tight_layout()
plt.savefig("log_polar_analysis.png", dpi=150, bbox_inches='tight')
plt.show()