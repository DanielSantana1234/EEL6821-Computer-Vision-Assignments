"""
VIII. Image Restoration
Apply at least two different transfer functions like diffraction, CCD interaction, 
atmospheric turbulence, rectangular aperture, or horizontal motion and see what happens to the original 
image as you change the parameters of the different transfer functions you used. This will teach you 
the effect such transfer functions have on images.
Those of you who took image processing, use your own Fast Fourier Transform to show 
that you could use band reject filters to resolve images that have been corrupted by sinusoidal noise. 
Others use MATLAB for FFT.
"""
"""
VIII. Image Restoration
- Applies transfer functions: Atmospheric Turbulence & Horizontal Motion Blur
- Demonstrates parameter effects on images
- Uses custom FFT-based Band Reject Filter to remove sinusoidal noise
"""

"""
VIII. Image Restoration
- Two transfer functions: Atmospheric Turbulence & Horizontal Motion Blur
- Custom FFT-based Band-Reject Filter to remove sinusoidal noise
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread

IMAGE_PATH = r"C:\Users\danie\EEL6821-Computer-Vision-Assignments\Images\Assignment 1 Images\Street.jpg"

img = imread(IMAGE_PATH)
if img.ndim == 3:
    img = rgb2gray(img)
img = img.astype(np.float64)
img /= img.max()
M, N = img.shape

def apply_transfer(img, H):
    F = np.fft.fftshift(np.fft.fft2(img))
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return np.clip(result / result.max(), 0, 1)

def atm_turbulence(shape, k):
    M, N = shape
    u = np.fft.fftshift(np.fft.fftfreq(M))
    v = np.fft.fftshift(np.fft.fftfreq(N))
    V, U = np.meshgrid(v, u)
    return np.exp(-k * (U**2 + V**2) ** (5 / 6))

k_values = [0.001, 0.01, 0.05]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
for ax, k in zip(axes[1:], k_values):
    ax.imshow(apply_transfer(img, atm_turbulence(img.shape, k)), cmap="gray")
    ax.set_title(f"Atm. Turbulence\nk = {k}"); ax.axis("off")
fig.suptitle("Transfer Function 1 — Atmospheric Turbulence", fontweight="bold")
plt.tight_layout(); plt.show()

def motion_blur(shape, T, a):
    M, N = shape
    u = np.fft.fftshift(np.fft.fftfreq(M))
    v = np.fft.fftshift(np.fft.fftfreq(N))
    V, U = np.meshgrid(v, u)
    phi = np.pi * U * a
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(phi == 0, T, (T / phi) * np.sin(phi) * np.exp(-1j * phi))
    return H

a_values = [0.02, 0.1, 0.3]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
for ax, a in zip(axes[1:], a_values):
    ax.imshow(apply_transfer(img, motion_blur(img.shape, T=0.1, a=a)), cmap="gray")
    ax.set_title(f"Motion Blur\nT=0.1, a={a}"); ax.axis("off")
fig.suptitle("Transfer Function 2 — Horizontal Motion Blur", fontweight="bold")
plt.tight_layout(); plt.show()

u0, v0 = 50, 50
noise = 0.3 * np.sin(2 * np.pi * (u0 * np.arange(M)[:, None] / M + v0 * np.arange(N)[None, :] / N))
corrupted = np.clip(img + noise, 0, 1)

cu, cv = M // 2, N // 2
i_idx = np.arange(M)[:, None]
j_idx = np.arange(N)[None, :]
H_reject = np.ones((M, N))
for du, dv in [(u0, v0), (-u0, -v0), (u0, -v0), (-u0, v0)]:
    H_reject[(i_idx - cu - du)**2 + (j_idx - cv - dv)**2 <= 8**2] = 0

F_corrupted = np.fft.fftshift(np.fft.fft2(corrupted))
restored = np.abs(np.fft.ifft2(np.fft.ifftshift(F_corrupted * H_reject)))
restored = np.clip(restored / restored.max(), 0, 1)

spec = lambda x: np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(x))))

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Band-Reject Filter | Sinusoidal Noise Removal via Custom FFT", fontweight="bold")
for ax, im, title in zip(axes[0], [img, corrupted, restored], ["Original", "Corrupted (noise added)", "Restored (band-reject)"]):
    ax.imshow(im, cmap="gray"); ax.set_title(title); ax.axis("off")
for ax, im, title in zip(axes[1], [spec(img), spec(corrupted), spec(restored)],
["Spectrum — Original", "Spectrum — Corrupted\n(spikes visible)", "Spectrum — Restored\n(spikes removed)"]):
    ax.imshow(im, cmap="gray"); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()

mse = np.mean((img - restored) ** 2)
print(f"PSNR after restoration: {10 * np.log10(1.0 / mse):.2f} dB")