"""
II. The Visual Machinery of the human Brain
Note: For all the problems given in sections II through VII, use both synthetic images and natural images 
that are interesting for the problems addressed
Just as you learned about the different edge operators, create your own operators to mimic the behavior 
(operational structure) of simple cells
Indicate what angular resolution you have achieved (10º is ideal but others are also acceptable) and what 
threshold you have used to decide whether a pixel is an edge point or not
Combine (superimpose) all the edge extraction results you have obtained in (1)
Contrast these results to the directional Kirsch operator
Contrast these results to another edge operator of your choice with the 8 gradient directions and again 
superimpose the results. An example will be the Prewitt compass already given to you.
Make use of multiple operators (mimicking simple cells) but with different degrees of fuzziness 
on blurred images or on images that have things like clouds, waves in an ocean, and express your 
thoughts on the results.
To do 
1. You have to make custom edge operators to convolve over images
2. You have to use both natural & synthetic images 
3. You then get gradient magnitudes which will determine if it is an edge or not
4. You then have a set of binary maps then you stack them together 
5. Then use this same process for other edge operators 
"""
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.signal import convolve2d

def load_image(path):
    try:
        img = Image.open(path).convert("L")
    except FileNotFoundError:
        print("Image not found — using synthetic image.")
        img = Image.fromarray(np.uint8(np.random.rand(200, 200) * 255))
    return np.array(img, dtype=np.float32)

BASE_KERNEL = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=float)

def make_simple_cells(step=10):
    return [rotate(BASE_KERNEL, a, reshape=False, mode='nearest') for a in range(0, 180, step)]

PREWITT = [
    np.array([[ 1, 1, 1],[ 0, 0, 0],[-1,-1,-1]]),  # N
    np.array([[ 0, 1, 1],[-1, 0, 1],[-1,-1, 0]]),  # NE
    np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]),  # E
    np.array([[-1,-1, 0],[-1, 0, 1],[ 0, 1, 1]]),  # SE
    np.array([[-1,-1,-1],[ 0, 0, 0],[ 1, 1, 1]]),  # S
    np.array([[ 0,-1,-1],[ 1, 0,-1],[ 1, 1, 0]]),  # SW
    np.array([[ 1, 0,-1],[ 1, 0,-1],[ 1, 0,-1]]),  # W
    np.array([[ 1, 1, 0],[ 1, 0,-1],[ 0,-1,-1]]),  # NW
]

KIRSCH = [
    np.array([[ 5, 5, 5],[-3, 0,-3],[-3,-3,-3]]),  # N
    np.array([[ 5, 5,-3],[ 5, 0,-3],[-3,-3,-3]]),  # NW
    np.array([[ 5,-3,-3],[ 5, 0,-3],[ 5,-3,-3]]),  # W
    np.array([[-3,-3,-3],[ 5, 0,-3],[ 5, 5,-3]]),  # SW
    np.array([[-3,-3,-3],[-3, 0,-3],[ 5, 5, 5]]),  # S
    np.array([[-3,-3,-3],[-3, 0, 5],[-3, 5, 5]]),  # SE
    np.array([[-3,-3, 5],[-3, 0, 5],[-3,-3, 5]]),  # E
    np.array([[-3, 5, 5],[-3, 0, 5],[-3,-3,-3]]),  # NE
]

def superimpose_edges(image, operators, threshold):
    """Convolve all operators, threshold each result, then take pixel-wise max."""
    result = np.zeros_like(image)
    for op in operators:
        magnitude = np.abs(convolve2d(image, op, mode='same', boundary='symm'))
        result = np.maximum(result, np.where(magnitude > threshold, 255, 0))
    return result

def show(images, titles, suptitle=""):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray'); ax.set_title(title); ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout(); plt.show()

THRESHOLD = 50
ANGULAR_RES = 10  # degrees

img = load_image("Images/Assignment 2 Images/lena_gray.bmp")
simple_cells = make_simple_cells(ANGULAR_RES)

print(f"Angular resolution: {ANGULAR_RES}°  |  Threshold: {THRESHOLD}")

custom_result  = superimpose_edges(img, simple_cells, THRESHOLD)
prewitt_result = superimpose_edges(img, PREWITT,      THRESHOLD)
kirsch_result  = superimpose_edges(img, KIRSCH,       THRESHOLD)

show(
    [img, custom_result, prewitt_result, kirsch_result],
    ['Original', f'Simple Cells ({ANGULAR_RES}° res)', 'Prewitt Compass', 'Kirsch Compass'],
    suptitle=f"Threshold = {THRESHOLD}"
)

# ── Part 4: Fuzziness study on blurred images ──────────────────────────────────
pil_img = Image.fromarray(img.astype(np.uint8))
blur_levels = [2, 5, 10]  # Gaussian blur radii

blurred_images  = [np.array(pil_img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) for r in blur_levels]
blurred_results = [superimpose_edges(b, simple_cells, THRESHOLD) for b in blurred_images]

show(
    [img] + blurred_results,
    ['Original'] + [f'Blur radius={r}' for r in blur_levels],
    suptitle="Simple Cells on Blurred Images — Fuzziness Study"
)

blurry = blurred_images[-1]
threshold_study = [superimpose_edges(blurry, simple_cells, t) for t in [20, 50, 100]]

show(
    threshold_study,
    [f'Blurred + Threshold={t}' for t in [20, 50, 100]],
    suptitle="Effect of Threshold on Blurred Image"
)