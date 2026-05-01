from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def make_ladder_mask(width: int = 220, height: int = 220, spacing: int = 24) -> np.ndarray:
    ladder = np.zeros((height, width), dtype=np.uint8)

    left_rail = width // 3
    right_rail = width - left_rail

    ladder[12:height - 12, left_rail - 2:left_rail + 2] = 255
    ladder[12:height - 12, right_rail - 2:right_rail + 2] = 255

    for y in range(16, height - 16, spacing):
        ladder[y - 1:y + 2, left_rail:right_rail] = 255

    return ladder

def warp_perspective(img1: Image.Image, img2: np.ndarray) -> Image.Image:
    width1, height1 = img1.size
    height2, width2 = img2.shape

    projection_matrix = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 1.0,  0.0, 0.0],
        [0.0, 0.0,  1.0, 0.0],
        [0.0, 0.0, -1.0/500.0, 1.0]
    ], dtype=float)

    Sx = (width1 * 0.15) / (width2 / 2)
    Sy = Sx * 1.2
    scaling_matrix = np.array([
        [Sx, 0.0, 0.0, 0.0],
        [0.0, Sy, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    k1 = width1 / 2.0
    k2 = height1 * 0.70
    translation_matrix = np.array([
        [1.0, 0.0, 0.0, k1],
        [0.0, 1.0, 0.0, k2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    combined = translation_matrix @ scaling_matrix @ projection_matrix

    output = img1.copy()

    for x in range(width2):
        for y in range(height2):
            if img2[y, x] == 0:
                continue

            xc = x - (width2 / 2.0)
            yc = y - (height2 / 2.0)
            z  = yc * 4.0

            pixel_coord = np.array([xc, yc, z, 1.0], dtype=float)
            result = combined @ pixel_coord

            w = result[3]
            if abs(w) < 1e-8:
                continue

            x_new = int(round(result[0] / w))
            y_new = int(round(result[1] / w))

            if 0 <= x_new < width1 and 0 <= y_new < height1:
                output.putpixel((x_new, y_new), int(img2[y, x]))

    return output

image_path1 = "Images/Assignment 1 Images/Street.bmp"

image_2 = make_ladder_mask()

img1 = Image.open(image_path1)
gray_img1 = img1.convert("L")

warped_img = warp_perspective(gray_img1, image_2)
warped_img.show()