from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path1 = "Images/Assignment 1 Images/Street.bmp"
image_2 = np.zeros((200, 200))

spacing = 20

image_2[::spacing, :] = 1
image_2[:, ::spacing] = 1

img1 = Image.open(image_path1)
gray_img1 = img1.convert("L")

image_2 = Image.fromarray((image_2 * 255).astype(np.uint8))

width1, height1 = gray_img1.size

width2, height2 = image_2.size

def warp_perspective(img1: Image, img2: Image) -> Image:
    """
    Perspective warp using homogeneous coordinates.

    The source image (img2) is projected with a simple perspective matrix,
    then translated into img1's coordinate space.
    """
    width1, height1 = img1.size
    width2, height2 = img2.size

    projection_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0 / 500.0, 1.0]
    ], dtype=float)

    # Center the warped grid on top of img1 (instead of shifting far off-screen).
    k1 = width1 / 2.0
    k2 = height1 * 0.65
    translation_matrix = np.array([
        [1.0, 0.0, 0.0, k1],
        [0.0, 1.0, 0.0, k2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

    output = img1.copy()

    for x in range(width2):
        for y in range(height2):
            value = img2.getpixel((x, y))
            if value == 0:
                continue

            # Use centered local coordinates before projection.
            xc = x - (width2 / 2.0)
            yc = y - (height2 / 2.0)
            z = yc * 4.0

            pixel_coord = np.array([xc, yc, z, 1.0], dtype=float)
            result = projection_matrix @ pixel_coord
            result_translated = translation_matrix @ result

            w = result_translated[3]
            if abs(w) < 1e-8:
                continue

            result_translated[0] /= w
            result_translated[1] /= w

            x_new = int(round(result_translated[0]))
            y_new = int(round(result_translated[1]))

            if 0 <= x_new < width1 and 0 <= y_new < height1:
                output.putpixel((x_new, y_new), value)

    output.show()
    return output

warped_img = warp_perspective(gray_img1, image_2)