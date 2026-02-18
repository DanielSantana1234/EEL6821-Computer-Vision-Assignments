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

#gray_img1.paste(image_2, (int(width1/2) - 750, int(height1/2) + 100), mask = image_2)

def projection_result(img1: Image, img2: Image) -> Image:
    """
        A function to calculate the 3x3 homography matrix.
            Args:
                img1: The first image to be transformed.
                img2: The second image to be used as a reference for the transformation.
    """
    
    pass

def warp_perspective(img1: Image, img2: Image) -> None:
    """
        A function to perform the perspective warp on a given 2d image.
            Args:
                img: The image to be transformed.
                src_points: The source points for the perspective transform.
                dst_points: The destination points for the perspective transform.
    """

    projection_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -1/500, 1]
    ])
    # warped_image = Image.new("L", (width1, height1))
    for x in range(width2):
        for y in range(height2):
            curr_coord = img2.getpixel((x, y))
            if curr_coord == 0:
                continue

            z = (y - height2 / 2.0) * 4.0
            
            pixel_coord = np.array([x, y, z, 1])
            result = np.dot(projection_matrix, pixel_coord)

            # Normalize the result
            result[0] /= result[3]
            result[1] /= result[3]
            result[2] /= result[3]

            # Map the result back to the image coordinates
            x_new = int(result[0])
            y_new = int(result[1])
            z_new = int(result[2])

            # Set the pixel value in the warped image
            if 0 <= x_new < width1 and 0 <= y_new < height1:
                img1.putpixel((x_new, y_new), img2.getpixel((x, y)))
                
    img1.show()

    return gray_img1

# warp_perspective(gray_img1, img2)
# gray_img1.show()
warp_perspective(gray_img1, image_2)
