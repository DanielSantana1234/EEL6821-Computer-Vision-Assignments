from PIL import Image
import numpy as np

def projection_result(img1: Image, img2: Image) -> Image:
    """
        A function to calculate the 3x3 homography matrix.
            Args:
                img1: The first image to be transformed.
                img2: The second image to be used as a reference for the transformation.
    """
    
    pass

def warp_perspective(img1: Image, img2: Image) -> Image:
    """
        A function to perform the perspective warp on a given 2d image.
            Args:
                img: The image to be transformed.
                src_points: The source points for the perspective transform.
                dst_points: The destination points for the perspective transform.
    """
    pass

image_path1 = "Images/Assignment 1 Images/Street.bmp"
image_path2 = "Images/Assignment 1 Images/Black Grid.bmp"

img1 = Image.open(image_path1)
gray_img1 = img1.convert("L")
img2 = Image.open(image_path2)

width1, height1 = gray_img1.size
width2, height2 = img2.size

res2 = img2.resize((1600, 1600))



gray_img1.paste(res2, (int(width1/2) - 750, int(height1/2) + 100), mask = res2)

gray_img1.show()