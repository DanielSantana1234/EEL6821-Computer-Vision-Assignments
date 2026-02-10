from PIL import Image

image_path1 = "Images/Assignment 1 Images/Street.bmp"
image_path2 = "Images/Assignment 1 Images/Black Grid.bmp"

img1 = Image.open(image_path1)
img2 = Image.open(image_path2)

width1, height1 = img1.size
width2, height2 = img2.size

img1.paste(img2, (int(width1/2) - 300, int(height1/2) + 700), mask = img2)

img1.show()