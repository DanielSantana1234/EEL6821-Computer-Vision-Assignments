"""
II. The Visual Machinery of the human Brain
Note: For all the problems given in sections II through VII, use both synthetic images and natural images that are interesting for the problems addressed
Just as you learned about the different edge operators, create your own operators to mimic the behavior (operational structure) of simple cells
Indicate what angular resolution you have achieved (10º is ideal but others are also acceptable) and what threshold you have used to decide whether a pixel is an edge point or not
Combine (superimpose) all the edge extraction results you have obtained in (1)
Contrast these results to the directional Kirsch operator
Contrast these results to another edge operator of your choice with the 8 gradient directions and again superimpose the results. An example will be the Prewitt compass already given to you.
Make use of multiple operators (mimicking simple cells) but with different degrees of fuzziness on blurred images or on images that have things like clouds, waves in an ocean, and express your thoughts on the results.
To do 
1. You have to make custom edge operators to convolve over images
2. You have to se both natural & synthetic images 
3. You then get gradient magnitudes which will determine if it is an edge or not
4. You then have a set of binary maps then you stack them together 
5. Then use this same process for other edge operators 
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.signal import convolve2d

image_path1 = "Images/Assignment 2 Images/lena_gray.bmp"

try:
    img1 = Image.open(image_path1)
except FileNotFoundError:
    print("Image not found. Creating a synthetic image for testing...")
    img1 = Image.fromarray(np.uint8(np.random.rand(200, 200) * 255))

gray_img1 = img1.convert("L")
width1, height1 = gray_img1.size

# Convert the PIL Image to a NumPy array for math operations
img_array = np.array(gray_img1, dtype=np.float32)

base_operator = np.array(
    [
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ]
)

# ==========================================
# 1. Make custom edge operators (Simple Cells)
# ==========================================
# We will rotate the base operator by 10-degree increments to achieve 10º resolution.
custom_operators = []
for angle in range(0, 180, 10):
    # reshape=False keeps it 3x3. mode='nearest' prevents weird interpolation artifacts
    rotated_op = rotate(base_operator, angle, reshape=False, mode='nearest')
    custom_operators.append(rotated_op)

# ==========================================
# 3 & 4. Convolve, get gradients, threshold (binary), and stack (superimpose)
# ==========================================
def apply_and_superimpose(image_matrix, operators, threshold_value):
    # Create an empty canvas of zeros with the same shape as the image
    superimposed_result = np.zeros_like(image_matrix)
    
    for op in operators:
        # Step 3: Convolve the operator over the image
        # mode='same' ensures the output matrix is the exact same size as the input
        gradient = convolve2d(image_matrix, op, mode='same', boundary='symm')
        
        # Get the magnitude (absolute value) since we care about edges, not direction
        magnitude = np.abs(gradient)
        
        # Step 4: Create a binary map using the threshold
        # If the magnitude is greater than the threshold, set to 255 (white edge), else 0
        binary_map = np.where(magnitude > threshold_value, 255, 0)
        
        # Stack/Superimpose by taking the maximum value at each pixel location
        superimposed_result = np.maximum(superimposed_result, binary_map)
        
    return superimposed_result

threshold = 50
custom_result = apply_and_superimpose(img_array, custom_operators, threshold)

prewitt_compass = [
    np.array([[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]]), # N
    np.array([[ 0,  1,  1], [-1,  0,  1], [-1, -1,  0]]), # NE
    np.array([[-1,  0,  1], [-1,  0,  1], [-1,  0,  1]]), # E
    np.array([[-1, -1,  0], [-1,  0,  1], [ 0,  1,  1]]), # SE
    np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]]), # S
    np.array([[ 0, -1, -1], [ 1,  0, -1], [ 1,  1,  0]]), # SW
    np.array([[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]]), # W
    np.array([[ 1,  1,  0], [ 1,  0, -1], [ 0, -1, -1]])  # NW
]

kirsch_compass = [
    np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]]), # N
    np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]]), # NW
    np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]]), # W
    np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]]), # SW
    np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]]), # S
    np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]]), # SE
    np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]]), # E
    np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]])  # NE
]

prewitt_result = apply_and_superimpose(img_array, prewitt_compass, threshold)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(prewitt_result, cmap='gray')
axes[1].set_title('Prewitt Compass (8 Directions)')
axes[1].axis('off')

axes[2].imshow(custom_result, cmap='gray')
axes[2].set_title('Custom Simple Cells (10º Resolution)')
axes[2].axis('off')

plt.tight_layout()
plt.show()