import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.io import imread

# Load the image (replace with the path to your image file)

image_float = rgb2gray(cv2.imread("soldier.png"))

# Define a kernel for high-pass filtering
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply the high-pass filter
high_pass_img = convolve2d(image_float, kernel, mode='same', boundary='wrap')

# Normalize the high-pass filtered image
high_pass_img = (high_pass_img - high_pass_img.min()) / (high_pass_img.max() - high_pass_img.min())

# Apply a low-pass filter (Gaussian blur) to the high-pass image
low_pass_img = gaussian(high_pass_img, sigma=2)

# Extract the texture pattern
texture_pattern = high_pass_img - low_pass_img

# Normalize the texture pattern
texture_pattern = (texture_pattern - texture_pattern.min()) / (texture_pattern.max() - texture_pattern.min())

# Subtract the texture pattern from the original image to descreen
descreened_image = image_float - texture_pattern

# Normalize the descreened image
descreened_image = (descreened_image - descreened_image.min()) / (descreened_image.max() - descreened_image.min())

# Save the descreened image (optional)
descreened_image_path = 'path_to_save_descreened_image.png'  # Replace with the desired save location
plt.imsave(descreened_image_path, descreened_image, cmap='gray')

# Display the results
plt.imshow(descreened_image, cmap='gray')
plt.title('Descreened Image')
plt.axis('off')
plt.show()