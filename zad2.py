import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.filters import gaussian

image_float = rgb2gray(cv2.imread("soldier.png"))

kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

high_pass_img = convolve2d(image_float, kernel, mode='same', boundary='wrap')

high_pass_img = (high_pass_img - high_pass_img.min()) / (high_pass_img.max() - high_pass_img.min())

low_pass_img = gaussian(high_pass_img, sigma=2)

texture_pattern = high_pass_img - low_pass_img

texture_pattern = (texture_pattern - texture_pattern.min()) / (texture_pattern.max() - texture_pattern.min())

descreened_image = image_float - texture_pattern

descreened_image = (descreened_image - descreened_image.min()) / (descreened_image.max() - descreened_image.min())

plt.imshow(descreened_image, cmap='gray')
plt.title('Descreened Image')
plt.axis('off')
plt.show()