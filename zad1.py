import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def apply_filter(image, filter_func, strength=1.0):
    f_image = fft2(image)
    f_shifted = fftshift(f_image)

    mask = filter_func(f_shifted.shape, strength)

    f_filtered = f_shifted * mask

    f_ishifted = ifftshift(f_filtered)
    img_back = ifft2(f_ishifted)
    img_filtered = np.abs(img_back)

    return img_filtered


def low_pass_filter(shape, strength):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    d = min(crow, ccol) / strength
    center_square = np.array((crow - d, crow + d, ccol - d, ccol + d), dtype=int)
    mask[center_square[0]:center_square[1], center_square[2]:center_square[3]] = 1
    return mask


def high_pass_filter(shape, strength):
    return 1 - low_pass_filter(shape, strength)


def sharpening_filter(shape, strength):
    lp = low_pass_filter(shape, strength)
    hp = 1 - lp
    return 2 + hp


image = color.rgb2gray(cv2.imread("soldier.png"))

low_pass_strength = 30
high_pass_strength = 30
sharpening_strength = 30

img_low_passed = apply_filter(image, low_pass_filter, low_pass_strength)

img_high_passed = apply_filter(image, high_pass_filter, high_pass_strength)

img_sharpened = apply_filter(image, sharpening_filter, sharpening_strength)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].title.set_text('Original Image')
axs[0, 0].axis('off')
axs[0, 1].imshow(img_low_passed, cmap='gray')
axs[0, 1].title.set_text('Low Pass Filtered Image')
axs[0, 1].axis('off')
axs[1, 0].imshow(img_high_passed, cmap='gray')
axs[1, 0].title.set_text('High Pass Filtered Image')
axs[1, 0].axis('off')
axs[1, 1].imshow(img_sharpened, cmap='gray')
axs[1, 1].title.set_text('Sharpened Image')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()