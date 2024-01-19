import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def apply_descreen(image, max_filter_size=3, min_filter_size=3):
    # Transform the image to frequency domain
    f_shifted = fftshift(fft2(image))

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_shifted)

    # Find local maxima using a maximum filter
    local_maxima = maximum_filter(magnitude_spectrum, size=max_filter_size)

    # Reduce the maxima by using a minimum filter
    reduced_maxima = minimum_filter(local_maxima, size=min_filter_size)

    # Find the maxima locations
    maxima_locations = magnitude_spectrum == reduced_maxima

    # Suppress the maxima in the frequency domain by setting them to the mean value
    mean_val = np.mean(magnitude_spectrum)
    suppressed_spectrum = np.where(maxima_locations, mean_val, magnitude_spectrum)

    # Replace the original magnitude spectrum with the suppressed one
    f_shifted_suppressed = f_shifted / magnitude_spectrum * suppressed_spectrum

    # Shift back and inverse FFT to get back to image domain
    img_descreened = np.abs(ifft2(ifftshift(f_shifted_suppressed)))

    return img_descreened, magnitude_spectrum, suppressed_spectrum


# Load a sample image from skimage.data and convert it to grayscale
image_gray = rgb2gray(cv2.imread("soldier.png"))

# Apply descreen algorithm to the image
img_descreened, magnitude_spectrum, suppressed_spectrum = apply_descreen(image_gray)

# Display the images
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Magnitude spectrum of the original image
axs[1].imshow(np.log(1 + magnitude_spectrum), cmap='gray')
axs[1].set_title('Magnitude Spectrum (Before)')
axs[1].axis('off')

# Descreened image
axs[2].imshow(img_descreened, cmap='gray')
axs[2].set_title('Descreened Image')
axs[2].axis('off')

plt.show()