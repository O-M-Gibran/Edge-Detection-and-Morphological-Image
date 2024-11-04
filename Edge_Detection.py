import numpy as np
import cv2
import matplotlib.pyplot as plt

# Baca gambar dalam mode grayscale
image = cv2.imread('edge_detection_1.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Gaussian Blur untuk mereduksi noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Terapkan deteksi tepi Canny
edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

# Tampilkan hasil
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection Result")
plt.show()
