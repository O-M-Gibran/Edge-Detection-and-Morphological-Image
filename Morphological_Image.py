import cv2
import numpy as np


image = cv2.imread('Hitoshura.png', cv2.IMREAD_UNCHANGED)


if image.shape[2] == 4: 
    alpha_channel = image[:, :, 3]
    mask = alpha_channel > 0
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
else:
    mask = np.ones(image.shape[:2], dtype=bool)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_gray = np.array([0, 0, 40])
upper_gray = np.array([180, 30, 200])
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])


mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
mask_skin = cv2.inRange(hsv_image, lower_skin, upper_skin)

mask_jacket = cv2.bitwise_or(mask_gray, mask_green)


masked_output = np.zeros_like(image)
masked_output[mask_jacket > 0] = [128, 128, 128]  # Gray for jacket
masked_output[mask_black > 0] = [0, 0, 0]         # Black for pants
masked_output[mask_skin > 0] = [0, 255, 0]        # Green for skin
masked_output[mask_green > 0] = [0, 0, 255]      # Change green hoodie to red


masked_output[~mask] = [0, 0, 0]  

cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image", masked_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
