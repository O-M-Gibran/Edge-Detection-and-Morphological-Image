import cv2
import numpy as np

# Load the image with alpha channel (transparency)
image = cv2.imread('Hitoshura.png', cv2.IMREAD_UNCHANGED)

# Check if the image has an alpha channel
if image.shape[2] == 4:  # RGBA image
    # Create a mask for non-transparent areas
    alpha_channel = image[:, :, 3]
    mask = alpha_channel > 0
    
    # Convert the image to BGR for processing, ignoring the alpha channel
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
else:
    mask = np.ones(image.shape[:2], dtype=bool)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for segmentation based on the colors in the image
# Adjust these values if needed to fit the specific shades in your image

# Jacket (Gray with Green Hood)
lower_gray = np.array([0, 0, 40])
upper_gray = np.array([180, 30, 200])
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Pants (Black)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Skin (we'll use a generic skin tone range here, may need adjustment)
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])

# Create masks for each color
mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
mask_skin = cv2.inRange(hsv_image, lower_skin, upper_skin)

# Combine jacket colors (gray and green hood)
mask_jacket = cv2.bitwise_or(mask_gray, mask_green)

# Apply mask to get the main object without background
masked_output = np.zeros_like(image)
masked_output[mask_jacket > 0] = [128, 128, 128]  # Gray for jacket
masked_output[mask_black > 0] = [0, 0, 0]         # Black for pants
masked_output[mask_skin > 0] = [0, 255, 0]        # Green for skin

# Apply transparency mask to keep only the main character area
masked_output[~mask] = [0, 0, 0]  # Set background to black

# Display the original and the result
cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image", masked_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
