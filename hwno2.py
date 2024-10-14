import os
import cv2
import numpy as np

# Define the path to your image
img_path = os.path.join('pics/messi.jpg')

# Read the image
img = cv2.imread(img_path)

# Create a copy of the original image to draw the rectangle and text on
img_overlay = img.copy()

# Convert the image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply simple threshold with 40% (which is 102 out of 255)
threshold_value = 0.40 * 255  # 40% of 255
ret, simple_thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

# Apply Canny edge detection with threshold values (150, 150)
img_edge = cv2.Canny(img_gray, 150, 150)
img_edge_d = cv2.dilate(img_edge, np.ones((3, 3), dtype=np.int8))  # Dilation

# Define the rectangle parameters with your specified properties
cv2.rectangle(img_overlay, (500, 50), (900, 325), (0, 255, 0), 2)  # Green rectangle with thickness 2

# Define the text parameters with your specified properties
cv2.putText(img_overlay, 'Lionel Messi', (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text with thickness 2

# Show the final image with the rectangle and text
cv2.imshow('Drawn Overlay', img_overlay)

# Show the original, thresholded, and edge-detected images for comparison
cv2.imshow('Original', img)
cv2.imshow('Simple Threshold', simple_thresh)
cv2.imshow('Edge Detection', img_edge)
cv2.imshow('Dilated Edge Detection', img_edge_d)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
