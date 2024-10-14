import os
import cv2
import numpy as np

img_path = os.path.join('pics/messi.jpg')

img = cv2.imread(img_path)
img_overlay = img.copy()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold_value = 0.40 * 255 
ret, simple_thresh = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

img_edge = cv2.Canny(img_gray, 150, 150)
img_edge_d = cv2.dilate(img_edge, np.ones((3, 3), dtype=np.int8))

cv2.rectangle(img_overlay, (500, 50), (900, 325), (0, 255, 0), 2)
cv2.putText(img_overlay, 'Lionel Messi', (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Original Image, window labelled as "Original"
cv2.imshow('Original', img)
# Simple Threshold Image with a threshold value of 40%, max value of 255, window labelled as "Simple Threshold"
cv2.imshow('Simple Threshold', simple_thresh)
# Edge Detected Image using Canny Function with threshold set to (150,150), window labelled as "Edge Detection"
cv2.imshow('Edge Detection', img_edge)
# The previous Edge Detected Image with Dilate function set with np.ones((3, 3), dtype=np.int8), window labelled as "Dilated Edge Detection"
cv2.imshow('Dilated Edge Detection', img_edge_d)
# The Original Image with a drawn rectangle on a focused object and a one-word text describing the image used, colors of the rectangle 
# and text must be in contrast with the background image, window labelled as "Drawn Overlay"
cv2.imshow('Drawn Overlay', img_overlay)

cv2.waitKey(0)