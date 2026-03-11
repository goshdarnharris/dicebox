import cv2
import numpy as np
import random
import math

# Image size
img_size = 50

# Create a black image
img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Define square side length and center
side = 30
center = (img_size // 2, img_size // 2)

# Generate a random angle in degrees
angle = random.uniform(0, 360)

# Define square corners (centered at origin)
half_side = side / 2
pts = np.array([
    [-half_side, -half_side],
    [ half_side, -half_side],
    [ half_side,  half_side],
    [-half_side,  half_side]
])

# Convert angle to radians
theta = math.radians(angle)

# Rotation matrix
rotation_matrix = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta),  math.cos(theta)]
])

# Rotate and translate points
rotated_pts = np.dot(pts, rotation_matrix) + center
rotated_pts = rotated_pts.astype(np.int32)

# Draw the rotated square
cv2.polylines(img, [rotated_pts], isClosed=True, color=(255, 255, 255), thickness=1)

# Show and save image
cv2.imshow('Rotated Square', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
