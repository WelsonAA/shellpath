"""from station import Station

if "__main__"==__name__:
    # Read coordinates from text file
    with open('coordinates.txt', 'r') as f:
        lines = f.readlines()

    stations = []
    for line in lines:
        line = line.strip().replace("[", "").replace("]", "").replace(" ", "")
        coords=[]

        coords = (line.split(","))
        stati = Station(coords[0],coords[1],coords[2])
        stations.append(stati)

    print(stations)"""

import cv2
import numpy as np

# Load image
img = cv2.imread('comp.png')

# Convert image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for red color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Create a mask for red color in HSV
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                           param1=100, param2=9, minRadius=3, maxRadius=8)

# Check if circles were detected
if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # Get the center of the image
    center = (float(img.shape[1]) / 2,float( img.shape[0]) / 2)

    # Loop over the detected circles
    for (x, y, r) in circles:
        # Print the coordinates relative to the center of the image
        print(f"Circle center coordinates: ({x - center[0]}, {y - center[1]})")

        # Draw the circle on the original image
        cv2.circle(img, (x, y), r, (0, 0, 0), thickness=-1)

# Display the modified image
cv2.imshow("Red Circles to Black", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
