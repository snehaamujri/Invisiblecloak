import cv2
import numpy as np
import time

# Open the webcam
cap = cv2.VideoCapture(0)
time.sleep(2)

# Capture the background frame (static)
ret, background = cap.read()
background = cv2.flip(background, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue color (adjust if necessary)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Create a mask to detect blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Refine the mask with morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Inverse mask for areas without blue
    mask_inv = cv2.bitwise_not(mask)

    # Isolate the background where the blue color is detected
    cloak_area = cv2.bitwise_and(background, background, mask=mask)

    # Show the current frame without the blue regions
    visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine the two areas
    result = cv2.add(cloak_area, visible_area)

    # Display the result
    cv2.imshow("Invisibility Cloak", result)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
