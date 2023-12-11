import cv2
import numpy as np


def create_mask(frame, min_values, max_values):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_values, max_values)
    return mask


def update_values(trackbar_value):
    pass


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Create a named window for trackbars
cv2.namedWindow('Trackbars')

# Initialize trackbar values
min_HSV = np.array([0, 0, 0])
max_HSV = np.array([179, 255, 255])

# Create trackbars for adjusting HSV thresholds
cv2.createTrackbar('Hue_min', 'Trackbars', min_HSV[0], 179, update_values)
cv2.createTrackbar('Sat_min', 'Trackbars', min_HSV[1], 255, update_values)
cv2.createTrackbar('Val_min', 'Trackbars', min_HSV[2], 255, update_values)
cv2.createTrackbar('Hue_max', 'Trackbars', max_HSV[0], 179, update_values)
cv2.createTrackbar('Sat_max', 'Trackbars', max_HSV[1], 255, update_values)
cv2.createTrackbar('Val_max', 'Trackbars', max_HSV[2], 255, update_values)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current trackbar positions
    min_HSV[0] = cv2.getTrackbarPos('Hue_min', 'Trackbars')
    min_HSV[1] = cv2.getTrackbarPos('Sat_min', 'Trackbars')
    min_HSV[2] = cv2.getTrackbarPos('Val_min', 'Trackbars')
    max_HSV[0] = cv2.getTrackbarPos('Hue_max', 'Trackbars')
    max_HSV[1] = cv2.getTrackbarPos('Sat_max', 'Trackbars')
    max_HSV[2] = cv2.getTrackbarPos('Val_max', 'Trackbars')

    # Create mask dynamically based on current trackbar values
    mask = create_mask(frame, min_HSV, max_HSV)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display original frame and the dynamically generated mask
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Dynamic Mask', mask)
    # cv2.imshow('Result', result)

    # Press 's' to save the updated trackbar values
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Values Saved:", min_HSV, max_HSV)
        # Store the updated trackbar values for further processing or use

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
