import cv2
import numpy as np


def show_video():
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 480)

    def empty(a):
        pass

    cv2.namedWindow('HSV')
    cv2.resizeWindow('HSV', 640, 480)
    cv2.createTrackbar('HUE Min', 'HSV', 0, 179, empty)
    cv2.createTrackbar('SAT Min', 'HSV', 0, 255, empty)
    cv2.createTrackbar('VALUE Min', 'HSV', 0, 255, empty)
    cv2.createTrackbar('HUE Max', 'HSV', 179, 179, empty)
    cv2.createTrackbar('SAT Max', 'HSV', 255, 179, empty)
    cv2.createTrackbar('VALUE Max', 'HSV', 255, 179, empty)

    while True:
        ret, frame = video.read()
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos('HUE Min', 'HSV')
        h_max = cv2.getTrackbarPos('HUE Max', 'HSV')
        s_min = cv2.getTrackbarPos('SAT Min', 'HSV')
        s_max = cv2.getTrackbarPos('SAT Max', 'HSV')
        v_min = cv2.getTrackbarPos('VALUE Min', 'HSV')
        v_max = cv2.getTrackbarPos('VALUE Max', 'HSV')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(frame_hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        if not ret:
            break

        cv2.imshow('video', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

# show_video()


def show_vid():
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 480)

    # Single color pen mask
    my_colors = [35, 93, 59, 94, 179, 179]
    colr = (0, 255, 0)
    moving_points = []  # [x,y]

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert frame to HSV color space
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a color mask
        lower = np.array(my_colors[0:3])
        upper = np.array(my_colors[3:6])
        mask = cv2.inRange(hsv_img, lower, upper)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process only the largest contour
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colr, 2)
            moving_points.append((x + w // 2, y))

        # Draw circles at detected points
        for point in moving_points:
            cv2.circle(frame, point, 10, colr, cv2.FILLED)

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


show_vid()
