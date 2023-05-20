import cv2
import numpy as np


def find_color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_img, lower, upper)
    cv2.imshow('face', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video():
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 480)
    while True:
        ret, frame = video.read()

        if not ret:
            break
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


show_video()
