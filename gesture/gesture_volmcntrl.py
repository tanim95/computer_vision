import cv2
import mediapipe as mp
from hand_tracking_module import HandTracker
import time
import numpy as np

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(3, 640)
video.set(4, 480)
hand_tracker = HandTracker()

while True:
    ret, frame = video.read()
    if not ret:
        break

    hand_tracker.track_hands(frame=frame)
    lndmrk_list = hand_tracker.point_position(frame=frame)
    print(lndmrk_list)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
