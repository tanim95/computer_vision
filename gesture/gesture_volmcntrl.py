import cv2
import mediapipe as mp
from hand_tracking_module import HandTracker
import time
import numpy as np


video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(3, 640)
video.set(4, 480)
tracker = HandTracker()

while True:
    ret, frame = video.read()
    if not ret:
        break

    tracker.track_hands(frame=frame)
    landmarks = tracker.landmarks
    if len(landmarks) != 0:
        print(landmarks[2])
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
