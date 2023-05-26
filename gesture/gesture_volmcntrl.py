import cv2
import mediapipe as mp
from hand_tracking_module import HandTracker
import math
import numpy as np
import pycaw  # it's a windows library developed by AndreMiras
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# for audio controll
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# ///////////////////////////////////////////////////////////

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
        # print(landmarks[2], landmarks[4])
        x1, y1 = landmarks[4][1], landmarks[4][2]
        x2, y2 = landmarks[8][1], landmarks[8][2]
        cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)

        length = math.hypot(x2-x1, y2-y1)
        # print(length) #min : 10,max : 300
        if length > 150:
            cv2.circle(frame, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # volume controll
        volrange = volume.GetVolumeRange()
        min_vol = volrange[0]
        max_vol = volrange[1]
        vol = np.interp(length, [10, 250], [min_vol, max_vol])
        vol_bar = np.interp(length, [10, 250], [400, 150])
        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(frame, (50, int(vol_bar)), (85, 400),
                      (0, 255, 0), cv2.FILLED)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
