import cv2
import mediapipe as mp
import time


def hand_tracking():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video.set(3, 640)
    video.set(4, 480)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        prev_time = time.time()
        while True:
            ret, frame = video.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # tracking
            results = hands.process(image_rgb)

            # landmarks on hand
            if results.multi_hand_landmarks:
                for hland in results.multi_hand_landmarks:
                    for i, l in enumerate(hland.landmark):
                        h, w, c = frame.shape
                        # as x & y values are in ratio i converted it to pixel by multiplying with height and width
                        cx, cy = int(l.x * w), int(l.y * h)
                        # print(i,cx, cy)
                    mp_drawing.draw_landmarks(
                        frame, hland, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            prev_time = cur_time
            cv2.putText(frame, f"FPS: {round(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


hand_tracking()
