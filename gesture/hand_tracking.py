import cv2
import mediapipe as mp


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

        while True:
            ret, frame = video.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # tracking
            results = hands.process(image_rgb)

            # landmarks on hand
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


# Call the hand_tracking function
hand_tracking()
