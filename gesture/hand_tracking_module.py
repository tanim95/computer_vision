import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video.set(3, 640)
        self.video.set(4, 480)
        self.prev_time = time.time()
        self.results = None
        self.landmarks = None

    def track_hands(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Cannot receive frame (stream end?). Exiting...")
                break

            with self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.results = hands.process(image_rgb)

                if self.results.multi_hand_landmarks:
                    for hland in self.results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hland, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing.DrawingSpec(
                                color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                self.display_fps(frame)
                self.get_landmarks(frame)

                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.video.release()
        cv2.destroyAllWindows()

    def get_landmarks(self, frame, hand=0):
        self.landmarks = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand]
            for i, l in enumerate(myhand.landmark):
                h, w, _ = frame.shape
                cx, cy = int(l.x * w), int(l.y * h)
                self.landmarks.append([i, cx, cy])

    def display_fps(self, frame):
        cur_time = time.time()
        fps = 1 / (cur_time - self.prev_time)
        self.prev_time = cur_time
        cv2.putText(frame, f"FPS: {round(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


def main():
    hand_tracker = HandTracker()
    hand_tracker.track_hands()


if __name__ == '__main__':
    main()
