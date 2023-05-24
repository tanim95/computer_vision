import cv2
import mediapipe as mp


def hand_tracking():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Configure video capture
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            # Read frame from video capture
            ret, frame = video.read()
            if not ret:
                break

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand tracking
            results = hands.process(image_rgb)

            # Draw landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame
            cv2.imshow('Hand Tracking', frame)

            # Exit loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()


# Call the hand_tracking function
hand_tracking()
