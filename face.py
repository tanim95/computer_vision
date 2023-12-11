import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
local_xml = cv2.CascadeClassifier('face_detection.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('can not open webcam')
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = local_xml.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# haarcascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# with open(haarcascade, 'r') as file:
#     xml_content = file.read()
# with open('face_detection.xml', 'w') as output_file:
#     output_file.write(xml_content)
