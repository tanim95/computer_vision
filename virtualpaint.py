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

# def show_vid():
#     video = cv2.VideoCapture(0)
#     video.set(3, 640)
#     video.set(4, 480)

#     # 3 diffrent color pen mask
#     my_colors = [[35, 93, 59, 94, 179, 179], [
#         50, 80, 70, 100, 120, 125], [45, 80, 65, 100, 150, 150]]

#     def find_color(img, colors):
#         hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         masks = []
#         for color in colors:
#             lower = np.array(color[0:3])
#             upper = np.array(color[3:6])
#             mask = cv2.inRange(hsv_img, lower, upper)
#             masks.append(mask)

#         combined_mask = cv2.bitwise_or(*masks)
#         find_contours(img, combined_mask)
#         cv2.imshow('Combined Mask', combined_mask)
#         cv2.waitKey(1)

#     def find_contours(img, mask):
#         contours, _ = cv2.findContours(
#             mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

#     while True:
#         ret, frame = video.read()
#         copied_img = frame.copy()
#         find_color(copied_img, my_colors)

#         if not ret:
#             break

#         cv2.imshow('video', copied_img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video.release()
#     cv2.destroyAllWindows()


# show_vid()

def show_vid():
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 480)

    # single color pen mask
    my_colors = [35, 93, 59, 94, 179, 179]
    colr = (0, 255, 0)
    moving_points = []  # [x,y]

    def find_color(img, color):
        newpoints = []
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(hsv_img, lower, upper)
        x, y = find_contours(img, mask)
        cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
        if (x != 0 and y != 0):
            newpoints.append([x, y])
        cv2.imshow('Mask', mask)
        cv2.waitKey(1)
        return newpoints

    def find_contours(img, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x+w // 2, y

    def draw(img, points, color):
        for point in points:
            cv2.circle(img, (point[0], point[1]), 10, color, cv2.FILLED)

    while True:
        ret, frame = video.read()
        copied_img = frame.copy()
        newpoints = find_color(copied_img, my_colors)
        if newpoints is not None:
            for p in newpoints:
                moving_points.append(p)
        if moving_points is not None:
            draw(copied_img, moving_points, colr)

        if not ret:
            break

        cv2.imshow('video', copied_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


show_vid()
