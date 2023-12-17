import numpy as np
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt


def load_img(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (600, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def display_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Load the image
image, _ = load_img('img/win 10.jpg')
img = np.copy(image)
marker_image = np.zeros(img.shape[:2], dtype=np.int32)
segments = np.zeros(img.shape, dtype=np.uint8)


def create_color(i):
    return tuple(np.array(cm.Set1(i)[:3]) * 255)


colors = [create_color(i) for i in range(10)]

# Global variables
current_marker = 1
marker_updated = False

# Mouse callback function


def mouse_callback(e, x, y, flags, params):
    global marker_updated
    if e == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)
        cv2.circle(img, (x, y), 10, colors[current_marker], -1)
        marker_updated = True


cv2.namedWindow('Ocean and Mountain')
cv2.setMouseCallback('Ocean and Mountain', mouse_callback)

while True:
    cv2.imshow('Watershed segments', segments)
    cv2.imshow('Ocean and Mountain', img)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('c'):
        img = image.copy()
        marker_image = np.zeros(img.shape[:2], dtype=np.int32)
        segments = np.zeros(img.shape, dtype=np.uint8)
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))
    if marker_updated:
        marker_img_copy = marker_image.copy()
        cv2.watershed(image, marker_img_copy)
        segments = np.zeros(image.shape, dtype=np.uint8)
        for c in range(10):
            segments[marker_img_copy == (c)] = colors[c]
        marker_updated = False

cv2.destroyAllWindows()
