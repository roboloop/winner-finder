import cv2
import numpy as np


def draw_circle(frame: np.ndarray, circle: np.ndarray, title: str = "Circle") -> None:
    image = frame.copy()
    center_x, center_y, radius = circle
    cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), thickness=5)
    cv2.circle(image, (center_x, center_y), 2, (0, 0, 255), 3)

    visualize(frame, title)


def visualize(frame: np.ndarray, title: str = "Image"):
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
