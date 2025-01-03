import cv2
import numpy as np

import config


def draw_circle(frame: np.ndarray, circle: np.ndarray, title: str = "Circle") -> None:
    image = frame.copy()
    center_x, center_y, radius = circle
    cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), thickness=5)
    cv2.circle(image, (center_x, center_y), 2, (0, 0, 255), 3)

    visualize(frame, title)


def draw_contours(frame: np.ndarray, contours: np.ndarray, fill: bool = False, title: str = "Contours") -> None:
    image = frame.copy()
    thickness = cv2.FILLED if fill else -1
    for i, contour in enumerate(contours):
        cv2.drawContours(image, contours, i, (255, 255, 255), thickness)

    visualize(image, title)


def draw_lines(frame: np.ndarray, lines: np.ndarray, title: str = "Lines") -> None:
    image = frame.copy()
    for i, line in enumerate(lines):
        if len(line) == 1:
            x1, y1, x2, y2 = line[0]
        elif len(line) == 4:
            x1, y1, x2, y2 = line
        else:
            raise Exception("invalid format")

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    visualize(image, title)


def visualize(frame: np.ndarray, title: str = "Image"):
    if config.VISUALIZATION_ENABLED:
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
