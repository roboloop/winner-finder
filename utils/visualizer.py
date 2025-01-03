import math
from functools import partial
from typing import List

import cv2
import numpy as np

import config
import utils


def draw_circle(frame: np.ndarray, circle: np.ndarray, title: str = "Circle") -> None:
    image = frame.copy()
    center_x, center_y, radius = circle
    cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), thickness=5)
    cv2.circle(image, (center_x, center_y), 2, (0, 0, 255), 3)

    visualize(frame, title)


def draw_contours(frame: np.ndarray, contours: np.ndarray, fill: bool = False, title: str = "Contours") -> None:
    image = frame.copy()
    colors = utils.generate_diverse_colors(len(contours))
    thickness = cv2.FILLED if fill else -1
    for i, contour in enumerate(contours):
        cv2.drawContours(image, contours, i, colors[i], thickness)

    visualize(image, title)


def draw_lines(frame: np.ndarray, lines: np.ndarray, title: str = "Lines") -> None:
    image = frame.copy()
    colors = utils.generate_diverse_colors(len(lines))
    for i, line in enumerate(lines):
        if len(line) == 1:
            x1, y1, x2, y2 = line[0]
        elif len(line) == 4:
            x1, y1, x2, y2 = line
        else:
            raise Exception("invalid format")

        cv2.line(image, (x1, y1), (x2, y2), colors[i], 1)

    visualize(image, title)


def draw_sectors(frame: np.ndarray, radius: int, sectors: List[tuple[float, float]], title: str = "Sectors") -> None:
    def _draw_rotated_text(image: np.ndarray, text: str, x_y: tuple[float, float], angle: float) -> None:
        thickness = 1
        font_scale = 0.5
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        canvas_size = int(np.hypot(text_width, text_height)) + 20
        text_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        text_x = (canvas_size - text_width) // 2
        text_y = (canvas_size + text_height) // 2
        cv2.putText(
            text_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness, cv2.LINE_AA
        )

        center = (canvas_size // 2, canvas_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_text = cv2.warpAffine(
            text_img,
            rotation_matrix,
            (canvas_size, canvas_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        mask = cv2.cvtColor(rotated_text, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        x, y = x_y
        y1, y2 = y - canvas_size // 2, y + canvas_size // 2
        x1, x2 = x - canvas_size // 2, x + canvas_size // 2

        if y1 < 0 or x1 < 0 or y2 >= image.shape[0] or x2 >= image.shape[1]:
            return
        roi = image[y1:y2, x1:x2]
        roi[np.where(mask)] = rotated_text[np.where(mask)]

    def _denormalize_angle(angle: float) -> float:
        return math.degrees(3 * math.pi / 2 - math.radians(angle))

    image = frame.copy()
    colors = utils.generate_diverse_colors(len(sectors))
    deferred_calls = []
    for i, sector in enumerate(sectors):
        start_angle, end_angle = sector
        # Make a text before normalization
        text = f"{round(start_angle, 2)}-{round(end_angle, 2)}"

        denormalized_end_angle_deg = _denormalize_angle(start_angle)
        denormalized_start_angle = _denormalize_angle(end_angle)
        if denormalized_start_angle > denormalized_end_angle_deg:
            denormalized_start_angle -= 360

        mid_angle = (denormalized_start_angle + denormalized_end_angle_deg) / 2
        x = int(radius + radius * 0.8 * np.cos(np.radians(mid_angle)))
        y = int(radius + radius * 0.8 * np.sin(np.radians(mid_angle)))
        text_angle = -mid_angle if mid_angle <= 90 or mid_angle >= 270 else (180 - mid_angle)

        # draw a conu part
        cv2.ellipse(
            image,
            (radius, radius),
            (radius, radius),
            0,
            denormalized_start_angle,
            denormalized_end_angle_deg,
            colors[i],
            -1,
            cv2.LINE_AA,
        )
        # the text will be drawn latter
        deferred_calls.append(partial(_draw_rotated_text, image, text, (x, y), text_angle))

    [call() for call in deferred_calls]

    visualize(image, title)


def visualize(frame: np.ndarray, title: str = "Image"):
    if config.VISUALIZATION_ENABLED:
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
