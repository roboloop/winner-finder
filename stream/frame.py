from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract

import config

logger = config.setup_logger(level=logging.INFO)


TOP_OFFSET = 70

class Frame:
    _wheel: Optional[np.ndarray]
    _lot_name: Optional[str]
    _init_text = ["победитель", "winner"]

    def __init__(self, frame: np.ndarray, index: int):
        self._frame = frame
        self._index = index
        self._wheel = None
        self._lot_name = None

    @property
    def frame(self) -> np.ndarray:
        return self._frame

    @property
    def index(self) -> int:
        return self._index

    @property
    def wheel(self) -> np.ndarray:
        return self.detect_wheel()

    def detect_wheel(self) -> np.ndarray:
        if self._wheel is not None:
            return self._wheel

        candidates = [((11, 11), 4), ((11, 11), 2)]
        for ksize, sigma_x in candidates:
            gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, ksize, sigma_x)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=500,
                param1=100,
                param2=100,
                minRadius=250,
                maxRadius=600,
            )

            if circles is None:
                continue

            circles = np.uint16(np.around(circles))
            self._wheel = max(circles[0], key=lambda c: c[2])
            return self._wheel

        raise Exception("cannot detect wheel")

    def detect_lot_name(self) -> str:
        if self._lot_name is not None:
            return self._lot_name

        center_x, center_y, radius = self.wheel
        roi_y_start = max(0, center_y - radius - TOP_OFFSET)
        roi_y_end = max(0, center_y - radius - 5)
        roi_x_start = max(0, center_x - radius)
        roi_x_end = min(self._frame.shape[1], center_x + radius)

        text_roi = self._frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

        text = pytesseract.image_to_string(thresh, lang=config.TESSERACT_LANG, config="--psm 6")
        logger.info("Lot name was detected", extra={"text": text.strip(), "frame": {self._index}})

        self._lot_name = text.strip()

        return self._lot_name

    def is_init_frame(self) -> bool:
        try:
            text = self.detect_lot_name()
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract was not found")
            raise
        except Exception:
            return False

        # search substring 'победитель' или 'winner' in the frame. This prevents some false parsing
        return any(substring in text.lower() for substring in self._init_text)

    def is_spin_frame(self) -> bool:
        try:
            text = self.detect_lot_name()
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract was not found")
            raise
        except Exception as e:
            return False

        return not any(substring in text.lower() for substring in self._init_text)

    # TODO: doesn't work right. It finds a strange contours
    def is_circle(self):
        circle = self.extract_circle_content()
        gray = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return False
        ellipse = cv2.fitEllipse(contour)
        (x, y), (major_axis, minor_axis), angle = ellipse
        return abs(major_axis - minor_axis) / max(major_axis, minor_axis) < 0.5

    def extract_circle_content(self) -> np.ndarray:
        center_x, center_y, radius = self.wheel

        mask = np.zeros(self._frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        circle_content = cv2.bitwise_and(self._frame, self._frame, mask=mask)
        x1, y1, x2, y2 = center_x - radius, center_y - radius, center_x + radius, center_y + radius
        cropped = circle_content[max(0, y1) : y2, max(0, x1) : x2]

        return cropped
