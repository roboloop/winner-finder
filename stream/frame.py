from __future__ import annotations

import collections
import logging
import math
import re
from typing import List, Optional

import cv2
import numpy as np
import pytesseract

import config
import stream
import utils

logger = config.setup_logger(level=logging.INFO)


TOP_OFFSET = 70

class Frame:
    _wheel: Optional[np.ndarray]
    _lot_name: Optional[str]
    _rotation_angle: Optional[float]
    _init_text = ["победитель", "winner"]

    def __init__(self, frame: np.ndarray, index: int):
        self._frame = frame
        self._index = index
        self._wheel = None
        self._lot_name = None
        self._rotation_angle = None

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

        utils.visualize(text_roi, 'text roi before preprocessing')

        text = pytesseract.image_to_string(thresh, lang=config.TESSERACT_LANG, config="--psm 6")
        logger.info("Lot name was detected", extra={"text": text.strip(), "frame": {self._index}})

        self._lot_name = text.strip()

        return self._lot_name

    def _find_length_section(self, block_roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(block_roi, cv2.COLOR_BGR2GRAY)
        # 50 and 100 — heuristics value. For someone's stream 5 and 10 should be used
        candidates = [(50, 100), (5, 10)]
        for threshold1, threshold2 in candidates:
            edges = cv2.Canny(gray, threshold1, threshold2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangles = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # 70 (50 for the range length) and 28 the min size of the target rectangle
                if w >= 50 and h >= 28:
                    rectangles.append((x, y, w, h))

            # drop all inner rectangles
            filtered_rectangles = []
            for i, (x1, y1, w1, h1) in enumerate(rectangles):
                is_inner = False
                for j, (x2, y2, w2, h2) in enumerate(rectangles):
                    if i != j and x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                        is_inner = True
                        break
                if not is_inner:
                    filtered_rectangles.append((x1, y1, w1, h1))


            # if there are only 2 rectangles: Крутимся и Длительность, then it was successful
            if len(filtered_rectangles) == 2:
                # x, y, w, h = max(final_rectangles, key=lambda item: item[1])
                x, y, w, h = filtered_rectangles[1]
                return block_roi[y + 8 : y + h - 8, x + 5 : x + w - 5]

        raise Exception("length roi wasn't found")

    def _detect_length(self, roi: np.ndarray) -> int:
        candidates = []
        for threshold in range(170, 140, -5):
            _, thresh = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresh, config="--psm 6 -c tessedit_char_whitelist=0123456789")
            matches = re.findall(r"\b(\d{2,3})\b", text)
            if not matches:
                continue

            candidates.append(int(matches[0]))

        if len(candidates) == 0:
            raise Exception(f"no candidates")

        length, total = collections.Counter(candidates).most_common(1)[0]
        logger.info("Length candidates", extra={"candidates": candidates})
        if total < 3:
            raise Exception(f"not enough the candidates of {length}")

        return length

    def detect_length(self) -> int:
        circle = self.detect_wheel()

        center_x, center_y, radius = circle
        roi_y_start = max(0, center_y - radius - TOP_OFFSET)
        roi_y_end = max(0, center_y - radius)
        roi_x_start = max(0, center_x + radius)
        roi_x_end = min(self._frame.shape[1], center_x + radius + radius)
        block_roi = self._frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        utils.visualize(block_roi, "Length section")
        rect = self._find_length_section(block_roi)
        utils.visualize(rect, "Cropped length section")

        length = self._detect_length(rect)

        return length


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
        circle = self.extract_circle_content(False)
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
        # return abs(major_axis - minor_axis) / max(major_axis, minor_axis) < 0.01

    def extract_circle_content(self, keep_center: bool) -> np.ndarray:
        center_x, center_y, radius = self.wheel

        mask = np.zeros(self._frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        if not keep_center:
            inner_radius = int(0.25 * radius)
            cv2.circle(mask, (center_x, center_y), inner_radius, 0, -1)

        circle_content = cv2.bitwise_and(self._frame, self._frame, mask=mask)
        x1, y1, x2, y2 = center_x - radius, center_y - radius, center_x + radius, center_y + radius
        cropped = circle_content[max(0, y1) : y2, max(0, x1) : x2]

        return cropped

    def force_set_wheel(self, wheel: np.ndarray) -> None:
        # TODO: no copy?
        self._wheel = wheel

    def calculate_rotation_with(self, another_frame: stream.Frame) -> float:
        if another_frame._rotation_angle is not None:
            return another_frame._rotation_angle

        gray1 = cv2.cvtColor(self.extract_circle_content(False).copy(), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(another_frame.extract_circle_content(False).copy(), cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.float64([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float64([keypoints2[m.trainIdx].pt for m in matches])

        if len(points1) < 4:
            raise Exception("Not enough matches to compute homography.")

        matrix, _ = cv2.estimateAffinePartial2D(points1, points2)
        if matrix is None:
            raise Exception("Homography matrix could not be computed.")

        angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * (180 / np.pi)
        another_frame._rotation_angle = float(angle if angle >= 0 else angle + 360)

        return another_frame._rotation_angle
