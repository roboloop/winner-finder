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

        utils.visualize(text_roi, "text roi before preprocessing")

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

            # drop all rectangles that intersect others
            # keep only with the biggest area.
            filtered_rectangles = sorted(filtered_rectangles, key=lambda item: item[2] * item[3], reverse=True)
            final_rectangles = []
            for rect in filtered_rectangles:
                x1, y1, h1, w1 = rect
                if all(
                    (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
                    for x2, y2, h2, w2 in final_rectangles
                ):
                    final_rectangles.append(rect)

            # keep rectangles from the left to the right
            final_rectangles = sorted(final_rectangles, key=lambda item: item[0])

            # if there are only 3 rectangles: Крутимся, От и До, then it was not successful
            if len(final_rectangles) == 3:
                raise Exception("range length was detected")

            # if there are only 2 rectangles: Крутимся и Длительность, then it was successful
            if len(final_rectangles) == 2:
                # x, y, w, h = max(final_rectangles, key=lambda item: item[1])
                x, y, w, h = final_rectangles[1]
                return block_roi[y + 8 : y + h - 8, x + 5 : x + w - 5]

            # if there is only one rectangle: Крутимся, then look to the left
            if len(final_rectangles) == 1:
                x, y, w, h = final_rectangles[0]
                if (
                    y + 8 < block_roi.shape[0]
                    and y + h - 8 <= block_roi.shape[0]
                    and x + w + 20 < block_roi.shape[1]
                    and x + w + 20 + 50 <= block_roi.shape[1]
                ):
                    return block_roi[y + 8 : y + h - 8, x + w + 20 : x + w + 20 + 50]

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

    def _extract_raw_lines(self) -> np.ndarray:
        image = self.extract_circle_content(True)
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255

        # fill out the text on the wheel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            # filter out the contour that may contain text
            if w + h < 60:
                cv2.drawContours(mask, contours, i, 0, thickness=cv2.FILLED)

        # fill out the circle center on the wheel and around that center space to avoid collisions
        _, _, radius = self._wheel
        cv2.circle(mask, (radius, radius), int(0.5 * radius), 0, thickness=cv2.FILLED)

        # mask is ready, apply to the image to extact lines
        masked = cv2.bitwise_and(image, image, mask=mask)
        masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        _, masked_thresh = cv2.threshold(masked_gray, 215, 255, cv2.THRESH_BINARY)

        return cv2.HoughLinesP(masked_thresh, 1, np.pi / 180, 100, minLineLength=75, maxLineGap=10)

    def _handle_raw_lines(self, lines: np.ndarray, radius: int) -> List[tuple[np.ndarray, float, float, np.ndarray]]:
        cx, cy = radius, radius

        def _to_line_payload(line: np.ndarray) -> Optional[tuple[np.ndarray, float, float]]:
            x1, y1, x2, y2 = line[0]
            # Line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            distance_to_center = abs(a * cx + b * cy + c) / math.sqrt(a**2 + b**2)
            if distance_to_center > 5.0:
                return None

            d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
            d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
            farthest_x, farthest_y = (x1, y1) if d1 > d2 else (x2, y2)
            angle = math.atan2(farthest_y - cy, farthest_x - cx)

            return line[0], distance_to_center, angle

        def _to_prolong_line(line: np.ndarray) -> np.ndarray:
            x1, y1, x2, y2 = line
            x_end, y_end = (
                (x2, y2) if ((x1 - cx) ** 2 + (y1 - cy) ** 2) < ((x2 - cx) ** 2 + (y2 - cy) ** 2) else (x1, y1)
            )

            dx, dy = x_end - cx, y_end - cy
            length = math.sqrt(dx**2 + dy**2)
            direction_x, direction_y = dx / length, dy / length

            new_x = cx + radius * direction_x
            new_y = cy + radius * direction_y

            return np.array([cx, cy, int(new_x), int(new_y)])

        def _uniq_lines(
            line_payloads: List[tuple[np.ndarray, float, float]]
        ) -> List[tuple[np.ndarray, float, float, np.ndarray]]:
            line_payloads.sort(key=lambda p: (p[2], p[1]))

            groups = []
            current_group = [line_payloads[0]]
            for i in range(1, len(line_payloads)):
                prev_slope = current_group[-1][2]
                current_slope = line_payloads[i][2]

                # 0.01 radian ~ 0.57°
                if abs(current_slope - prev_slope) < 0.01:
                    current_group.append(line_payloads[i])
                else:
                    groups.append(current_group)
                    current_group = [line_payloads[i]]
            groups.append(current_group)

            uniq_lines_: List[tuple[np.ndarray, float, float, np.ndarray]] = []
            for group in groups:
                closest_line = min(group, key=lambda p: p[1])  # Line with the smallest distance
                a, b, c = closest_line
                d = _to_prolong_line(a)
                uniq_lines_.append((a, b, c, d))

            return uniq_lines_

        not_filtered_line_payloads = map(_to_line_payload, lines)
        line_payloads = list(filter(lambda p: p is not None, not_filtered_line_payloads))

        return _uniq_lines(line_payloads)

    def extract_sectors(self) -> List[tuple[float, float]]:
        _, _, radius = self.wheel
        raw_lines = self._extract_raw_lines()
        handled_lines = self._handle_raw_lines(raw_lines, radius)

        layout = np.zeros_like(self.extract_circle_content(True))
        utils.draw_lines(layout, raw_lines, "raw lines")
        utils.draw_lines(layout, [l[0] for l in handled_lines], "filtered raw lines")
        utils.draw_lines(layout, [l[3] for l in handled_lines], "handled filtered raw lines")

        unsorted_angels = map(lambda l: l[2], handled_lines)
        angles = sorted(unsorted_angels)
        sectors: List[tuple[float, float]] = []
        for i in range(len(angles)):
            start_angle = angles[i]
            end_angle = angles[(i + 1) % len(angles)]

            # normalize angle relatively to North (counterclockwise)
            normalized_start_angle = (3 * math.pi / 2 - start_angle) % (2 * math.pi)
            normalized_end_angle_deg = (3 * math.pi / 2 - end_angle) % (2 * math.pi)

            start_angle_deg = math.degrees(normalized_start_angle)
            end_angle_deg = math.degrees(normalized_end_angle_deg)

            sectors.append((end_angle_deg, start_angle_deg))

        sectors.sort(key=lambda a: a[0])

        utils.draw_sectors(layout, radius, sectors, "Sectors")

        return sectors

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
