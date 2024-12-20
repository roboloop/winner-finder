from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

class Frame:
    _wheel: Optional[np.ndarray]

    def __init__(self, frame: np.ndarray, index: int):
        self._frame = frame
        self._index = index
        self._wheel = None

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

