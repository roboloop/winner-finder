from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import queue
from typing import List, Optional

import numpy as np

import config
import stream

logger = config.setup_logger(level=logging.INFO)

class Segment:
    def __init__(self, reader: stream.Reader):
        self._reader = reader
        self._first_spins_buffer: List[stream.Frame] = []
        self._init_frame: Optional[stream.Frame] = None

    def _binary_search(self, segment: List[stream.Frame]) -> Optional[int]:
        low = 0
        high = len(segment) - 1

        while low <= high:
            mid = (low + high) // 2
            if segment[mid].is_init_frame():
                if mid == len(segment) - 1 or segment[mid + 1].is_spin_frame():
                    return mid
                low = mid + 1
            else:
                high = mid - 1

        return None


    def _populate_first_spins(self) -> None:
        if len(self._first_spins_buffer) != 0:
            return

        buffer = self._reader.read_until_spin_found(config.READ_STEP)
        logger.info("Spin with a lot name was found")

        _init_frame_index = self._binary_search(buffer)
        if _init_frame_index is None:
            raise Exception("no init frame")
        self._first_spins_buffer = buffer[_init_frame_index + 1 :]
        self._init_frame = buffer[_init_frame_index]
        logger.info("Init frame was found", extra={"frame": {self._init_frame.index}})
        if self._init_frame.is_circle():
            logger.error("Circle is ellipse? The result could be vary")

        self._init_frame.wheel[2] -= 1


    def detect_winner(self):
        pass