from __future__ import annotations

import logging
import math
from typing import List, Optional

import config
import stream
import utils

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

    def _detect_length(self) -> int | List[int]:
        self._populate_first_spins()

        try:
            # The first spin frame.
            length = self._first_spins_buffer[0].detect_length()
            logger.info(f"Length was detected via screen parsing: {length}")
            return length
        except Exception as e:
            logger.error(f"The length wasn't detected via screen parsing: {e}")

        if config.ASK_LENGTH:
            # Manual enter
            with open("/dev/tty", "r+") as tty:
                tty.write("Cannot detect length. Manual enter: ")
                tty.flush()
                response = tty.readline().strip()
                tty.write("")

            if response.isdigit():
                length = int(response)
                logger.info(f"Length was set to {length}")

                return length

        raise Exception("cannot detect the length")

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

        # TODO: why was it added?
        self._init_frame.wheel[2] -= 1

        # if config.NASTY_OPTIMIZATION:
        #     [frame.force_set_wheel(self._init_frame.wheel) for frame in self._first_spins_buffer]

    def _populate_first_spins_with_length(self, length: int) -> None:
        if len(self._first_spins_buffer) == 0:
            self._populate_first_spins()

        min_range, max_range = utils.range(length)
        first_wheel_frames = math.ceil(
            utils.calculate_x(config.SPIN_BUFFER_SIZE * 360.0 / min_range) * self._reader.fps * length
        )
        sec = math.ceil(max(first_wheel_frames - len(self._first_spins_buffer), 0) / self._reader.fps)
        self._first_spins_buffer.extend(self._reader.read(sec))

    def detect_winner(self):
        length = self._detect_length()
        self._populate_first_spins_with_length(length)

        idx = 0
        # min_range, max_range = utils.range(length)
        max_read_frames = length * self._reader.fps
        max_skip_frames = max(
            self._reader.fps * config.MIN_SKIP_OF_WHEEL_SPIN * length,
            self._reader.fps * config.MIN_SKIP_SEC
        )
        skipped_frames = 0

        buffer = self._first_spins_buffer
        angles_window: List[(int, float)] = []

        logger.info(f"Skipping {round(max_skip_frames / self._reader.fps, 2)}s")

        while True:
            for frame in buffer:
                idx += 1
                if skipped_frames < max_skip_frames:
                    skipped_frames += 1
                    continue

                # Collect a window with angles. The size of window is config.ANGLE_WINDOW_LEN
                try:
                    angle = self._init_frame.calculate_rotation_with(frame)
                    angles_window.append((idx, angle,))
                except Exception as e:
                    logger.error("cannot calculate angle", extra={"frame_id": idx, "e": e})
                    continue

            if idx >= max_read_frames:
                break

            sec = math.ceil(min(max_read_frames - idx, self._reader.fps * config.READ_STEP) / self._reader.fps)
            buffer = self._reader.read(sec)

        return None
