import logging
from typing import List, Optional

from vidgear.gears import CamGear

import config
import stream

logger = config.setup_logger(level=logging.INFO)


class Reader:
    def __init__(self, filepath: str, fps: int = 60):
        self._frame_index = 1
        self._fps = fps
        self._can_read = True
        is_stream = filepath.startswith("https")
        self._stream = CamGear(source=filepath, stream_mode=is_stream, logging=True).start()

    def __del__(self):
        self._stream.stop()

    @property
    def fps(self) -> int:
        return self._fps

    def read(self, sec: int) -> List[stream.Frame]:
        if not self._can_read:
            raise Exception("Cannot read stream")

        frame_buffer: List[stream.Frame] = []
        for i in range(0, self._fps * sec):
            frame = self._stream.read()
            if frame is None:
                self._can_read = False
                logger.error("No frame was read", extra={"frame": self._frame_index})
                break

            frame_buffer.append(stream.Frame(frame, self._frame_index))
            self._frame_index += 1

        return frame_buffer

    def read_until_spin_found(self, sec: int) -> List[stream.Frame]:
        last_frame: Optional[stream.Frame] = None
        while True:
            buffer = self.read(sec)
            if last_frame is not None:
                buffer.insert(0, last_frame)

            last_frame = buffer[-1]
            if last_frame.is_spin_frame():
                break
            logger.warning(f"Frame on {last_frame.index / self._fps}s ({last_frame.index}f) is not a spin")

        return buffer

    def can_read(self) -> bool:
        return self._can_read
