import logging
from typing import List, Optional

from vidgear.gears import CamGear

import config
import stream

logger = config.setup_logger(level=logging.INFO)


class Reader:
    def __init__(self, filepath: str, fps: int = 60, skip_sec: int = 0):
        self._frame_index = 1
        self._fps = fps
        self._skip_sec = skip_sec
        self._is_skipped = False
        self._can_read = True
        is_stream = filepath.startswith("https")
        self._stream = CamGear(source=filepath, stream_mode=is_stream, logging=True).start()
        self._is_first_read = True

    def __del__(self):
        self._stream.stop()

    @property
    def fps(self) -> int:
        return self._fps

    def _skip_read(self) -> None:
        if self._skip_sec == 0 or self._is_skipped:
            return

        logger.info(f"Skipping {self._skip_sec}s")
        for i in range(0, self._fps * self._skip_sec):
            self._stream.read()
            self._frame_index += 1
        self._is_skipped = True

    def _emit_start(self) -> None:
        if self._is_first_read:
            self._is_first_read = False
            config.update_global_event_time()

    def read(self, sec: int) -> List[stream.Frame]:
        if not self._can_read:
            raise Exception("Cannot read stream")

        self._skip_read()
        self._emit_start()

        frame_buffer: List[stream.Frame] = []
        for i in range(0, self._fps * sec):
            frame = self._stream.read()
            if frame is None:
                self._can_read = False
                logger.error("No frame was read", extra={"frame": self._frame_index})
                break

            frame_buffer.append(stream.Frame(frame, self._frame_index))
            self._frame_index += 1

        config.add_global_event_time(sec)

        return frame_buffer

    def read_until_spin_found(self, sec: int) -> List[stream.Frame]:
        last_frame: Optional[stream.Frame] = None
        while True:
            buffer = self.read(sec)
            if last_frame is not None:
                buffer.insert(0, last_frame)

            last_frame = buffer[-1]
            if last_frame.is_spin_frame():
                # Take a frame before the previous one, to avoid a case when two serial frames are the same frames.
                prev_frame = buffer[-3]
                try:
                    diff = last_frame.calculate_rotation_with(prev_frame)
                    # else delta is so small then it's not spin
                    if min(diff, abs(diff - 360)) > 0.01:
                        break

                    logger.warning(
                        f"Frame on {last_frame.index / self._fps}s ({last_frame.index}f) looks like a spin but spin delta is low"
                    )
                    continue
                except Exception:
                    pass
            logger.warning(f"Frame on {last_frame.index / self._fps}s ({last_frame.index}f) is not a spin")

        # Nasty optimization for all other wheels
        # for frame in buffer:
        #     frame.force_set_wheel(last_frame.wheel)

        return buffer

    def skip(self, sec: int) -> None:
        logger.info(f"Extra skipping {sec}s")
        skip_frames = self._fps * sec
        self._frame_index += skip_frames

        for i in range(0, skip_frames):
            self._stream.read()

    def can_read(self) -> bool:
        return self._can_read
