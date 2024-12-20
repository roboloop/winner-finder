from typing import List

from vidgear.gears import CamGear

import stream


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
                break

            frame_buffer.append(stream.Frame(frame, self._frame_index))
            self._frame_index += 1

        return frame_buffer

    def can_read(self) -> bool:
        return self._can_read
