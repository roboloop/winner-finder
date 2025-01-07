from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import queue
from typing import List, Optional, Dict

import numpy as np

import config
import stream
import utils

logger = config.setup_logger(level=logging.INFO)

G = "\033[92m"
B = "\033[94m"
E = "\033[0m"


class Segment:
    def __init__(self, reader: stream.Reader):
        self._reader = reader
        self._first_spins_buffer: List[stream.Frame] = []
        self._init_frame: Optional[stream.Frame] = None
        self._circle_sectors: Optional[stream.CircleSectors] = None

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

        if config.SPIN_DETECT_LENGTH:
            try:
                length = self._populate_first_n_spins(config.SPIN_BUFFER_SIZE_FOR_LENGTH_DETECTION)
                logger.info(f"Length was detected via wheel spin: {length}")
                return length
            except Exception as e:
                logger.error(f"The length wasn't detected via wheel spin: {e}")

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

    def _populate_first_n_spins(self, max_spins: int) -> int:
        def _binary_search(part: List[stream.Frame]) -> Optional[int]:
            low = 0
            high = len(part) - 1
            while low <= high:
                mid = (low + high) // 2
                # if mid == len(part) - 1:
                #     return mid
                if mid == 0:
                    return None
                angle = self._init_frame.calculate_rotation_with(part[mid])
                prev_angle = self._init_frame.calculate_rotation_with(part[mid - 1])
                if angle < prev_angle and abs(360.0 + angle - prev_angle) % 360.0 < 60.0:
                    return mid

                if angle > self._init_frame.calculate_rotation_with(part[-1]):
                    low = mid + 1
                else:
                    high = mid - 1

            return None

        def _find_new_spin_frame(part: List[stream.Frame], reverse: bool) -> int:
            iteration = range(1, len(part)) if reverse is False else range(len(part) - 1, 0, -1)
            for i in iteration:
                angle = self._init_frame.calculate_rotation_with(part[i])
                prev_angle = self._init_frame.calculate_rotation_with(part[i - 1])
                if angle < prev_angle and abs(360.0 + angle - prev_angle) % 360.0 < 60.0:
                    return i

            raise Exception("there is no new spin")

        def _exclude_not_matched_second(buffer: List[stream.Frame], index: int, idx: int, seconds: List[int]):
            part_behind = buffer[max(0, index - config.FRAMES_STEP_FOR_LENGTH_DETECTION):index + 1]
            new_spin_index = _find_new_spin_frame(part_behind, True)
            new_spin_idx = idx - (len(part_behind) - new_spin_index - 1)

            for sec in seconds:
                if spins > round(sec * 270 / 360):
                    logger.error('there is no point to look further', extra={"sec": sec, "spins": spins})
                    continue

                min_range, max_range = utils.range(sec)
                y_min = spins * 360.0 / min_range
                x_min = utils.calculate_x_gsap(y_min)
                idx_min = x_min * (60 * sec)

                y_max = spins * 360.0 / max_range
                x_max = utils.calculate_x_gsap(y_max)
                idx_max = x_max * (60 * sec)

                if not (math.floor(idx_max) <= float(new_spin_idx) <= math.ceil(idx_min)):
                    seconds.remove(sec)

        if len(self._first_spins_buffer) == 0:
            self._populate_first_spins()

        idx = 0
        spins = 0
        prev_angle = 0.0
        starts, ends = config.EXCLUDE_SECONDS_RANGE
        seconds = np.arange(starts, ends + 1).tolist()

        for index, frame in enumerate(self._first_spins_buffer):
            if len(self._first_spins_buffer) - 1 == index:
                self._first_spins_buffer.extend(self._reader.read(config.READ_STEP))

            idx += 1
            if idx % config.FRAMES_STEP_FOR_LENGTH_DETECTION != 0:
                continue

            try:
                if config.NASTY_OPTIMIZATION:
                    frame.force_set_wheel(self._init_frame.wheel)
                angle = self._init_frame.calculate_rotation_with(frame)
            except Exception as e:
                logger.error("cannot calculate angle", extra={"frame_id": idx, "e": e})
                continue

            if angle < prev_angle:
                spins += 1
                _exclude_not_matched_second(self._first_spins_buffer, index, idx, seconds)
                if spins > max_spins or idx > 600:
                    break
                if len(seconds) == 1:
                    logger.info("Only one length candidate has left")
                    break
            prev_angle = angle

        if len(seconds) == 0:
            raise Exception(f"There are no length candidates")

        seconds.sort()

        return seconds

    def _add_lot_names_around(self, angle: float, length: int) -> None:
        min_range, max_range = utils.range(length)
        angles = [angle + spins * 360.0 for spins in range(0, config.SPIN_BUFFER_SIZE)]

        task_queue = queue.Queue()

        for angle in angles:
            start_frame_id = math.floor(utils.calculate_x_gsap(angle / max_range) * self._reader.fps * length)
            end_frame_id = math.ceil(utils.calculate_x_gsap(angle / min_range) * self._reader.fps * length)

            for frame_id in range(start_frame_id, end_frame_id):
                frame = self._first_spins_buffer[frame_id]
                task_queue.put(frame)

        def _worker(task_queue: queue.Queue) -> None:
            while not task_queue.empty():
                frame = task_queue.get()
                lot_name = frame.detect_lot_name()
                frame_angle = self._init_frame.calculate_rotation_with(frame)
                self._circle_sectors.add_lot_name(frame_angle, lot_name)
                task_queue.task_done()

        max_workers = os.cpu_count() - 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(max_workers):
                executor.submit(_worker, task_queue)

        task_queue.join()

    def _filter_anomalies_linear(self, predicted_angles: List[float], deviation: float) -> float:
        filtered = []
        mean = np.mean(predicted_angles)
        deviations = np.abs(np.array(predicted_angles) - mean)
        threshold = np.mean(deviations)
        for j in range(len(predicted_angles)):
            if deviations[j] <= threshold and abs(predicted_angles[j] - mean) <= deviation:
                filtered.append(predicted_angles[j])

        if len(filtered) == 0:
            raise Exception(f"mean is nan. angle set: {predicted_angles}")

        return float(np.mean(filtered))

    def _format_lot_name(self, lot: tuple[float, float, str, float]) -> str:
        percent, synthetic_percent, lot_name, lot_percent = lot
        synthetic_suffix = f" + {synthetic_percent}%" if synthetic_percent != 0.0 else ""

        return f"{B}[{percent}%{synthetic_suffix}]{E} {G}{lot_name} ({lot_percent}%){E}"

    def _calc_mean_angle(self, length: int, angles_window: List[(int, float)]) -> Optional[float]:
        predicted_angles: List[float] = []
        for idx, angle in angles_window:
            x = idx / (self._reader.fps * length)
            y = utils.calculate_y_gsap(x)
            min_range, max_range = utils.range(length)
            predicted_spins = math.ceil((y * min_range - angle) / 360)
            predicted_angle = ((angle + predicted_spins * 360) / y) % 360
            predicted_angles.append(predicted_angle)

        try:
            mean_angle = self._filter_anomalies_linear(predicted_angles, 2.5)
            return mean_angle
        except Exception as e:
            logger.warning(f"[{length}s] fail to calculate the mean", extra={"e": e})

        return None

    def detect_winner(self):
        length = self._detect_length()
        min_length, max_length, length_candidates = length, length, [length]
        if not isinstance(length, int):
            min_length, max_length = min(length), max(length)
            length_candidates = length
            length_candidates = sorted(set(l for length in length_candidates for l in range(length - 1, length + 1)))
            logger.info(f"Speculate the possible length: {length_candidates}")

        self._populate_first_spins_with_length(max_length)

        sectors = self._init_frame.extract_sectors()
        self._circle_sectors = stream.CircleSectors(sectors)
        logger.info(f"Total lots: {len(sectors)}")

        idx = 0
        # min_range, max_range = utils.range(length)
        max_read_frames = min_length * self._reader.fps
        max_skip_frames = max(
            self._reader.fps * config.MIN_SKIP_OF_WHEEL_SPIN * max_length,
            self._reader.fps * config.MIN_SKIP_SEC
        )
        skipped_frames = 0

        buffer = self._first_spins_buffer
        # predicted_angles = {length: [] for length in length_candidates}
        # predicted_angles = {}
        angles_window: List[(int, float)] = []
        prev_mean_angle = {}

        logger.info(f"Skipping {round(max_skip_frames / self._reader.fps, 2)}s")

        # optimization
        # self._reader.skip(math.ceil(max_skip_frames / self._reader.fps))
        # skipped_frames = max_skip_frames

        while True:
            for frame in buffer:
                idx += 1
                if skipped_frames < max_skip_frames:
                    skipped_frames += 1
                    continue

                # Collect a window with angles. The size of window is config.ANGLE_WINDOW_LEN
                try:
                    if config.NASTY_OPTIMIZATION:
                        frame.force_set_wheel(self._init_frame.wheel)
                    angle = self._init_frame.calculate_rotation_with(frame)
                    angles_window.append((idx, angle,))
                except Exception as e:
                    logger.error("cannot calculate angle", extra={"frame_id": idx, "e": e})

                    angles_window = []
                    max_skip_frames += config.CALCULATION_STEP
                    continue

                if len(angles_window) <= config.ANGLE_WINDOW_LEN:
                    continue

                # Examine each length candidate and drop
                for length in length_candidates:
                    mean_angle = self._calc_mean_angle(length, angles_window)
                    if mean_angle is None:
                        continue

                    # Remove the candidate if it doesn't fit to the final angle
                    if length in prev_mean_angle:
                        diff = abs(mean_angle - prev_mean_angle[length])
                        if min(diff, 360.0 - diff) > config.MAX_MEAN_ANGLE_DELTA:
                            length_candidates.remove(length)
                            logger.error(f"Looks like the length {length}s was detected incorrectly. Remove it from the candidates.")
                            # if only one left, use it as the main length
                            if len(length_candidates) == 1:
                                max_read_frames = length_candidates[0] * self._reader.fps
                                logger.info(f"The length is: {length_candidates[0]}")
                    prev_mean_angle[length] = mean_angle

                if len(length_candidates) == 0:
                    raise Exception("no length candidates have left")

                # if only one length candidate has left use it as the main length
                if len(length_candidates) == 1:
                    # dirty hack
                    if length_candidates[0] not in prev_mean_angle:
                        continue
                    mean_angle = prev_mean_angle[length_candidates[0]]

                    self._add_lot_names_around(mean_angle, length)
                    self._circle_sectors.vote(mean_angle)
                    most_voted = self._circle_sectors.most_voted()
                    most_voted_formatted = "\n".join(
                        [f"{i+1}. {self._format_lot_name(winner)}" for i, winner in enumerate(most_voted)]
                    )

                    by_angle = self._circle_sectors.by_angle(mean_angle)
                    logger.info(
                        f"Last voted: {self._format_lot_name(by_angle)}\nMost voted:\n{most_voted_formatted}",
                        extra={
                            "sec": f"{int(idx / self._reader.fps)} ({round(idx / (self._reader.fps * length) * 100, 2)}%)",
                            "angle": f"{round(mean_angle, 2)}",
                        },
                    )

                max_skip_frames += config.CALCULATION_STEP
                angles_window = []

            if idx >= max_read_frames:
                break

            sec = math.ceil(min(max_read_frames - idx, self._reader.fps * config.READ_STEP) / self._reader.fps)
            buffer = self._reader.read(sec)

        return None
