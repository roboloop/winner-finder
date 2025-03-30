from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import queue
from typing import Dict, List, Optional

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
        self._initial_frame: Optional[stream.Frame] = None
        self._circle_sectors: Optional[stream.CircleSectors] = None

    def _binary_search(self, segment: List[stream.Frame]) -> Optional[int]:
        low = 0
        high = len(segment) - 1

        while low <= high:
            mid = (low + high) // 2
            if segment[mid].is_initial_frame():
                if mid == len(segment) - 1 or segment[mid + 1].is_spin_frame():
                    return mid
                low = mid + 1
            else:
                high = mid - 1

        return None

    def _detect_duration(self) -> int | List[int]:
        self._populate_first_spins()

        try:
            duration = self._first_spins_buffer[0].detect_duration()
            logger.info(f"Duration detected via screen parsing: {duration}")
            return duration
        except Exception as e:
            logger.error(f"Duration not detected via screen parsing: {e}")

        try:
            duration = self._populate_first_n_spins()
            logger.info(f"Duration detected via wheel spin: {duration}")
            return duration
        except Exception as e:
            logger.error(f"Duration not detected via wheel spin: {e}")

        if config.PROMPT_FOR_DURATION:
            # Manual enter
            with open("/dev/tty", "r+") as tty:
                tty.write("Cannot detect duration. Manual enter: ")
                tty.flush()
                response = tty.readline().strip()
                tty.write("")

            if response.isdigit():
                duration = int(response)
                logger.info(f"Duration set to {duration}")

                return duration

        raise Exception("Cannot detect the duration")

    def _populate_first_spins(self) -> None:
        if len(self._first_spins_buffer) != 0:
            return

        buffer = self._reader.read_until_spin_found(config.READ_STEP)
        logger.info("Spin with the lot name found")

        _initial_frame_index = self._binary_search(buffer)
        if _initial_frame_index is None:
            raise Exception("No initial frame found")
        self._first_spins_buffer = buffer[_initial_frame_index + 1 :]
        self._initial_frame = buffer[_initial_frame_index]
        logger.info("Initial frame found", extra={"frame": {self._initial_frame.index}})
        if self._initial_frame.is_circle():
            logger.error("Circle is an ellipse. Result may vary")

        # TODO: why was it added?
        self._initial_frame.wheel[2] -= 1

    def _populate_first_spins_with_duration(self, duration: int) -> None:
        if len(self._first_spins_buffer) == 0:
            self._populate_first_spins()

        min_range, max_range = utils.range(duration)
        first_wheel_frames = math.ceil(
            utils.calculate_x(config.SPIN_BUFFER_SIZE * 360.0 / min_range) * self._reader.fps * duration
        )
        sec = math.ceil(max(first_wheel_frames - len(self._first_spins_buffer), 0) / self._reader.fps)
        self._first_spins_buffer.extend(self._reader.read(sec))

    def _populate_first_n_spins(self) -> int:
        def _find_new_spin_frame(part: List[stream.Frame], reverse: bool) -> int:
            iteration = range(1, len(part)) if reverse is False else range(len(part) - 1, 0, -1)
            for i in iteration:
                angle = self._initial_frame.calculate_rotation_with(part[i])
                prev_angle = self._initial_frame.calculate_rotation_with(part[i - 1])
                if angle < prev_angle and abs(360.0 + angle - prev_angle) % 360.0 < 60.0:
                    return i

            raise Exception("No new spins")

        def _exclude_not_matched_seconds(buffer: List[stream.Frame], index: int, idx: int, seconds: List[int]):
            part_behind = buffer[max(0, index - config.DURATION_DETECTION_FRAME_STEP) : index + 1]
            new_spin_index = _find_new_spin_frame(part_behind, True)
            new_spin_idx = idx - (len(part_behind) - new_spin_index - 1)

            to_remove = []
            for sec in seconds:
                if spins > round(sec * 270 / 360):
                    logger.error("No point to look further", extra={"sec": sec, "spins": spins})
                    continue

                min_range, max_range = utils.range(sec)
                y_min = spins * 360.0 / min_range
                x_min = utils.calculate_x_gsap(y_min)
                idx_min = x_min * (60 * sec)

                y_max = spins * 360.0 / max_range
                x_max = utils.calculate_x_gsap(y_max)
                idx_max = x_max * (60 * sec)

                if not (math.floor(idx_max) <= float(new_spin_idx) <= math.ceil(idx_min)):
                    to_remove.append(sec)

            for sec in to_remove:
                seconds.remove(sec)

        if len(self._first_spins_buffer) == 0:
            self._populate_first_spins()

        idx = 0
        spins = 0
        prev_angle = 0.0
        starts, ends = config.DURATION_DETECTION_TIME_RANGE
        seconds = np.arange(starts, ends + 1).tolist()

        for index, frame in enumerate(self._first_spins_buffer):
            if len(self._first_spins_buffer) - 1 == index:
                self._first_spins_buffer.extend(self._reader.read(config.READ_STEP))

            idx += 1
            if idx % config.DURATION_DETECTION_FRAME_STEP != 0:
                continue

            try:
                if config.SPECULATIVE_OPTIMIZATION:
                    frame.force_set_wheel(self._initial_frame.wheel)
                # TODO: could calculate the angle incorrectly
                angle = self._initial_frame.calculate_rotation_with(frame)
            except Exception as e:
                logger.error("Cannot calculate angle", extra={"frame_id": idx, "e": e})
                continue

            # diff = abs(angle - prev_angle)
            # if angle < prev_angle and min(diff, 360.0 - diff) < 60.0:
            if angle < prev_angle:
                spins += 1
                _exclude_not_matched_seconds(self._first_spins_buffer, index, idx, seconds)
                if spins > config.DURATION_DETECTION_MAX_SPINS or idx > config.DURATION_DETECTION_MAX_FRAMES:
                    break
                if len(seconds) == 1:
                    logger.info("One duration candidate left")
                    break
            prev_angle = angle

        if len(seconds) == 0:
            raise Exception("No duration candidates")

        seconds.sort()

        return seconds

    def _add_lot_names_around(self, angle: float, duration: int) -> None:
        min_range, max_range = utils.range(duration)
        angles = [angle + spins * 360.0 for spins in range(0, config.SPIN_BUFFER_SIZE)]

        task_queue = queue.Queue()

        for angle in angles:
            start_frame_id = math.floor(utils.calculate_x_gsap(angle / max_range) * self._reader.fps * duration)
            end_frame_id = math.ceil(utils.calculate_x_gsap(angle / min_range) * self._reader.fps * duration)

            for frame_id in range(start_frame_id, end_frame_id):
                frame = self._first_spins_buffer[frame_id]
                task_queue.put(frame)

        def _worker(task_queue: queue.Queue) -> None:
            while not task_queue.empty():
                frame = task_queue.get()
                lot_name = frame.detect_lot_name()
                frame_angle = self._initial_frame.calculate_rotation_with(frame)
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
            raise Exception(f"Mean is NaN. Angle set: {predicted_angles}")

        return float(np.mean(filtered))

    def _format_lot_name(self, lot: tuple[float, float, str, float]) -> str:
        percent, synthetic_percent, lot_name, lot_percent = lot
        synthetic_suffix = f" + {synthetic_percent}%" if synthetic_percent != 0.0 else ""

        return f"{B}[{percent}%{synthetic_suffix}]{E} {G}{lot_name} ({lot_percent}%){E}"

    def _calc_mean_angle(self, duration: int, angles_window: List[(int, float)]) -> Optional[float]:
        predicted_angles: List[float] = []
        for idx, angle in angles_window:
            x = idx / (self._reader.fps * duration)
            y = utils.calculate_y_gsap(x)
            min_range, max_range = utils.range(duration)
            predicted_spins = math.ceil((y * min_range - angle) / 360)
            predicted_angle = ((angle + predicted_spins * 360) / y) % 360
            predicted_angles.append(predicted_angle)

        try:
            mean_angle = self._filter_anomalies_linear(predicted_angles, 2.5)
            return mean_angle
        except Exception as e:
            logger.warning(f"[{duration}s] mean calculation failed", extra={"e": e})

        return None

    def detect_winner(self):
        duration = self._detect_duration()
        min_duration, max_duration, duration_candidates = duration, duration, [duration]
        if not isinstance(duration, int):
            min_duration, max_duration = min(duration), max(duration)
            duration_candidates = duration
            duration_candidates = sorted(set(l for d in duration_candidates for l in range(d - 1, d + 1)))
            logger.info(f"Speculate possible duration: {duration_candidates}")

        self._populate_first_spins_with_duration(max_duration)

        sectors = self._initial_frame.extract_sectors()
        self._circle_sectors = stream.CircleSectors(sectors)
        logger.info(f"Total lots: {len(sectors)}")

        idx = 0
        max_read_frames = min_duration * self._reader.fps
        max_skip_frames = max(
            self._reader.fps * config.MIN_WHEEL_SPIN_SKIP_RATIO * max_duration,
            self._reader.fps * config.MIN_SKIP_DURATION,
        )
        skipped_frames = 0

        buffer = self._first_spins_buffer
        angles_window: List[(int, float)] = []
        prev_mean_angle = {}

        logger.info(f"Skipping {round(max_skip_frames / self._reader.fps, 2)}s")

        while True:
            for frame in buffer:
                idx += 1
                if skipped_frames < max_skip_frames:
                    skipped_frames += 1
                    continue

                # Collect the window of angles
                try:
                    if config.SPECULATIVE_OPTIMIZATION:
                        frame.force_set_wheel(self._initial_frame.wheel)
                    angle = self._initial_frame.calculate_rotation_with(frame)
                    angles_window.append((idx, angle))
                except Exception as e:
                    logger.error("Cannot calculate angle", extra={"frame_id": idx, "e": e})

                    angles_window = []
                    max_skip_frames += config.CALCULATION_STEP
                    continue

                if len(angles_window) <= config.ANGLE_WINDOW_SIZE:
                    continue

                # Examine each duration candidate separatly
                for duration in duration_candidates:
                    mean_angle = self._calc_mean_angle(duration, angles_window)
                    if mean_angle is None:
                        continue

                    # Remove the candidate if it doesn't match to the target angle
                    if duration in prev_mean_angle:
                        diff = abs(mean_angle - prev_mean_angle[duration])
                        if min(diff, 360.0 - diff) > config.MAX_ANGLE_DELTA:
                            duration_candidates.remove(duration)
                            logger.error(f"Duration {duration}s detected incorrectly. Remove it from the candidates")
                            if len(duration_candidates) == 1:
                                max_read_frames = duration_candidates[0] * self._reader.fps
                                logger.info(f"Duration is: {duration_candidates[0]}")
                    prev_mean_angle[duration] = mean_angle

                if len(duration_candidates) == 0:
                    raise Exception("No duration candidates left")

                # If only one duration candidate is left, use it as the main duration
                if len(duration_candidates) == 1:
                    # hack
                    if duration_candidates[0] not in prev_mean_angle:
                        continue
                    mean_angle = prev_mean_angle[duration_candidates[0]]

                    self._add_lot_names_around(mean_angle, duration)
                    self._circle_sectors.vote(mean_angle)
                    most_voted = self._circle_sectors.most_voted()
                    most_voted_formatted = "\n".join(
                        [f"{i+1}. {self._format_lot_name(winner)}" for i, winner in enumerate(most_voted)]
                    )

                    by_angle = self._circle_sectors.by_angle(mean_angle)
                    logger.info(
                        f"Last voted: {self._format_lot_name(by_angle)}\nMost voted:\n{most_voted_formatted}",
                        extra={
                            "sec": f"{int(idx / self._reader.fps)} ({round(idx / (self._reader.fps * duration) * 100, 2)}%)",
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
