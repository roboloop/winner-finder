import argparse
import json
import logging
import math
import os
import queue
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

import config
import stream
import utils

logger = config.setup_logger(level=logging.INFO)


def handle_winner(filepath: Optional[str], fps: int, skip_sec: int, visualize: bool) -> None:
    try:
        logger.info(f"Start handle: {filepath}")

        if visualize:
            config.VISUALIZATION_ENABLED = True
            logger.info(f"Visualization is enabled")

        reader = stream.Reader(filepath, fps, skip_sec)
        segment = stream.Segment(reader)

        # The segment is found and ready. Let's find a winner
        segment.detect_winner()
    except Exception as e:
        logger.error("fail", extra={"e": e, "trace": traceback.format_exc()})


def handle_all_assets() -> None:
    def _worker(task_queue):
        while not task_queue.empty():
            func, args = task_queue.get()
            func(*args)
            task_queue.task_done()

    task_queue = queue.Queue()
    with open("input.json", "r") as file:
        data = json.load(file)
        for one in data:
            if not one["end_spin_frame"]:
                logger.warning(f"Skipped {one['filepath']}")
                continue
            logger.info(f"Adding to queue {one['filepath']}")
            task_queue.put((handle_winner, (one["filepath"], 60, math.floor(one["init_frame"] / 60))))

    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            executor.submit(_worker, task_queue)

    task_queue.join()
    print("All tasks completed.")


def handle_visualize_measure() -> None:
    def _filter_anomalies_linear(x: List[float], y: List[float], window_size: int, deviation: float):
        filtered_x = []
        filtered_y = []

        for i in range(0, len(y) - window_size + 1, window_size):
            window_y = y[i : i + window_size]
            window_x = x[i : i + window_size]

            filtered = []
            mean = np.mean(window_y)
            deviations = np.abs(np.array(window_y) - mean)
            threshold = np.mean(deviations)
            for j in range(len(window_y)):
                if deviations[j] <= threshold and abs(window_y[j] - mean) <= deviation:
                    filtered.append(window_y[j])

            mean = float(np.mean(filtered))

            filtered_x.append(window_x[-1])
            filtered_y.append(mean)

        return filtered_x, filtered_y

    filepaths = []
    for root, dirs, files in os.walk("measure"):
        for file in files:
            if file.endswith(".json"):
                filepaths.append(os.path.join(root, file))
    parsed_blocks = {}
    for filepath in filepaths:
        with open(filepath, "r") as file:
            key = re.sub(r".*/|\.([^/\\]+)$", "", filepath)
            parsed_blocks[key] = json.load(file)

    with open("input.json") as file:
        input = {re.sub(r".*/|\.([^/\\]+)$", "", obj["filepath"]): obj for obj in json.load(file)}

    for key in parsed_blocks:
        if key != "c2":
            continue
        b = parsed_blocks[key]
        # TODO: Good code
        filtered_x, filtered_y = _filter_anomalies_linear(b["x"], b["predicted_angle"], 60, 2.5)

        i = input[key]
        plt.figure(figsize=(12, 6))
        plt.plot(b["x"], b["predicted_angle"], ".", label="Predicted angle")
        plt.plot(filtered_x, filtered_y, ".", label="Filtered predicted angle")
        # plt.plot(b['x'], b['angle'], '.', label='Angle')
        # plt.plot(filtered_x, filtered_angle, '.', label='Filtered angle')
        plt.title(f"{i['filepath']} — {i['length']}s")
        plt.xlabel("x")
        plt.ylabel("angle")
        plt.legend()
        plt.grid()
        plt.show(block=False)
    plt.show()


def handle_spin_frames() -> None:
    min_sec = 30
    max_sec = 180
    print('spin,' + ','.join(f'{num}' for num in range(min_sec, max_sec)))
    for spins in range(1, round(max_sec * 270 / 360 / 2)):
        path = spins * 360.0
        data = []
        data.append(spins)
        for sec in range(min_sec, max_sec):
            if spins > round(sec * 270 / 360):
                data.append('')
                continue

            min_range, max_range = utils.range(sec)
            y = path / min_range
            x = utils.calculate_x_gsap(y)
            idx = x * (60 * sec)
            data.append(round(idx, 2))

        print(','.join(f'{num}' for num in data))

def main():
    parser = argparse.ArgumentParser(description="Detect largest circle in video or image input.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands help")

    winner_parser = subparsers.add_parser("winner", help="Find winner")
    winner_parser.add_argument("filepath", nargs="?", help="Path to the video file")
    winner_parser.add_argument("--fps", type=int, default=60, help="Frame per seconds")
    winner_parser.add_argument("--skip-sec", type=int, default=0, help="Skip seconds")
    winner_parser.add_argument("--visualize", action="store_true", help="Visualize all steps (for debugging purposes)")

    utils_parser = subparsers.add_parser("utils", help="utils")
    utils_parser.add_argument("sub", help="sub command")

    args = parser.parse_args()

    if args.command == "winner":
        handle_winner(args.filepath or "pipe:0", args.fps, args.skip_sec, args.visualize)
        return

    if args.command == "utils":
        if args.sub == "handle_all_assets":
            handle_all_assets()
            return
        if args.sub == "handle_visualize_measure":
            handle_visualize_measure()
            return

        if args.sub == "handle_spin_frames":
            handle_spin_frames()
            return

    parser.print_help()


if __name__ == "__main__":
    main()
