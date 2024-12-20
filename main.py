import argparse
import logging
from typing import List, Optional

import config
import stream
logger = config.setup_logger(level=logging.INFO)


    try:
        logger.info(f"Start handle: {filepath}")
        segment = stream.Segment(reader)
        segment.detect_winner()
    except Exception as e:
        logger.error("fail", extra={"e": e, "trace": traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser(description="Detect largest circle in video or image input.")
    parser.add_argument("filepath", nargs="?", help="Path to the video file")
    parser.add_argument("--fps", type=int, default=60, help="Frame per seconds")

    args = parser.parse_args()

    handle_winner(args.filepath or "pipe:0", args.fps)

if __name__ == "__main__":
    main()
