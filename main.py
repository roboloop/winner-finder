import argparse
import logging
from typing import List, Optional

import config
import stream
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


def main():
    parser = argparse.ArgumentParser(description="Detect largest circle in video or image input.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands help")

    winner_parser = subparsers.add_parser("winner", help="Find winner")
    winner_parser.add_argument("filepath", nargs="?", help="Path to the video file")
    winner_parser.add_argument("--fps", type=int, default=60, help="Frame per seconds")
    winner_parser.add_argument("--skip-sec", type=int, default=0, help="Skip seconds")
    winner_parser.add_argument("--visualize", action="store_true", help="Visualize all steps (for debugging purposes)")

    args = parser.parse_args()

    if args.command == "winner":
        handle_winner(args.filepath or "pipe:0", args.fps, args.skip_sec, args.visualize)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
