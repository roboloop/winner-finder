import argparse
from typing import List, Optional

import config
import stream


def handle_winner(filepath: Optional[str], fps: int) -> None:
    try:
        reader = stream.Reader(filepath, fps)
        segment = stream.Segment(reader)
        segment.detect_winner()
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser(description="Detect largest circle in video or image input.")
    parser.add_argument("filepath", nargs="?", help="Path to the video file")
    parser.add_argument("--fps", type=int, default=60, help="Frame per seconds")

    args = parser.parse_args()

    handle_winner(args.filepath or "pipe:0", args.fps)

if __name__ == "__main__":
    main()
