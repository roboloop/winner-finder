import logging
import time

import colorlog

global_event_time = time.time()


class DynamicAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra_str = " ".join([f"{key}={value}" for key, value in extra.items()])
        msg = f"{msg} {extra_str}" if extra_str else msg

        return msg, kwargs


class TimeDeltaFilter(logging.Filter):
    def filter(self, record):
        # Calculate the time difference in seconds from the global event
        current_time = time.time()
        delta_time = current_time - global_event_time
        record.time_since_event = delta_time
        return True


def setup_logger(level=logging.DEBUG) -> DynamicAdapter:
    logger = logging.getLogger("logger")
    if logger.hasHandlers():
        return DynamicAdapter(logger, {})

    logger.setLevel(level)

    formatter = colorlog.ColoredFormatter(
        "%(asctime)s.%(msecs)03d (%(time_since_event).2fs ago) - %(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addFilter(TimeDeltaFilter())

    return DynamicAdapter(logger, {})


def update_global_event_time() -> None:
    global global_event_time
    global_event_time = time.time()


def add_global_event_time(sec: int) -> None:
    global global_event_time
    global_event_time += sec
