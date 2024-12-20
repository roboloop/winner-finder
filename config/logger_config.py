import logging

import colorlog

def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger("logger")
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

    return logger
