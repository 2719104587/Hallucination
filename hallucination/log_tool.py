import logging
import sys

import colorlog


def get_log_stdout_handler() -> logging.StreamHandler:
    """
    Provide a logger handler template

    .. code-block:: python

        >>> # setup logger
        >>> logger = logging.getLogger("af2model")
        >>> logger.propagate = False
        >>> logger.handlers = [get_log_stdout_handler()]
        >>> logger.level = logging.INFO

    Returns:
        logging.StreamHandler

    """
    h = logging.StreamHandler()
    h.setStream(sys.stdout)
    h.setLevel(logging.DEBUG)
    log_colors = {
        "DEBUG": "thin_cyan",
        "INFO": "green",
        "WARNING": "bold_yellow",
        "ERROR": "bold_red",
        "CRITICAL": "bg_white,bold_red",
    }

    fmt = "%(levelname)s:%(asctime)s %(name)s(%(process)d) %(filename)s:%(lineno)d %(funcName)s - %(message)s"
    fmt = (
        fmt.replace("%(asctime)s", "%(green)s%(asctime)s%(reset)s")
        .replace("%(name)s", "%(blue)s%(name)s%(reset)s")
        .replace("%(levelname)s", "%(log_color)s%(levelname)-8s%(reset)s")
        .replace("%(filename)s:%(lineno)d", "%(cyan)s%(filename)s:%(lineno)d%(reset)s")
    )

    h.setFormatter(colorlog.ColoredFormatter(fmt, log_colors=log_colors))
    return h


def get_logger(
    logger_name: str = "hallucination", loglevel: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.handlers = [get_log_stdout_handler()]
    logger.addHandler(get_error_log_file_hdlr())
    logger.setLevel(loglevel)
    return logger


def get_error_log_file_hdlr():
    hdl = logging.FileHandler("error.log", mode="a+", encoding="utf-8")
    fmt = "%(levelname)s:%(asctime)s %(name)s(%(process)d) %(filename)s:%(lineno)d %(funcName)s - %(message)s"
    hdl.setFormatter(logging.Formatter(fmt))
    hdl.setLevel(logging.ERROR)
    return hdl


def delete_logger(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    logging.Logger.manager.loggerDict.pop(logger_name)


def setup_logger():
    logger = get_logger("hallucination", loglevel=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    return logger