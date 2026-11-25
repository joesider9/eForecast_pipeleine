# eForecast project author: George Sideratos
# License for eForecast contributions: MIT
# Header date: 2026-11-25

import logging
import os
from contextlib import contextmanager

QUIET = ('eforecast.datasets', 'eforecast.init')


@contextmanager
def quieted(level, names=QUIET):
    """Quieten the noisy library loggers for one command, then put them back.

    These entry points run in-process as well as from a shell: the managed stage path, the
    inference host and the test suite all call them. A level set for good costs every later
    caller in that process its diagnostics, with nothing in the log to say why they stopped.
    """
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')
    previous = {name: logging.getLogger(name).level for name in names}
    for name in names:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for name, value in previous.items():
            logging.getLogger(name).setLevel(value)


def create_logger(logger_name, abs_path, logger_path, write_type='w'):
    assert write_type in ('w', 'a')
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(os.path.join(abs_path, logger_path), mode=write_type, encoding="utf-8")
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
