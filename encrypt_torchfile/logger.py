import logging
import os

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING


class Logger():
    def __init__(self):
        self._logger = logging.getLogger("encrypt_torchfile")
        self._logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)