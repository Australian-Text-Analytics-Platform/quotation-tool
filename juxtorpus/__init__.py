import logging.config
import colorlog
from pathlib import Path

conf_path = Path("logging_conf.ini")
logging.config.fileConfig(conf_path)
rlogger = colorlog.getLogger()

# ARCHIVED: FileConfig unable to read key: log_colors
# Comment above and uncomment below if you want to change the log colours.
# import colorlog
# from colorlog import ColoredFormatter
#
# handler = colorlog.StreamHandler()
# logger = colorlog.getLogger()   # root logger
#
# formatter = ColoredFormatter(
#     "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
#     datefmt=None,
#     reset=True,
#     log_colors={
#         'DEBUG': 'cyan',
#         'INFO': 'green',
#         'WARNING': 'yellow',
#         'ERROR': 'red',
#         'CRITICAL': 'red,bg_white',
#     },
#     secondary_log_colors={},
#     style='%'
# )
#
# handler.setFormatter(formatter)
# logger.addHandler(handler)


from .jux import Jux