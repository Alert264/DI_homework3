import logging
import time
from colorama import Fore, Style

logfile = 'log.log'
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
formatter = logging.Formatter("[%(asctime)s]: %(message)s")
fh.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

save = True


def log(text):
    print(Fore.RED, end="")
    print("[%s]:" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print(Style.RESET_ALL, end="")
    print(text)
    if save:
        logger.debug(text)


if __name__ == '__main__':
    log("123")
