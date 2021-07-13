import logging
from datetime import datetime
import os
from utils.common import BASE_LOGS

def get_logger():
    # Login instance
    logname = datetime.now().strftime("%d-%m-%Y_%H_%M_%S") + "_run.log"
    logging.basicConfig(filename=os.path.join(BASE_LOGS, logname), level=logging.INFO,
                        format='[%(levelname)s] (%(asctime)s) : %(message)s')
    return logging
