import os
import time
import logging
from common.distributed import is_master, synchronize

def create_logger(args):
    
    logdir = os.path.join(args.exp_path, 'log')
    if not os.path.exists(logdir) and is_master():
        os.mkdir(logdir)
    synchronize()

    time_fmt = time.strftime('%Y-%m-%d-%H-%M-%S')
    log_file = '{}.rank.{}.log'.format(time_fmt, args.rank)
    log_name = os.path.join(logdir, log_file)
    message_fmt = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_name, format=message_fmt)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger('').addHandler(handler)

    return logger