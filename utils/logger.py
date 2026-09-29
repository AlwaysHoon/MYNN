import logging
import os

def setup_logger(save_path):
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "train.log")
    
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s", datefmt='%d-%b-%y %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
