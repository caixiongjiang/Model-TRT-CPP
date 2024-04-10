import logging
import os
from logging.handlers import RotatingFileHandler

def make_log(log):
    """
    创建日志
    log: 配置文件中转化来的log字典
    """
    # 控制台日志文件的级别
    if log["LOG_LEVEL"] == "DEBUG":
        log_level = logging.DEBUG
    elif log["LOG_LEVEL"] == "INFO":
        log_level = logging.INFO
    elif log["LOG_LEVEL"] == "WARN":
        log_level = logging.WARN
    elif log["LOG_LEVEL"] == "ERROR":
        log_level = logging.ERROR
    elif log["LOG_LEVEL"] == "CRITICAL":
        log_level = logging.CRITICAL

    # 日志文件的文件夹
    log_folder = log["LOG_FOLDER"]
    os.makedirs(log_folder, exist_ok=True)

    # 日志文件的名字
    log_name = log["LOG_FILENAME"]
    
    # 日志留存数
    levelSave_num = log["BACKUPCOUNT"]

    # 日志路径
    log_filename = os.path.join(log_folder, f'{log_name}')
    
    # 配置日志
    logging.basicConfig(filename=log_filename, level=log_level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # 获取一个Logger实例
    logger = logging.getLogger(__name__)

    # 创建一个RotatingFileHandler来按大小轮转日志文件，文件大小最大为1MB，最多保留5个日志文件
    file_handler = RotatingFileHandler(log_filename, maxBytes=1*1024*1024, backupCount=5)
    if levelSave_num == 5:
        file_handler.setLevel(logging.DEBUG)
    elif levelSave_num == 4:
        file_handler.setLevel(logging.INFO)
    elif levelSave_num == 3:
        file_handler.setLevel(logging.WARN)
    elif levelSave_num == 2:
        file_handler.setLevel(logging.ERROR)
    elif levelSave_num == 1:
        file_handler.setLevel(logging.CRITICAL)

    # 创建一个控制台处理程序日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # 将控制台处理程序添加到Logger实例
    logger.addHandler(console_handler)
    
    return logger