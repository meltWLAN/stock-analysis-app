#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志工具模块 - 提供系统日志记录功能
"""

import os
import logging
from datetime import datetime


def setup_logger(log_level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        log_level: 日志级别，默认为INFO
    """
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件名，包含日期
    log_file = os.path.join(log_dir, f'system_{datetime.now().strftime("%Y%m%d")}.log')
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 设置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)