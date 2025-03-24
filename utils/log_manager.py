#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志管理器 - 提供统一的日志管理功能
"""

import os
import sys
import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, Union
import threading

from utils.config_manager import get_config_manager

class LogManager:
    """日志管理器 - 集中管理应用程序日志"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 日志级别映射
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """初始化日志管理器"""
        # 获取配置管理器
        self.config_manager = get_config_manager()
        
        # 获取日志配置
        self.log_config = self.config_manager.get_logging_config()
        
        # 配置根日志记录器
        self._configure_root_logger()
        
        # 存储已创建的记录器
        self.loggers = {}
        
        # 记录器是否已初始化
        self.initialized = True
    
    def _configure_root_logger(self):
        """配置根日志记录器"""
        # 获取根日志记录器
        root_logger = logging.getLogger()
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置日志级别
        level_str = self.log_config.get('level', 'INFO')
        level = self.LEVEL_MAP.get(level_str, logging.INFO)
        root_logger.setLevel(level)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 创建格式化器
        formatter = logging.Formatter(
            self.log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        console_handler.setFormatter(formatter)
        
        # 添加处理器到根日志记录器
        root_logger.addHandler(console_handler)
        
        # 如果配置了文件日志，添加文件处理器
        log_file = self.log_config.get('file')
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 配置滚动文件处理器
            max_size = self.log_config.get('max_size', 10 * 1024 * 1024)  # 默认10MB
            backup_count = self.log_config.get('backup_count', 5)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            
            # 添加到根日志记录器
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取命名日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            logging.Logger: 日志记录器
        """
        if name in self.loggers:
            return self.loggers[name]
        
        # 创建新的日志记录器
        logger = logging.getLogger(name)
        
        # 存储到字典
        self.loggers[name] = logger
        
        return logger
    
    def set_level(self, level: Union[str, int], logger_name: Optional[str] = None):
        """
        设置日志级别
        
        Args:
            level: 日志级别，可以是字符串或整数
            logger_name: 日志记录器名称，如果为None则设置根日志记录器
        """
        # 将字符串级别转换为整数级别
        if isinstance(level, str):
            level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        
        if logger_name is None:
            # 设置根日志记录器级别
            logging.getLogger().setLevel(level)
        else:
            # 设置特定日志记录器级别
            logger = self.get_logger(logger_name)
            logger.setLevel(level)
    
    def add_file_handler(self, 
                        file_path: str, 
                        level: Union[str, int] = 'INFO',
                        logger_name: Optional[str] = None,
                        max_size: int = 10 * 1024 * 1024,
                        backup_count: int = 5,
                        formatter: Optional[logging.Formatter] = None):
        """
        添加文件处理器到指定日志记录器
        
        Args:
            file_path: 日志文件路径
            level: 日志级别
            logger_name: 日志记录器名称，如果为None则使用根日志记录器
            max_size: 最大文件大小（字节）
            backup_count: 备份文件数量
            formatter: 自定义格式化器
        """
        # 将字符串级别转换为整数级别
        if isinstance(level, str):
            level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        
        # 确保日志目录存在
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 创建处理器
        handler = RotatingFileHandler(
            file_path,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(level)
        
        # 设置格式化器
        if formatter is None:
            formatter = logging.Formatter(
                self.log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        handler.setFormatter(formatter)
        
        # 获取日志记录器并添加处理器
        if logger_name is None:
            logger = logging.getLogger()
        else:
            logger = self.get_logger(logger_name)
            
        logger.addHandler(handler)
        
        return handler
    
    def add_stream_handler(self,
                          stream=sys.stdout,
                          level: Union[str, int] = 'INFO',
                          logger_name: Optional[str] = None,
                          formatter: Optional[logging.Formatter] = None):
        """
        添加流处理器到指定日志记录器
        
        Args:
            stream: 输出流，默认为标准输出
            level: 日志级别
            logger_name: 日志记录器名称，如果为None则使用根日志记录器
            formatter: 自定义格式化器
        """
        # 将字符串级别转换为整数级别
        if isinstance(level, str):
            level = self.LEVEL_MAP.get(level.upper(), logging.INFO)
        
        # 创建处理器
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        
        # 设置格式化器
        if formatter is None:
            formatter = logging.Formatter(
                self.log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        handler.setFormatter(formatter)
        
        # 获取日志记录器并添加处理器
        if logger_name is None:
            logger = logging.getLogger()
        else:
            logger = self.get_logger(logger_name)
            
        logger.addHandler(handler)
        
        return handler
    
    def remove_handlers(self, logger_name: Optional[str] = None):
        """
        移除日志记录器的所有处理器
        
        Args:
            logger_name: 日志记录器名称，如果为None则使用根日志记录器
        """
        if logger_name is None:
            logger = logging.getLogger()
        else:
            logger = self.get_logger(logger_name)
            
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    def reload_config(self):
        """重新加载日志配置"""
        # 重新获取日志配置
        self.log_config = self.config_manager.get_logging_config()
        
        # 重新配置根日志记录器
        self._configure_root_logger()
        
    def create_class_logger(self, cls) -> logging.Logger:
        """
        创建类专用的日志记录器
        
        Args:
            cls: 类对象
            
        Returns:
            logging.Logger: 日志记录器
        """
        return self.get_logger(cls.__module__ + '.' + cls.__name__)
        
    def log_method_call(self, logger, level=logging.DEBUG):
        """
        创建方法调用装饰器
        
        Args:
            logger: 日志记录器或日志记录器名称
            level: 日志级别
            
        Returns:
            函数装饰器
        """
        # 如果传入的是字符串，获取对应的日志记录器
        if isinstance(logger, str):
            logger = self.get_logger(logger)
            
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 记录方法调用
                arg_str = ', '.join([repr(a) for a in args[1:]] if len(args) > 0 and hasattr(args[0], '__class__') else [repr(a) for a in args])
                kwarg_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                params = []
                if arg_str:
                    params.append(arg_str)
                if kwarg_str:
                    params.append(kwarg_str)
                    
                param_str = ', '.join(params)
                logger.log(level, f"调用 {func.__name__}({param_str})")
                
                # 执行原方法
                result = func(*args, **kwargs)
                
                # 记录返回值
                logger.log(level, f"{func.__name__} 返回: {repr(result)}")
                
                return result
            
            # 保留原函数的元数据
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            wrapper.__module__ = func.__module__
            
            return wrapper
        
        return decorator

# 创建默认日志管理器实例
log_manager = LogManager()

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器的快捷方式
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    return log_manager.get_logger(name)

def get_log_manager() -> LogManager:
    """获取日志管理器实例"""
    return log_manager 