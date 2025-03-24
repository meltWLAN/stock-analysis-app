#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()

"""
至简交易选股系统 - 主程序入口
基于简放交易理念 + AI + 量化因子的股票筛选系统
"""

import os
import sys
import logging
from PyQt5.QtWidgets import QApplication

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from ui.main_window import MainWindow
from utils.cache_manager import CacheManager
from data.data_source_manager import DataSourceManager
from data.data_pipeline import create_default_pipeline
from data.data_processor import EnhancedDataProcessor
from data.data_validator import FinancialDataValidator

# 配置日志系统
def setup_logger():
    """配置日志系统"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, 'stock_selection.log')
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    return root_logger

def initialize_system():
    """初始化系统组件"""
    logger = logging.getLogger(__name__)
    logger.info("初始化系统组件...")
    
    # Debug imports
    logger.info("Debug: Checking imports...")
    try:
        import os as os_debug
        logger.info(f"Debug: os module imported successfully as os_debug. Path: {os_debug.__file__}")
    except Exception as e:
        logger.error(f"Debug: Failed to import os module: {e}")
    
    # 初始化缓存管理器
    try:
        logger.info("Debug: Starting cache manager initialization...")
        import os  # Import os directly within the function to ensure it's available
        logger.info(f"Debug: os module imported successfully. Path: {os.__file__}")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        logger.info(f"Debug: Cache directory path: {cache_dir}")
        # 创建缓存目录
        if not os.path.exists(cache_dir):
            logger.info(f"Debug: Creating cache directory: {cache_dir}")
            os.makedirs(cache_dir)
        cache_manager = CacheManager.get_instance(cache_dir=cache_dir)
        logger.info("缓存管理器初始化完成")
    except Exception as e:
        logger.error(f"Debug: Cache manager initialization failed: {e}")
        raise
    
    # 初始化数据源管理器
    data_source_manager = DataSourceManager.get_instance()
    data_source_manager.init_default_sources()
    
    # 启动数据源健康监控
    data_source_manager.start_health_monitor(interval=1800)  # 每30分钟检查一次
    
    # 创建并注册默认数据处理流水线
    default_pipeline = create_default_pipeline()
    logger.info(f"默认数据处理流水线'{default_pipeline.name}'初始化完成，包含{len(default_pipeline.stages)}个处理阶段")
    
    # 初始化增强版数据处理器
    enhanced_processor = EnhancedDataProcessor()
    logger.info("增强版数据处理器初始化完成")
    
    # 初始化金融数据验证器
    data_validator = FinancialDataValidator()
    logger.info("金融数据验证器初始化完成")
    
    return {
        'cache_manager': cache_manager,
        'data_source_manager': data_source_manager,
        'default_pipeline': default_pipeline,
        'enhanced_processor': enhanced_processor,
        'data_validator': data_validator
    }

def main():
    """主程序入口函数"""
    # 设置日志
    logger = setup_logger()
    logger.info("股票筛选系统启动...")
    
    # 初始化系统组件
    try:
        system_components = initialize_system()
        logger.info("系统组件初始化完成")
    except Exception as e:
        logger.error(f"系统组件初始化失败: {e}")
        sys.exit(1)
    
    # 创建QT应用
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_window = MainWindow(system_components)
    main_window.show()
    
    # 启动事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()