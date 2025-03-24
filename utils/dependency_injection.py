#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
依赖注入容器 - 管理系统组件依赖
"""

from dependency_injector import containers, providers
from utils.data_validation import DataValidator
from utils.config_manager import ConfigManager
from utils.cache_manager import CacheManager
from utils.logging_service import LoggingService
from utils.monitoring_service import MonitoringService
from data.data_source_manager import (
    DataSourceManager, 
    TushareDataSource, 
    AkshareDataSource, 
    JoinQuantDataSource
)

class Container(containers.DeclarativeContainer):
    """依赖注入容器"""
    
    # 配置
    config = providers.Singleton(ConfigManager)
    
    # 工具服务
    logging_service = providers.Singleton(LoggingService, config=config)
    monitoring_service = providers.Singleton(MonitoringService, config=config)
    cache_manager = providers.Singleton(CacheManager, config=config)
    data_validator = providers.Singleton(DataValidator)
    
    # 数据源
    tushare_data_source = providers.Factory(
        TushareDataSource,
        token=config.provided.get.tushare.token,
        validator=data_validator,
        logger=logging_service.provided.get_logger,
        cache=cache_manager
    )
    
    akshare_data_source = providers.Factory(
        AkshareDataSource,
        priority=config.provided.get.akshare.priority,
        validator=data_validator,
        logger=logging_service.provided.get_logger,
        cache=cache_manager
    )
    
    joinquant_data_source = providers.Factory(
        JoinQuantDataSource,
        username=config.provided.get.joinquant.username,
        password=config.provided.get.joinquant.password,
        validator=data_validator,
        logger=logging_service.provided.get_logger,
        cache=cache_manager
    )
    
    # 数据源管理器
    data_source_manager = providers.Singleton(
        DataSourceManager,
        logger=logging_service.provided.get_logger,
        monitoring_service=monitoring_service
    )

# 创建全局容器实例
container = Container()

def get_container():
    """获取全局容器实例"""
    return container 