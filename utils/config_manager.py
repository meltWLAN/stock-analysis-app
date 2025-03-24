#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置管理器 - 提供统一的配置管理功能
"""

import os
import json
import yaml
import logging
import threading
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import copy

class ConfigManager:
    """统一配置管理器 - 管理应用程序所有配置"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        self.logger = logging.getLogger(__name__)
        
        # 配置数据，分层存储
        self.config: Dict[str, Any] = {
            'default': {},      # 默认配置
            'global': {},       # 全局配置
            'environment': {},  # 环境变量配置
            'file': {},         # 文件配置
            'runtime': {}       # 运行时配置
        }
        
        # 配置文件路径
        self.config_file_path = None
        
        # 配置文件修改时间
        self.config_file_mtime = 0
        
        # 自动重载配置
        self.auto_reload = False
        self.reload_interval = 60  # 秒
        self._reload_timer = None
        
        # 加载环境变量
        self._load_environment_variables()
        
        # 加载默认配置
        self._load_default_config()
        
    def _load_default_config(self):
        """加载默认配置"""
        # 数据源配置
        self.config['default'] = {
            # 通用配置
            'app': {
                'name': 'StockAnalyzer',
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            # 数据源配置
            'data_sources': {
                'tushare': {
                    'enabled': True,
                    'priority': 0,
                    'weight': 1.0,
                    'max_retry': 3,
                    'timeout': 30
                },
                'akshare': {
                    'enabled': True,
                    'priority': 1,
                    'weight': 0.8,
                    'max_retry': 3,
                    'timeout': 30
                },
                'joinquant': {
                    'enabled': True,
                    'priority': 0,
                    'weight': 1.0,
                    'max_retry': 3,
                    'timeout': 30,
                    'min_date': '2023-12-14',
                    'max_date': '2024-12-20'
                }
            },
            # 缓存配置
            'cache': {
                'enabled': True,
                'memory_size': 256,
                'disk_dir': './cache',
                'default_expiry': 3600
            },
            # 日志配置
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': './logs/app.log',
                'max_size': 10 * 1024 * 1024,  # 10MB
                'backup_count': 5
            },
            # 监控配置
            'monitoring': {
                'enabled': True,
                'metrics_port': 8000,
                'collect_interval': 60,
                'health_check_interval': 300
            },
            # 事件配置
            'events': {
                'max_workers': 5,
                'queue_size': 1000
            }
        }
    
    def _load_environment_variables(self):
        """从环境变量加载配置"""
        env_config = {}
        
        # 数据源凭据
        env_config['tushare'] = {
            'token': os.environ.get('TUSHARE_TOKEN')
        }
        
        env_config['joinquant'] = {
            'username': os.environ.get('JOINQUANT_USERNAME'),
            'password': os.environ.get('JOINQUANT_PASSWORD')
        }
        
        # 应用配置
        if 'APP_DEBUG' in os.environ:
            env_config['app'] = env_config.get('app', {})
            env_config['app']['debug'] = os.environ.get('APP_DEBUG').lower() in ('true', '1', 'yes')
            
        if 'LOG_LEVEL' in os.environ:
            env_config['logging'] = env_config.get('logging', {})
            env_config['logging']['level'] = os.environ.get('LOG_LEVEL')
        
        self.config['environment'] = env_config
        self.logger.debug("已加载环境变量配置")
    
    def load_config_file(self, file_path: Union[str, Path]):
        """
        从文件加载配置
        
        Args:
            file_path: 配置文件路径
        
        Returns:
            bool: 是否成功加载
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if not file_path.exists():
            self.logger.warning(f"配置文件不存在: {file_path}")
            return False
            
        try:
            # 记录文件路径和修改时间
            self.config_file_path = file_path
            self.config_file_mtime = os.path.getmtime(file_path)
            
            # 根据文件扩展名决定加载方式
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config['file'] = json.load(f)
            elif file_path.suffix.lower() in ('.yaml', '.yml'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config['file'] = yaml.safe_load(f)
            else:
                self.logger.error(f"不支持的配置文件格式: {file_path.suffix}")
                return False
                
            self.logger.info(f"已加载配置文件: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return False
    
    def save_config_file(self, file_path: Optional[Union[str, Path]] = None):
        """
        保存配置到文件
        
        Args:
            file_path: 配置文件路径，默认使用当前配置文件路径
            
        Returns:
            bool: 是否成功保存
        """
        if file_path is None:
            if self.config_file_path is None:
                self.logger.error("未指定配置文件路径")
                return False
            file_path = self.config_file_path
            
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        try:
            # 创建目录（如果不存在）
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 合并配置（不包括环境变量配置）
            config_to_save = copy.deepcopy(self.config['file'])
            
            # 根据文件扩展名决定保存方式
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            elif file_path.suffix.lower() in ('.yaml', '.yml'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
            else:
                self.logger.error(f"不支持的配置文件格式: {file_path.suffix}")
                return False
                
            # 更新文件修改时间
            self.config_file_mtime = os.path.getmtime(file_path)
            
            self.logger.info(f"已保存配置到文件: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            return False
    
    def reload_if_changed(self):
        """如果配置文件已更改，则重新加载"""
        if self.config_file_path is None:
            return False
            
        try:
            current_mtime = os.path.getmtime(self.config_file_path)
            if current_mtime > self.config_file_mtime:
                self.logger.info(f"检测到配置文件变更，重新加载: {self.config_file_path}")
                return self.load_config_file(self.config_file_path)
        except Exception as e:
            self.logger.error(f"检查配置文件变更失败: {e}")
            
        return False
    
    def enable_auto_reload(self, interval=60):
        """
        启用自动重载配置
        
        Args:
            interval: 检查间隔（秒）
        """
        self.auto_reload = True
        self.reload_interval = interval
        
        # 启动定时器
        def reload_timer():
            if self.auto_reload:
                self.reload_if_changed()
                # 重新设置定时器
                self._reload_timer = threading.Timer(self.reload_interval, reload_timer)
                self._reload_timer.daemon = True
                self._reload_timer.start()
                
        # 设置并启动初始定时器
        self._reload_timer = threading.Timer(self.reload_interval, reload_timer)
        self._reload_timer.daemon = True
        self._reload_timer.start()
        
        self.logger.info(f"已启用配置自动重载，间隔: {interval}秒")
    
    def disable_auto_reload(self):
        """禁用自动重载配置"""
        self.auto_reload = False
        if self._reload_timer:
            self._reload_timer.cancel()
            self._reload_timer = None
            
        self.logger.info("已禁用配置自动重载")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，使用点号分隔层级，如'app.debug'
            default: 默认值，当配置不存在时返回
            
        Returns:
            配置值或默认值
        """
        # 按优先级从高到低查找配置
        for config_level in ['runtime', 'environment', 'file', 'global', 'default']:
            value = self._get_nested_value(self.config[config_level], key, None)
            if value is not None:
                return value
                
        return default
    
    def _get_nested_value(self, config: Dict, key: str, default: Any) -> Any:
        """
        从嵌套字典获取值
        
        Args:
            config: 配置字典
            key: 键路径，如'app.debug'
            default: 默认值
            
        Returns:
            找到的值或默认值
        """
        keys = key.split('.')
        
        # 初始值为整个配置字典
        value = config
        
        # 逐层查找
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any, level: str = 'runtime') -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键，使用点号分隔层级，如'app.debug'
            value: 配置值
            level: 配置级别，可选值: 'default', 'global', 'file', 'runtime'
                  注意：'environment'级别的配置不能通过此方法设置
                  
        Returns:
            bool: 是否成功设置
        """
        if level == 'environment':
            self.logger.warning("不能直接设置环境变量配置，请使用系统环境变量")
            return False
            
        if level not in self.config:
            self.logger.error(f"无效的配置级别: {level}")
            return False
            
        # 分解键路径
        keys = key.split('.')
        
        # 获取目标配置字典
        config = self.config[level]
        
        # 逐层查找，并在需要时创建嵌套字典
        for i, k in enumerate(keys[:-1]):
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
            
        # 设置最终值
        config[keys[-1]] = value
        
        # 如果是文件级别的配置，且配置文件路径存在，则保存到文件
        if level == 'file' and self.config_file_path:
            self.save_config_file()
            
        return True
    
    def delete(self, key: str, level: str = 'runtime') -> bool:
        """
        删除配置项
        
        Args:
            key: 配置键，使用点号分隔层级，如'app.debug'
            level: 配置级别，可选值: 'default', 'global', 'file', 'runtime'
                  注意：'environment'级别的配置不能通过此方法删除
                  
        Returns:
            bool: 是否成功删除
        """
        if level == 'environment':
            self.logger.warning("不能直接删除环境变量配置")
            return False
            
        if level not in self.config:
            self.logger.error(f"无效的配置级别: {level}")
            return False
            
        # 分解键路径
        keys = key.split('.')
        
        # 获取目标配置字典
        config = self.config[level]
        
        # 逐层查找
        parent_configs = []
        parent_configs.append((None, None, config))  # (父节点, 键, 子节点)
        
        current = config
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                # 键路径不存在
                return False
            parent_configs.append((current, k, current[k]))
            current = current[k]
            
        # 检查最终键是否存在
        last_key = keys[-1]
        if last_key not in current:
            return False
            
        # 删除配置项
        del current[last_key]
        
        # 清理空字典
        for parent, key, _ in reversed(parent_configs[1:]):
            if parent[key] == {}:
                del parent[key]
            else:
                break
                
        # 如果是文件级别的配置，且配置文件路径存在，则保存到文件
        if level == 'file' and self.config_file_path:
            self.save_config_file()
            
        return True
    
    def export_config(self, include_levels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        导出配置
        
        Args:
            include_levels: 要包含的配置级别列表，默认包含所有级别
            
        Returns:
            Dict: 合并后的配置字典
        """
        if include_levels is None:
            include_levels = ['default', 'global', 'environment', 'file', 'runtime']
            
        # 从低优先级到高优先级合并配置
        result = {}
        for level in include_levels:
            if level in self.config:
                self._deep_merge(result, self.config[level])
                
        return result
    
    def _deep_merge(self, target: Dict, source: Dict):
        """
        深度合并字典
        
        Args:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # 如果两者都是字典，递归合并
                self._deep_merge(target[key], value)
            else:
                # 否则直接覆盖
                target[key] = copy.deepcopy(value)
                
    def get_app_config(self) -> Dict[str, Any]:
        """
        获取应用程序配置
        
        Returns:
            Dict: 应用程序配置字典
        """
        return self.get('app', {})
        
    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """
        获取数据源配置
        
        Args:
            source_name: 数据源名称
            
        Returns:
            Dict: 数据源配置字典
        """
        return self.get(f'data_sources.{source_name}', {})
        
    def get_cache_config(self) -> Dict[str, Any]:
        """
        获取缓存配置
        
        Returns:
            Dict: 缓存配置字典
        """
        return self.get('cache', {})
        
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            Dict: 日志配置字典
        """
        return self.get('logging', {})
        
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        获取监控配置
        
        Returns:
            Dict: 监控配置字典
        """
        return self.get('monitoring', {})
        
    def get_events_config(self) -> Dict[str, Any]:
        """
        获取事件配置
        
        Returns:
            Dict: 事件配置字典
        """
        return self.get('events', {})

# 创建默认配置管理器实例
config_manager = ConfigManager()

def get_config_manager():
    """获取默认配置管理器实例"""
    return config_manager 