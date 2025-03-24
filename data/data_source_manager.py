#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源管理模块 - 提供多数据源集成与管理功能
"""

import os
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
from functools import lru_cache
import concurrent.futures
from threading import Lock

# 导入自定义工具模块
from utils.data_validation import DataValidator
from .data_validator import FinancialDataValidator
from .data_pipeline import DataPipeline, validate_price_data, validate_financial_data

logger = logging.getLogger(__name__)

class DataSource:
    """数据源基类"""
    
    def __init__(self, name, priority=0, weight=1.0):
        """
        初始化数据源
        
        Args:
            name: 数据源名称
            priority: 优先级，数字越小优先级越高
            weight: 在数据融合中的权重
        """
        self.name = name
        self.priority = priority
        self.weight = weight
        self.status = "initialized"
        self.health_score = 1.0  # 健康分数，1.0代表完全健康
        self.error_count = 0
        self.total_requests = 0
        self.success_requests = 0
        self.last_error_time = None
        self.last_success_time = None
        self.average_response_time = 0
        self.validator = DataValidator()
        self.financial_validator = FinancialDataValidator()  # 添加新的验证器
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    def get_data(self, **kwargs):
        """
        获取数据抽象方法，子类必须实现
        
        Returns:
            pd.DataFrame: 获取的数据
        """
        raise NotImplementedError("子类必须实现get_data方法")
        
    def health_check(self):
        """
        检查数据源健康状况
        
        Returns:
            bool: 是否健康
        """
        # 默认的健康检查逻辑
        if self.total_requests == 0:
            return True
            
        # 计算成功率
        success_rate = self.success_requests / max(1, self.total_requests)
        
        # 计算健康分数
        self.health_score = success_rate * (0.9 ** self.error_count)
        
        # 如果最近有成功请求，增加健康分数
        if self.last_success_time and datetime.now() - self.last_success_time < timedelta(hours=1):
            self.health_score = min(1.0, self.health_score + 0.1)
            
        # 如果响应时间过长，降低健康分数
        if self.average_response_time > 5.0:  # 假设5秒是阈值
            self.health_score *= 0.9
            
        return self.health_score > 0.5  # 健康分数大于0.5认为是健康的
        
    def record_request(self, success, response_time):
        """
        记录请求结果
        
        Args:
            success: 请求是否成功
            response_time: 响应时间(秒)
        """
        self.total_requests += 1
        
        if success:
            self.success_requests += 1
            self.last_success_time = datetime.now()
            
            # 更新平均响应时间 (指数移动平均)
            alpha = 0.2  # 权重因子
            if self.average_response_time == 0:
                self.average_response_time = response_time
            else:
                self.average_response_time = (1 - alpha) * self.average_response_time + alpha * response_time
        else:
            self.error_count += 1
            self.last_error_time = datetime.now()
            
        # 自动恢复逻辑
        if self.status == "down" and success:
            self.status = "up"
            self.logger.info(f"数据源 {self.name} 已恢复正常")
            
        # 错误过多导致数据源下线
        if self.error_count > 5 and self.total_requests > 0 and success_rate < 0.3:
            self.status = "down"
            self.logger.warning(f"数据源 {self.name} 由于错误过多已暂时下线")
            
    def get_data_safe(self, **kwargs):
        """
        安全地获取数据，包含错误处理和统计
        
        Returns:
            pd.DataFrame: 获取的数据
        """
        if self.status == "down":
            self.logger.warning(f"数据源 {self.name} 当前处于下线状态，跳过请求")
            return pd.DataFrame()
            
        start_time = time.time()
        success = False
        result = pd.DataFrame()
        
        try:
            result = self.get_data(**kwargs)
            success = not result.empty
        except Exception as e:
            self.logger.error(f"从数据源 {self.name} 获取数据失败: {e}")
            success = False
            
        response_time = time.time() - start_time
        self.record_request(success, response_time)
        
        return result


class TushareDataSource(DataSource):
    """Tushare数据源"""
    
    def __init__(self, token=None, priority=0, weight=1.0):
        """初始化Tushare数据源"""
        super().__init__(name="tushare", priority=priority, weight=weight)
        
        # 尝试导入Tushare
        try:
            import tushare as ts
            self.ts = ts
            
            # 初始化API
            self.token = token or os.environ.get('TUSHARE_TOKEN')
            if self.token:
                try:
                    ts.set_token(self.token)
                    self.pro_api = ts.pro_api()
                    self.status = "up"
                    self.logger.info("Tushare API初始化成功")
                except Exception as e:
                    self.logger.error(f"Tushare API初始化失败: {e}")
                    self.status = "down"
                    self.pro_api = None
            else:
                self.logger.warning("未设置Tushare Token，相关功能将不可用")
                self.status = "down"
                self.pro_api = None
                
        except ImportError:
            self.logger.error("未安装Tushare库，此数据源不可用")
            self.status = "down"
            self.ts = None
            self.pro_api = None
            
    def get_data(self, api_name, **kwargs):
        """
        通过Tushare API获取数据
        
        Args:
            api_name: API名称，如'daily', 'stock_basic'等
            **kwargs: API参数
            
        Returns:
            pd.DataFrame: API返回的数据
        """
        if self.status == "down" or self.pro_api is None:
            self.logger.warning("Tushare API不可用")
            return pd.DataFrame()
            
        try:
            # 获取API方法
            api_method = getattr(self.pro_api, api_name, None)
            if api_method is None:
                self.logger.error(f"Tushare API '{api_name}'不存在")
                return pd.DataFrame()
                
            # 调用API
            df = api_method(**kwargs)
            
            # 验证数据有效性
            valid, error_msg = self.validator.validate_stock_data(df)
            if not valid:
                self.logger.warning(f"从Tushare获取的数据验证失败: {error_msg}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"调用Tushare API '{api_name}'失败: {e}")
            return pd.DataFrame()
            
    def health_check(self):
        """Tushare特定的健康检查"""
        if self.pro_api is None:
            return False
            
        try:
            # 尝试获取一些基本数据作为健康检查
            result = self.pro_api.query('trade_cal', exchange='SSE', start_date='20230101', end_date='20230110')
            is_healthy = not result.empty
            
            if is_healthy:
                self.status = "up"
                self.health_score = 1.0
            else:
                self.status = "degraded"
                self.health_score = 0.5
                
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Tushare健康检查失败: {e}")
            self.status = "down"
            self.health_score = 0.0
            return False


class AkshareDataSource(DataSource):
    """AkShare数据源"""
    
    def __init__(self, priority=1, weight=0.8):
        """初始化AkShare数据源"""
        super().__init__(name="akshare", priority=priority, weight=weight)
        
        # 尝试导入AkShare
        try:
            import akshare as ak
            self.ak = ak
            self.status = "up"
            self.logger.info("AkShare初始化成功")
        except ImportError:
            self.logger.error("未安装AkShare库，此数据源不可用")
            self.status = "down"
            self.ak = None
            
    def get_data(self, func_name, **kwargs):
        """
        通过AkShare函数获取数据
        
        Args:
            func_name: 函数名称，如'stock_zh_a_daily'
            **kwargs: 函数参数
            
        Returns:
            pd.DataFrame: 函数返回的数据
        """
        if self.status == "down" or self.ak is None:
            self.logger.warning("AkShare不可用")
            return pd.DataFrame()
            
        try:
            # 获取AkShare函数
            func = getattr(self.ak, func_name, None)
            if func is None:
                self.logger.error(f"AkShare函数'{func_name}'不存在")
                return pd.DataFrame()
                
            # 调用函数
            df = func(**kwargs)
            
            # 验证数据有效性
            valid, error_msg = self.validator.validate_stock_data(df)
            if not valid:
                self.logger.warning(f"从AkShare获取的数据验证失败: {error_msg}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"调用AkShare函数'{func_name}'失败: {e}")
            return pd.DataFrame()
            
    def health_check(self):
        """AkShare特定的健康检查"""
        if self.ak is None:
            return False
            
        try:
            # 尝试获取一些基本数据作为健康检查
            result = self.ak.stock_zh_index_daily(symbol="sh000001")
            is_healthy = not result.empty
            
            if is_healthy:
                self.status = "up"
                self.health_score = 1.0
            else:
                self.status = "degraded"
                self.health_score = 0.5
                
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"AkShare健康检查失败: {e}")
            self.status = "down"
            self.health_score = 0.0
            return False


class JoinQuantDataSource(DataSource):
    """JoinQuant数据源"""
    
    def __init__(self, username=None, password=None, priority=0, weight=1.0):
        """
        初始化JoinQuant数据源
        
        Args:
            username: JoinQuant账号
            password: JoinQuant密码
            priority: 优先级，默认最高优先级0
            weight: 在数据融合中的权重，默认1.0
        """
        super().__init__(name="joinquant", priority=priority, weight=weight)
        
        # 保存账号信息
        self.username = username or os.environ.get('JOINQUANT_USERNAME')
        self.password = password or os.environ.get('JOINQUANT_PASSWORD')
        self.auth_token = None
        self.token_expire_time = None
        
        # 设置默认允许的日期范围（根据账户权限）
        self.min_date = '2023-12-14'  # 最早允许的日期
        self.max_date = '2024-12-20'  # 最晚允许的日期
        
        # 尝试导入jqdatasdk
        try:
            import jqdatasdk as jq
            self.jq = jq
            
            # 初始化连接
            if self.username and self.password:
                try:
                    jq.auth(self.username, self.password)
                    self.status = "up"
                    self.auth_token = True  # JoinQuant认证后不返回token，这里用布尔值代替
                    self.token_expire_time = datetime.now() + timedelta(days=1)  # 假设token有效期为1天
                    self.logger.info("JoinQuant API初始化成功")
                except Exception as e:
                    self.logger.error(f"JoinQuant API认证失败: {e}")
                    self.status = "down"
            else:
                self.logger.warning("未设置JoinQuant账号密码，相关功能将不可用")
                self.status = "down"
                
        except ImportError:
            self.logger.error("未安装jqdatasdk库，此数据源不可用")
            self.status = "down"
            self.jq = None
    
    def _adjust_date_range(self, start_date, end_date):
        """
        根据账户权限调整日期范围
        
        Args:
            start_date: 原始开始日期（字符串或datetime）
            end_date: 原始结束日期（字符串或datetime）
            
        Returns:
            Tuple[str, str]: 调整后的日期范围（字符串格式：'YYYY-MM-DD'）
        """
        # 转换日期格式为字符串
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # 确保日期格式正确
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            self.logger.warning(f"日期格式错误，使用默认范围: {self.min_date} 至 {self.max_date}")
            return self.min_date, self.max_date
            
        # 调整开始日期
        if start_date < self.min_date:
            self.logger.warning(f"起始日期 {start_date} 早于允许的最早日期 {self.min_date}，已自动调整")
            start_date = self.min_date
            
        # 调整结束日期
        if end_date > self.max_date:
            self.logger.warning(f"结束日期 {end_date} 晚于允许的最晚日期 {self.max_date}，已自动调整")
            end_date = self.max_date
            
        return start_date, end_date
            
    def get_data(self, func_name, **kwargs):
        """
        通过JoinQuant API获取数据
        
        Args:
            func_name: API函数名称，如'get_price', 'get_fundamentals'等
            **kwargs: API参数
            
        Returns:
            pd.DataFrame: API返回的数据
        """
        if self.status == "down" or self.jq is None:
            self.logger.warning("JoinQuant API不可用")
            return pd.DataFrame()
            
        # 检查token是否过期，需要重新认证
        if not self.auth_token or (self.token_expire_time and datetime.now() > self.token_expire_time):
            try:
                self.jq.auth(self.username, self.password)
                self.auth_token = True
                self.token_expire_time = datetime.now() + timedelta(days=1)
                self.logger.info("JoinQuant API重新认证成功")
            except Exception as e:
                self.logger.error(f"JoinQuant API重新认证失败: {e}")
                self.status = "down"
                return pd.DataFrame()
                
        # 处理日期参数，确保在允许的范围内
        if func_name in ['get_price', 'get_bars', 'get_factor_values', 'get_index_stocks', 'get_industry_stocks']:
            # 调整日期范围
            if 'start_date' in kwargs and 'end_date' in kwargs:
                start_date, end_date = self._adjust_date_range(kwargs['start_date'], kwargs['end_date'])
                kwargs['start_date'] = start_date
                kwargs['end_date'] = end_date
                self.logger.info(f"使用调整后的日期范围: {start_date} 至 {end_date}")
            elif 'date' in kwargs:
                date = kwargs['date']
                if isinstance(date, datetime):
                    date = date.strftime('%Y-%m-%d')
                    
                # 确保日期在允许范围内
                if date < self.min_date:
                    date = self.min_date
                    self.logger.warning(f"日期 {date} 早于允许的最早日期 {self.min_date}，已自动调整")
                elif date > self.max_date:
                    date = self.max_date
                    self.logger.warning(f"日期 {date} 晚于允许的最晚日期 {self.max_date}，已自动调整")
                    
                kwargs['date'] = date
                
        try:
            # 获取API方法
            api_method = getattr(self.jq, func_name, None)
            if api_method is None:
                self.logger.error(f"JoinQuant API '{func_name}'不存在")
                return pd.DataFrame()
                
            # 调用API
            df = api_method(**kwargs)
            
            # 转换为DataFrame（如果返回值不是DataFrame）
            if not isinstance(df, pd.DataFrame):
                if isinstance(df, (list, tuple)):
                    df = pd.DataFrame(df)
                else:
                    self.logger.warning(f"JoinQuant API '{func_name}'返回值不是DataFrame，尝试转换")
                    df = pd.DataFrame([df]) if df is not None else pd.DataFrame()
            
            # 验证数据有效性
            valid, error_msg = self.validator.validate_stock_data(df)
            if not valid:
                self.logger.warning(f"从JoinQuant获取的数据验证失败: {error_msg}")
                
            return df
            
        except Exception as e:
            error_msg = str(e)
            # 检查是否是日期范围错误
            if "权限仅能获取" in error_msg and "的数据" in error_msg:
                try:
                    # 尝试从错误信息中提取允许的日期范围
                    import re
                    date_range = re.search(r'(\d{4}-\d{2}-\d{2})至(\d{4}-\d{2}-\d{2})', error_msg)
                    if date_range:
                        self.min_date = date_range.group(1)
                        self.max_date = date_range.group(2)
                        self.logger.info(f"更新允许的日期范围: {self.min_date} 至 {self.max_date}")
                        
                        # 如果是日期相关函数，尝试用新的日期范围重新调用
                        if func_name in ['get_price', 'get_bars', 'get_factor_values']:
                            if 'start_date' in kwargs and 'end_date' in kwargs:
                                kwargs['start_date'], kwargs['end_date'] = self._adjust_date_range(
                                    kwargs['start_date'], kwargs['end_date']
                                )
                                self.logger.info(f"使用新的日期范围重试: {kwargs['start_date']} 至 {kwargs['end_date']}")
                                return self.get_data(func_name, **kwargs)
                except:
                    pass
                    
            self.logger.error(f"调用JoinQuant API '{func_name}'失败: {error_msg}")
            return pd.DataFrame()
            
    def health_check(self):
        """JoinQuant特定的健康检查"""
        if self.jq is None:
            return False
            
        try:
            # 尝试获取一些基本数据作为健康检查
            result = self.jq.get_trade_days(start_date='2023-01-01', end_date='2023-01-10')
            is_healthy = len(result) > 0
            
            if is_healthy:
                self.status = "up"
                self.health_score = 1.0
            else:
                self.status = "degraded"
                self.health_score = 0.5
                
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"JoinQuant健康检查失败: {e}")
            self.status = "down"
            self.health_score = 0.0
            return False


class DataSourceManager:
    """
    数据源管理器 - 单例模式实现
    负责管理和协调多个数据源
    """
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """获取单例实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    def __init__(self):
        """初始化数据源管理器"""
        self.sources = {}  # 存储数据源的字典
        logger.info("数据源管理器初始化")
        self.validator = DataValidator()
        self.source_mapping = {
            'stock_list': {
                'tushare': {'api_name': 'stock_basic', 'params': {'exchange': '', 'list_status': 'L'}},
                'akshare': {'func_name': 'stock_info_a_code_name', 'params': {}},
                'joinquant': {'func_name': 'get_all_securities', 'params': {'types': ['stock']}}
            },
            'daily_data': {
                'tushare': {'api_name': 'daily', 'params': {}},
                'akshare': {'func_name': 'stock_zh_a_daily', 'params': {}},
                'joinquant': {'func_name': 'get_price', 'params': {'frequency': 'daily', 'fields': ['open', 'close', 'high', 'low', 'volume']}}
            },
            'dividend': {
                'tushare': {'api_name': 'dividend', 'params': {}},
                'akshare': {'func_name': 'stock_dividend_cninfo', 'params': {}},
                'joinquant': {'func_name': 'get_dividend', 'params': {}}
            },
            'stock_splits': {
                'tushare': {'api_name': 'stock_basic', 'params': {'fields': 'ts_code,name,area,list_date'}},
                'akshare': {'func_name': 'stock_info_a_code_name', 'params': {}},
                'joinquant': {'func_name': 'get_extras', 'params': {'info': 'is_st'}}
            },
            'trade_cal': {
                'tushare': {'api_name': 'trade_cal', 'params': {}},
                'akshare': {'func_name': 'tool_trade_date_hist_sina', 'params': {}},
                'joinquant': {'func_name': 'get_trade_days', 'params': {}}
            },
            'financial_data': {
                'tushare': {'api_name': 'income', 'params': {}},
                'akshare': {'func_name': 'stock_financial_report_sina', 'params': {'symbol': '{stock_code}'}},
                'joinquant': {'func_name': 'get_fundamentals', 'params': {'query_object': 'query(income)'}}
            },
            # 添加其他数据类型的映射
            'index_daily': {
                'tushare': {'api_name': 'index_daily', 'params': {}},
                'akshare': {'func_name': 'stock_zh_index_daily', 'params': {'symbol': '{index_code}'}},
                'joinquant': {'func_name': 'get_price', 'params': {'frequency': 'daily', 'index': True}}
            },
            'stock_company': {
                'tushare': {'api_name': 'stock_company', 'params': {}},
                'akshare': {'func_name': 'stock_info_a_code_name', 'params': {}},
                'joinquant': {'func_name': 'get_security_info', 'params': {}}
            }
        }
        
    def register_source(self, name, source):
        """
        注册数据源
        
        参数:
            name (str): 数据源名称
            source (object): 数据源对象
            
        返回:
            bool: 是否注册成功
        """
        if name in self.sources:
            logger.warning(f"数据源 '{name}' 已存在，将被覆盖")
        
        self.sources[name] = source
        logger.info(f"注册数据源: {name}")
        return True
    
    def get_source(self, name):
        """
        获取数据源
        
        参数:
            name (str): 数据源名称
            
        返回:
            object: 数据源对象，不存在则返回None
        """
        if name not in self.sources:
            logger.warning(f"数据源 '{name}' 不存在")
            return None
        
        return self.sources[name]
    
    def list_sources(self):
        """
        列出所有可用的数据源
        
        返回:
            list: 数据源名称列表
        """
        return list(self.sources.keys())
    
    def init_default_sources(self):
        """
        初始化默认数据源
        
        返回:
            bool: 是否成功初始化
        """
        try:
            # 模拟创建本地数据源
            class LocalDataSource:
                def __init__(self):
                    self.name = "本地数据源"
                    self.data_dir = "./data/local"
                    
                def get_data(self, code, start_date=None, end_date=None):
                    # 模拟获取数据
                    return None
            
            # 模拟创建在线数据源
            class OnlineDataSource:
                def __init__(self):
                    self.name = "在线数据源"
                    self.api_key = "demo_key"
                    
                def get_data(self, code, start_date=None, end_date=None):
                    # 模拟获取数据
                    return None
            
            # 注册数据源
            self.register_source("local", LocalDataSource())
            self.register_source("online", OnlineDataSource())
            
            logger.info("默认数据源初始化完成")
            return True
        except Exception as e:
            logger.error(f"初始化默认数据源失败: {e}")
            return False
    
    def get_best_source(self, data_type=None):
        """
        根据数据类型获取最佳数据源
        
        参数:
            data_type (str, optional): 数据类型
            
        返回:
            object: 数据源对象
        """
        # 简单实现，后期可以根据数据类型和质量评分选择最佳源
        if not self.sources:
            return None
            
        # 优先级：local > online
        if "local" in self.sources:
            return self.sources["local"]
        
        # 返回第一个可用源
        return next(iter(self.sources.values()))
    
    def get_data_sources(self, data_type=None):
        """
        获取所有可用的数据源
        
        Args:
            data_type: 可选，指定数据类型筛选数据源
            
        Returns:
            list: 数据源列表，按优先级排序
        """
        # 过滤状态良好的数据源
        available_sources = [
            source for source in self.sources.values()
            if source.status != "down" and source.health_check()
        ]
        
        # 如果指定了数据类型，进一步筛选支持该类型的数据源
        if data_type and data_type in self.source_mapping:
            available_sources = [
                source for source in available_sources
                if source.name in self.source_mapping[data_type]
            ]
            
        # 按优先级排序
        return sorted(available_sources, key=lambda s: s.priority)
        
    def get_data(self, data_type, fallback=True, validate=True, **kwargs):
        """
        从最合适的数据源获取数据
        
        Args:
            data_type: 数据类型
            fallback: 是否在主数据源失败时尝试备用数据源
            validate: 是否验证数据
            **kwargs: 传递给数据源的参数
            
        Returns:
            获取的数据，失败返回None
        """
        start_time = time.time()
        sources = self.get_data_sources(data_type)
        
        if not sources:
            self.logger.warning(f"无可用的数据源提供 {data_type} 类型数据")
            return None
            
        # 创建验证上下文
        context = {'data_type': data_type}
        
        # 标记是否有数据源成功
        success = False
        result_data = None
        
        for source in sources:
            try:
                self.logger.info(f"尝试从 {source.name} 获取 {data_type} 数据")
                data = source.get_data_safe(data_type=data_type, **kwargs)
                
                if data is not None:
                    self.logger.info(f"从 {source.name} 成功获取 {data_type} 数据")
                    success = True
                    result_data = data
                    
                    # 执行数据验证
                    if validate:
                        try:
                            # 根据数据类型选择验证方法
                            if data_type in ['daily', 'price', 'bar']:
                                # 使用新的验证流程
                                pipeline = DataPipeline(f"validate_{data_type}")
                                pipeline.add_stage(validate_price_data)
                                validation_result = pipeline.execute(data, context)
                                
                                # 检查验证结果
                                if not context.get('validation_result', {}).get('valid', True):
                                    self.logger.warning(f"数据源 {source.name} 提供的 {data_type} 数据验证失败")
                                    if fallback:
                                        # 失败时尝试下一个数据源
                                        continue
                                    
                            elif data_type in ['fundamental', 'financial']:
                                pipeline = DataPipeline(f"validate_{data_type}")
                                pipeline.add_stage(validate_financial_data)
                                validation_result = pipeline.execute(data, context)
                                
                                # 检查验证结果
                                if not context.get('validation_result', {}).get('valid', True):
                                    self.logger.warning(f"数据源 {source.name} 提供的 {data_type} 数据验证失败")
                                    if fallback:
                                        # 失败时尝试下一个数据源
                                        continue
                        except Exception as ve:
                            self.logger.warning(f"验证数据时出错: {ve}")
                    
                    # 记录成功的请求
                    elapsed = time.time() - start_time
                    source.record_request(True, elapsed)
                    break
                else:
                    self.logger.warning(f"数据源 {source.name} 返回空数据")
                    source.record_request(False, time.time() - start_time)
                    
            except Exception as e:
                self.logger.warning(f"从数据源 {source.name} 获取 {data_type} 数据失败: {e}")
                source.record_request(False, time.time() - start_time)
                
            # 如果不需要故障转移，则在第一个源失败后直接返回
            if not fallback:
                break
                
        # 所有数据源都失败的情况
        if not success:
            self.logger.error(f"所有数据源获取 {data_type} 数据均失败")
            return None
            
        # 返回成功获取的数据
        return result_data
        
    def get_data_parallel(self, data_type, combine_method='vote', timeout=10, **kwargs):
        """
        并行从多个数据源获取数据并进行融合
        
        Args:
            data_type: 数据类型
            combine_method: 数据融合方法，可选'vote'(投票),'weighted'(加权),'primary'(以主数据源为主)
            timeout: 超时时间(秒)
            **kwargs: 其他参数
            
        Returns:
            pd.DataFrame: 融合后的数据
        """
        # 获取可用数据源
        sources = self.get_data_sources(data_type)
        
        if not sources:
            self.logger.error(f"没有可用的数据源用于获取'{data_type}'")
            return pd.DataFrame()
            
        source_results = {}
        
        # 准备线程池执行并行请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as executor:
            # 提交所有任务
            future_to_source = {}
            for source in sources:
                source_config = self.source_mapping.get(data_type, {}).get(source.name, {})
                if not source_config:
                    continue
                    
                # 准备参数
                api_params = {**source_config.get('params', {}), **kwargs}
                
                if source.name == 'tushare':
                    future = executor.submit(
                        source.get_data,
                        api_name=source_config.get('api_name'),
                        **api_params
                    )
                elif source.name == 'akshare':
                    future = executor.submit(
                        source.get_data,
                        func_name=source_config.get('func_name'),
                        **api_params
                    )
                else:
                    continue
                    
                future_to_source[future] = source
                
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_source, timeout=timeout):
                source = future_to_source[future]
                try:
                    data = future.result()
                    # 验证数据
                    valid, _ = self.validator.validate_stock_data(data)
                    if valid and not data.empty:
                        source_results[source.name] = data
                except Exception as e:
                    self.logger.error(f"并行获取数据时出错(源:{source.name}): {e}")
                    
        # 进行数据融合
        if not source_results:
            self.logger.error(f"从所有数据源并行获取{data_type}数据均失败")
            return pd.DataFrame()
            
        if len(source_results) == 1:
            # 只有一个数据源返回结果，直接使用
            return list(source_results.values())[0]
        
        # 有多个数据源，进行数据融合
        return self._combine_data(source_results, combine_method, data_type)
        
    def _combine_data(self, source_results, method, data_type):
        """
        融合来自多个数据源的数据
        
        Args:
            source_results: 各数据源的结果字典 {source_name: dataframe}
            method: 融合方法
            data_type: 数据类型
            
        Returns:
            pd.DataFrame: 融合后的数据
        """
        if method == 'primary':
            # 使用优先级最高的数据源结果
            primary_source = min(self.sources.values(), key=lambda s: s.priority)
            if primary_source.name in source_results:
                return source_results[primary_source.name]
            else:
                # 如果主数据源没有结果，使用第一个可用的
                return list(source_results.values())[0]
                
        # 对于其他融合方法，需要具体处理
        if data_type == 'stock_list':
            return self._combine_stock_lists(source_results, method)
        elif data_type == 'daily_data':
            return self._combine_daily_data(source_results, method)
        else:
            # 默认返回第一个结果
            self.logger.warning(f"未实现数据类型'{data_type}'的融合方法，返回第一个结果")
            return list(source_results.values())[0]
            
    def _combine_stock_lists(self, source_results, method):
        """融合股票列表数据"""
        # 提取所有数据源的股票代码列表
        all_codes = set()
        code_columns = ['ts_code', 'code', 'symbol']  # 可能的股票代码列名
        
        # 找出每个数据源中的股票代码列
        source_code_cols = {}
        for source_name, df in source_results.items():
            for col in code_columns:
                if col in df.columns:
                    source_code_cols[source_name] = col
                    all_codes.update(df[col].tolist())
                    break
                    
        if not all_codes:
            self.logger.error("无法找到任何股票代码列")
            return list(source_results.values())[0]
            
        # 创建合并后的DataFrame
        result_df = pd.DataFrame({"code": list(all_codes)})
        
        # 对于每个股票代码，从所有数据源获取信息并进行投票或加权
        for source_name, df in source_results.items():
            if source_name not in source_code_cols:
                continue
                
            code_col = source_code_cols[source_name]
            
            # 将该数据源的列添加到结果中，使用后缀区分
            df_renamed = df.rename(columns={col: f"{col}_{source_name}" for col in df.columns})
            result_df = pd.merge(
                result_df,
                df_renamed,
                left_on="code",
                right_on=f"{code_col}_{source_name}",
                how="left"
            )
            
        # 为每个列创建最终值
        name_columns = [col for col in result_df.columns if 'name' in col.lower()]
        if name_columns:
            if method == 'vote':
                # 投票选择出现最多的名称
                result_df['name'] = result_df[name_columns].mode(axis=1)[0]
            elif method == 'weighted':
                # 加权选择，优先选择权重高的数据源
                for col in sorted(name_columns, key=lambda c: self.sources[c.split('_')[-1]].weight, reverse=True):
                    if 'name' not in result_df.columns:
                        result_df['name'] = result_df[col]
                    else:
                        result_df['name'] = result_df['name'].fillna(result_df[col])
                        
        # 处理行业列
        industry_columns = [col for col in result_df.columns if 'industry' in col.lower()]
        if industry_columns:
            if method == 'vote':
                result_df['industry'] = result_df[industry_columns].mode(axis=1)[0]
            elif method == 'weighted':
                for col in sorted(industry_columns, key=lambda c: self.sources[c.split('_')[-1]].weight, reverse=True):
                    if 'industry' not in result_df.columns:
                        result_df['industry'] = result_df[col]
                    else:
                        result_df['industry'] = result_df['industry'].fillna(result_df[col])
                        
        # 保留必要的列
        keep_columns = ['code', 'name', 'industry', 'area', 'market', 'list_date']
        final_columns = ['code', 'name']  # 必须保留的列
        
        for col in keep_columns:
            if col in result_df.columns:
                final_columns.append(col)
            else:
                # 查找带有该列名的列并选择一个
                similar_cols = [c for c in result_df.columns if col in c.lower()]
                if similar_cols:
                    if method == 'weighted':
                        # 按权重排序
                        similar_cols.sort(
                            key=lambda c: self.sources[c.split('_')[-1]].weight if len(c.split('_')) > 1 else 0,
                            reverse=True
                        )
                    result_df[col] = result_df[similar_cols[0]]
                    final_columns.append(col)
                    
        return result_df[final_columns]
        
    def _combine_daily_data(self, source_results, method):
        """融合日线数据"""
        # 这里假设所有数据源都有date和对应的股票代码列
        all_dates = set()
        for df in source_results.values():
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            if date_col:
                all_dates.update(df[date_col].tolist())
                
        if not all_dates:
            self.logger.error("无法找到任何日期列")
            return list(source_results.values())[0]
            
        # 创建结果DataFrame
        stock_code = next(iter(kwargs.get('ts_code', kwargs.get('symbol', ''))))
        result_df = pd.DataFrame({"date": list(all_dates), "stock_code": stock_code})
        
        # 处理OHLCV数据
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in ohlcv_columns:
            col_values = {}
            col_weights = {}
            
            for source_name, df in source_results.items():
                source_weight = self.sources[source_name].weight
                
                # 找到对应的列
                source_col = next((c for c in df.columns if col.lower() in c.lower()), None)
                if not source_col:
                    continue
                    
                # 获取日期列
                date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                if not date_col:
                    continue
                    
                # 将该源的数据添加到合并字典
                for date, value in zip(df[date_col], df[source_col]):
                    if pd.notna(value):
                        if date not in col_values:
                            col_values[date] = []
                            col_weights[date] = []
                            
                        col_values[date].append(value)
                        col_weights[date].append(source_weight)
                        
            # 根据方法计算最终值
            if method == 'vote':
                # 投票法 - 使用众数
                result_df[col] = result_df['date'].apply(
                    lambda date: pd.Series(col_values.get(date, [np.nan])).mode()[0] 
                    if date in col_values else np.nan
                )
            elif method == 'weighted':
                # 加权平均
                result_df[col] = result_df['date'].apply(
                    lambda date: np.average(col_values.get(date, [np.nan]), weights=col_weights.get(date, [1]))
                    if date in col_values and len(col_values[date]) > 0 else np.nan
                )
                
        return result_df
        
    def health_check_all(self):
        """
        检查所有数据源的健康状况
        
        Returns:
            dict: 各数据源的健康状况
        """
        health_status = {}
        
        for name, source in self.sources.items():
            is_healthy = source.health_check()
            health_status[name] = {
                'status': source.status,
                'health_score': source.health_score,
                'is_healthy': is_healthy,
                'error_count': source.error_count,
                'success_rate': source.success_requests / max(1, source.total_requests),
                'avg_response_time': source.average_response_time
            }
            
        return health_status
        
    def start_health_monitor(self, interval=3600):
        """
        启动数据源健康监控线程
        
        Args:
            interval: 检查间隔(秒)
        """
        if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
            self.logger.info("健康监控线程已在运行")
            return
            
        def monitor_task():
            while True:
                try:
                    self.logger.info("执行数据源健康检查...")
                    health_status = self.health_check_all()
                    
                    # 输出健康状况
                    for name, status in health_status.items():
                        self.logger.info(f"数据源 {name} 健康状况: {status}")
                        
                    # 尝试恢复状态为down但健康检查通过的数据源
                    for name, source in self.sources.items():
                        if source.status == "down" and source.health_check():
                            self.logger.info(f"数据源 {name} 已自动恢复")
                            source.status = "up"
                            
                    # 休眠指定时间
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"健康监控任务异常: {e}")
                    time.sleep(60)  # 出错后短暂休眠
                    
        # 创建并启动监控线程
        self._monitor_thread = threading.Thread(target=monitor_task, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"数据源健康监控已启动，间隔 {interval} 秒")
        
    def get_source_status_report(self):
        """
        获取数据源状态报告
        
        Returns:
            Dict: 数据源状态报告
        """
        report = {}
        
        for name, source in self.sources.items():
            report[name] = {
                'status': source.status,
                'health_score': source.health_score,
                'priority': source.priority,
                'weight': source.weight,
                'total_requests': source.total_requests,
                'success_rate': source.success_requests / max(1, source.total_requests) * 100,
                'avg_response_time': source.average_response_time,
                'last_error': source.last_error_time.isoformat() if source.last_error_time else None,
                'last_success': source.last_success_time.isoformat() if source.last_success_time else None,
                'error_count': source.error_count
            }
            
        return report 