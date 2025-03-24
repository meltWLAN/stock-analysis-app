#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据源示例 - 演示如何使用依赖注入设计模式实现可扩展的数据源
"""

import logging
import time
import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import threading

from utils.config_manager import get_config_manager
from utils.cache_manager import CacheManager

class DataSource:
    """数据源基类 - 使用依赖注入模式"""
    
    def __init__(self, 
                 source_name: str,
                 validator=None, 
                 cache_manager=None, 
                 logger=None):
        """
        初始化数据源
        
        Args:
            source_name: 数据源名称，用于获取配置
            validator: 数据验证器
            cache_manager: 缓存管理器
            logger: 日志记录器
        """
        # 设置数据源名称
        self.source_name = source_name
        
        # 获取配置管理器
        self.config_manager = get_config_manager()
        
        # 加载数据源配置
        self.config = self.config_manager.get_data_source_config(source_name)
        
        # 依赖注入
        self.validator = validator
        self.logger = logger or logging.getLogger(__name__)
        self.cache = cache_manager or CacheManager()
        
        # 健康状态指标
        self.health_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_error': None,
            'last_error_time': None,
            'avg_response_time': 0,
            'total_response_time': 0,
        }
        
        # 检查数据源是否启用
        if not self.config.get('enabled', True):
            self.logger.warning(f"数据源 {self.source_name} 已禁用")
    
    def get_data(self, 
                params: Dict[str, Any], 
                use_cache: bool = True,
                **kwargs) -> Tuple[bool, Any]:
        """
        获取数据
        
        Args:
            params: 请求参数
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            Tuple[bool, Any]: (是否成功, 数据或错误信息)
        """
        self.health_stats['total_requests'] += 1
        start_time = time.time()
        
        # 快速检查数据源是否禁用
        if not self.config.get('enabled', True):
            error_msg = f"数据源 {self.source_name} 已禁用"
            self.logger.warning(error_msg)
            self._update_health_stats(False, time.time() - start_time, error_msg)
            return False, error_msg
        
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(params, **kwargs) if use_cache else None
            
            # 尝试从缓存获取数据 - 优化点：直接使用缓存键而不是调用单独的方法
            if use_cache and cache_key:
                # 直接从缓存获取
                data = self.cache.get(cache_key)
                if data is not None:
                    self.logger.debug(f"从缓存获取数据: {cache_key}")
                    self._update_health_stats(True, time.time() - start_time)
                    return True, data
            
            # 如果缓存中没有，则获取新数据
            success, data = self._fetch_data(params, **kwargs)
            
            if success:
                # 如果启用缓存且获取数据成功，将新数据存入缓存
                if use_cache and cache_key and self.config_manager.get('cache.enabled', True):
                    expiry = self.config_manager.get('cache.default_expiry', 3600)
                    # 异步设置缓存，避免阻塞
                    self._async_set_cache(cache_key, data, expiry)
                
                # 如果有验证器，验证数据
                if self.validator:
                    is_valid, validation_msg = self.validate_data(data)
                    if not is_valid:
                        self.logger.error(f"数据验证失败: {validation_msg}")
                        self._update_health_stats(False, time.time() - start_time, validation_msg)
                        return False, validation_msg
                
                # 更新健康状态
                self._update_health_stats(True, time.time() - start_time)
                
                # 返回数据
                return True, data
            else:
                # 更新健康状态
                self._update_health_stats(False, time.time() - start_time, data)
                return False, data
                
        except Exception as e:
            error_msg = f"获取数据异常: {str(e)}"
            self.logger.exception(error_msg)
            
            # 更新健康状态
            self._update_health_stats(False, time.time() - start_time, str(e))
            
            return False, error_msg
    
    def _update_health_stats(self, success: bool, elapsed_time: float, error_msg: str = None):
        """
        更新健康统计信息 - 提取为独立方法以提高可维护性
        
        Args:
            success: 操作是否成功
            elapsed_time: 操作耗时
            error_msg: 错误信息（如果有）
        """
        # 更新总请求响应时间统计
        total_time = self.health_stats['total_response_time'] + elapsed_time
        total_requests = self.health_stats['total_requests']
        
        self.health_stats['total_response_time'] = total_time
        self.health_stats['avg_response_time'] = total_time / total_requests if total_requests > 0 else 0
        
        # 更新成功/失败次数
        if success:
            self.health_stats['successful_requests'] += 1
        else:
            self.health_stats['failed_requests'] += 1
            self.health_stats['last_error'] = error_msg
            self.health_stats['last_error_time'] = datetime.datetime.now()
    
    def _async_set_cache(self, key: str, data: Any, expiry: int):
        """
        异步设置缓存，避免阻塞主流程
        
        Args:
            key: 缓存键
            data: 缓存数据
            expiry: 过期时间（秒）
        """
        try:
            # 使用线程异步设置缓存
            threading.Thread(
                target=self.cache.set,
                args=(key, data, expiry),
                daemon=True
            ).start()
        except Exception as e:
            self.logger.warning(f"异步设置缓存失败: {e}")
            # 回退到同步设置
            try:
                self.cache.set(key, data, expiry)
            except Exception as e2:
                self.logger.error(f"同步设置缓存也失败: {e2}")
    
    def _generate_cache_key(self, 
                           params: Dict[str, Any], 
                           **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            params: 请求参数
            **kwargs: 其他参数
            
        Returns:
            str: 缓存键
        """
        # 基本键包含数据源名称
        key_parts = [self.source_name]
        
        # 优化点：添加版本号，便于缓存控制
        key_parts.append(f"v={self.config.get('version', '1.0')}")
        
        # 添加排序后的参数以确保一致性
        if params:
            param_parts = []
            for k, v in sorted(params.items()):
                # 优化点：对复杂对象使用固定长度的哈希
                if isinstance(v, (dict, list, tuple, set)):
                    import hashlib
                    import json
                    v_hash = hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()[:8]
                    param_parts.append(f"{k}={v_hash}")
                else:
                    param_parts.append(f"{k}={v}")
            key_parts.append(",".join(param_parts))
            
        # 添加其他关键参数
        for k, v in sorted(kwargs.items()):
            if k not in ('validator', 'logger', 'cache'):
                # 同样处理复杂对象
                if isinstance(v, (dict, list, tuple, set)):
                    import hashlib
                    import json
                    v_hash = hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()[:8]
                    key_parts.append(f"{k}={v_hash}")
                else:
                    key_parts.append(f"{k}={v}")
                
        # 组合为字符串
        return ':'.join(key_parts)
    
    def _fetch_data(self, 
                   params: Dict[str, Any], 
                   **kwargs) -> Tuple[bool, Any]:
        """
        从数据源获取数据（由子类实现）
        
        Args:
            params: 请求参数
            **kwargs: 其他参数
            
        Returns:
            Tuple[bool, Any]: (是否成功, 数据或错误信息)
        """
        raise NotImplementedError("子类必须实现_fetch_data方法")
    
    def validate_data(self, data: Any) -> Tuple[bool, str]:
        """
        验证数据
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if self.validator:
            return self.validator.validate(data)
        return True, ""
    
    def health_check(self) -> Dict[str, Any]:
        """
        检查数据源健康状态
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        # 计算成功率
        total = self.health_stats['total_requests']
        success_rate = (self.health_stats['successful_requests'] / total) * 100 if total > 0 else 0
        
        # 判断健康状态
        health_status = "HEALTHY"
        if success_rate < 90:
            health_status = "WARNING"
        if success_rate < 70:
            health_status = "UNHEALTHY"
            
        # 构建健康报告
        health_report = {
            'source_name': self.source_name,
            'status': health_status,
            'success_rate': success_rate,
            'avg_response_time': self.health_stats['avg_response_time'],
            'total_requests': total,
            'successful_requests': self.health_stats['successful_requests'],
            'failed_requests': self.health_stats['failed_requests'],
            'last_error': self.health_stats['last_error'],
            'last_error_time': self.health_stats['last_error_time'],
            'config': {
                'enabled': self.config.get('enabled', True),
                'priority': self.config.get('priority', 0),
                'max_retry': self.config.get('max_retry', 3),
                'timeout': self.config.get('timeout', 30),
            }
        }
        
        return health_report
    
    def prepare_date_range(self, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Tuple[str, str]:
        """
        准备并验证日期范围
        
        Args:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            Tuple[str, str]: (开始日期, 结束日期)
        """
        # 获取配置的日期范围
        config_min_date = self.config.get('min_date')
        config_max_date = self.config.get('max_date')
        
        # 如果未提供开始日期，使用配置中的最小日期
        if not start_date and config_min_date:
            start_date = config_min_date
        
        # 如果未提供结束日期，使用配置中的最大日期或当前日期
        if not end_date:
            if config_max_date:
                end_date = config_max_date
            else:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 确保日期格式一致
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        
        # 验证日期范围
        if start_date > end_date:
            self.logger.warning(f"开始日期 {start_date} 大于结束日期 {end_date}，将交换日期")
            start_date, end_date = end_date, start_date
        
        # 如果配置了有效日期范围，确保请求在范围内
        if config_min_date and start_date < self._normalize_date(config_min_date):
            self.logger.warning(f"开始日期 {start_date} 小于配置的最小日期 {config_min_date}，将使用最小日期")
            start_date = self._normalize_date(config_min_date)
            
        if config_max_date and end_date > self._normalize_date(config_max_date):
            self.logger.warning(f"结束日期 {end_date} 大于配置的最大日期 {config_max_date}，将使用最大日期")
            end_date = self._normalize_date(config_max_date)
            
        return start_date, end_date
        
    def _normalize_date(self, date_str: str) -> str:
        """
        标准化日期格式
        
        Args:
            date_str: 日期字符串
            
        Returns:
            str: 标准化的日期字符串 (YYYY-MM-DD)
        """
        if not date_str:
            return datetime.datetime.now().strftime('%Y-%m-%d')
            
        try:
            # 尝试多种日期格式
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%m/%d/%Y']:
                try:
                    dt = datetime.datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
                    
            # 如果所有格式都失败，使用当前日期
            self.logger.warning(f"无法解析日期格式: {date_str}，将使用当前日期")
            return datetime.datetime.now().strftime('%Y-%m-%d')
            
        except Exception as e:
            self.logger.error(f"日期格式化失败: {str(e)}")
            return datetime.datetime.now().strftime('%Y-%m-%d')

    def get_data_batch(self, 
                     params_list: List[Dict[str, Any]], 
                     use_cache: bool = True,
                     **kwargs) -> List[Tuple[bool, Any]]:
        """
        批量获取数据 - 优化点：支持批量操作以提高性能
        
        Args:
            params_list: 请求参数列表
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            List[Tuple[bool, Any]]: 每个参数对应的(是否成功, 数据或错误信息)结果列表
        """
        results = []
        cache_hits = []
        cache_misses = []
        
        if not params_list:
            return results
            
        # 检查数据源是否启用
        if not self.config.get('enabled', True):
            error_msg = f"数据源 {self.source_name} 已禁用"
            self.logger.warning(error_msg)
            return [(False, error_msg) for _ in params_list]
        
        # 步骤1: 生成所有请求的缓存键并尝试从缓存获取
        if use_cache:
            # 创建参数到缓存键的映射
            param_to_key = {}
            cache_keys = []
            
            for params in params_list:
                cache_key = self._generate_cache_key(params, **kwargs)
                param_to_key[id(params)] = cache_key
                cache_keys.append(cache_key)
            
            # 批量从缓存获取数据
            if hasattr(self.cache, 'get_multi'):
                # 如果缓存管理器支持批量获取
                cached_data = self.cache.get_multi(cache_keys)
                
                # 处理缓存结果
                for i, params in enumerate(params_list):
                    cache_key = param_to_key[id(params)]
                    if cache_key in cached_data and cached_data[cache_key] is not None:
                        # 缓存命中
                        results.append((True, cached_data[cache_key]))
                        cache_hits.append(i)
                    else:
                        # 缓存未命中
                        results.append(None)  # 暂存None，稍后填充
                        cache_misses.append(i)
            else:
                # 回退到单个获取
                for i, params in enumerate(params_list):
                    cache_key = param_to_key[id(params)]
                    cached_data = self.cache.get(cache_key)
                    
                    if cached_data is not None:
                        # 缓存命中
                        results.append((True, cached_data))
                        cache_hits.append(i)
                    else:
                        # 缓存未命中
                        results.append(None)  # 暂存None，稍后填充
                        cache_misses.append(i)
        else:
            # 不使用缓存，所有请求都需要获取新数据
            results = [None] * len(params_list)
            cache_misses = list(range(len(params_list)))
        
        # 步骤2: 对于缓存未命中的请求，批量获取新数据
        if cache_misses:
            # 获取缓存未命中的参数
            miss_params = [params_list[i] for i in cache_misses]
            
            # 检查数据源是否支持批量获取
            if hasattr(self, '_fetch_data_batch'):
                # 批量获取数据
                batch_results = self._fetch_data_batch(miss_params, **kwargs)
                
                # 更新结果和缓存
                for i, (success, data) in enumerate(batch_results):
                    orig_idx = cache_misses[i]
                    results[orig_idx] = (success, data)
                    
                    # 如果成功且启用缓存，将数据存入缓存
                    if success and use_cache:
                        params = params_list[orig_idx]
                        cache_key = self._generate_cache_key(params, **kwargs)
                        expiry = self.config_manager.get('cache.default_expiry', 3600)
                        self._async_set_cache(cache_key, data, expiry)
            else:
                # 回退到依次获取
                for i, params in enumerate(miss_params):
                    orig_idx = cache_misses[i]
                    success, data = self._fetch_data(params, **kwargs)
                    results[orig_idx] = (success, data)
                    
                    # 如果成功且启用缓存，将数据存入缓存
                    if success and use_cache:
                        cache_key = self._generate_cache_key(params, **kwargs)
                        expiry = self.config_manager.get('cache.default_expiry', 3600)
                        self._async_set_cache(cache_key, data, expiry)
        
        # 步骤3: 数据验证
        if self.validator:
            for i, result in enumerate(results):
                if result is not None and result[0]:  # 成功获取的数据
                    success, data = result
                    is_valid, validation_msg = self.validate_data(data)
                    
                    if not is_valid:
                        self.logger.error(f"批量请求 #{i} 数据验证失败: {validation_msg}")
                        results[i] = (False, validation_msg)
        
        # 确保所有结果都已填充
        for i, result in enumerate(results):
            if result is None:
                results[i] = (False, "未知错误")
                
        # 更新健康统计
        self.health_stats['total_requests'] += len(params_list)
        self.health_stats['successful_requests'] += sum(1 for success, _ in results if success)
        self.health_stats['failed_requests'] += sum(1 for success, _ in results if not success)
        
        if cache_hits:
            self.logger.debug(f"批量请求中有 {len(cache_hits)}/{len(params_list)} 个缓存命中")
        
        return results 