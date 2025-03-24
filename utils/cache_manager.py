#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
缓存管理模块 - 负责管理系统中的数据缓存
"""

import os
import logging
import json
import pickle
import time
from threading import Lock
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

class CacheManager:
    """
    缓存管理器 - 单例模式实现
    负责管理系统中的各种缓存数据
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
    
    def __init__(self, cache_dir="./cache", max_age=24*60*60):
        """
        初始化缓存管理器
        
        参数:
            cache_dir (str): 缓存目录
            max_age (int): 缓存最大有效期（秒）
        """
        self.cache_dir = cache_dir
        self.max_age = max_age
        self.memory_cache = {}
        self.memory_cache_timestamp = {}
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化时清理过期缓存
        self.clean_expired_cache()
        
        logger.info(f"缓存管理器初始化完成，缓存目录: {cache_dir}")
    
    def get(self, key, default=None, use_memory=True):
        """
        获取缓存数据
        
        参数:
            key (str): 缓存键名
            default: 默认返回值
            use_memory (bool): 是否优先使用内存缓存
            
        返回:
            缓存数据或默认值
        """
        # 检查内存缓存
        if use_memory and key in self.memory_cache:
            timestamp = self.memory_cache_timestamp.get(key, 0)
            if time.time() - timestamp <= self.max_age:
                logger.debug(f"从内存缓存获取数据: {key}")
                return self.memory_cache[key]
            else:
                # 内存缓存过期，删除
                del self.memory_cache[key]
                if key in self.memory_cache_timestamp:
                    del self.memory_cache_timestamp[key]
        
        # 检查文件缓存
        cache_path = os.path.join(self.cache_dir, f"{key}.pickle")
        if os.path.exists(cache_path):
            # 检查文件缓存是否过期
            file_time = os.path.getmtime(cache_path)
            if time.time() - file_time <= self.max_age:
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # 更新内存缓存
                    if use_memory:
                        self.memory_cache[key] = data
                        self.memory_cache_timestamp[key] = time.time()
                    
                    logger.debug(f"从文件缓存获取数据: {key}")
                    return data
                except Exception as e:
                    logger.warning(f"读取缓存文件出错: {e}")
                    # 删除损坏的缓存文件
                    try:
                        os.remove(cache_path)
                    except:
                        pass
            else:
                # 删除过期缓存文件
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        return default
    
    def set(self, key, value, use_memory=True, use_file=True):
        """
        设置缓存数据
        
        参数:
            key (str): 缓存键名
            value: 缓存数据
            use_memory (bool): 是否使用内存缓存
            use_file (bool): 是否使用文件缓存
            
        返回:
            bool: 设置是否成功
        """
        # 更新内存缓存
        if use_memory:
            self.memory_cache[key] = value
            self.memory_cache_timestamp[key] = time.time()
        
        # 更新文件缓存
        if use_file:
            cache_path = os.path.join(self.cache_dir, f"{key}.pickle")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                logger.debug(f"更新文件缓存: {key}")
                return True
            except Exception as e:
                logger.warning(f"写入缓存文件出错: {e}")
                return False
        
        return True
    
    def delete(self, key):
        """
        删除缓存数据
        
        参数:
            key (str): 缓存键名
            
        返回:
            bool: 删除是否成功
        """
        # 删除内存缓存
        if key in self.memory_cache:
            del self.memory_cache[key]
        if key in self.memory_cache_timestamp:
            del self.memory_cache_timestamp[key]
        
        # 删除文件缓存
        cache_path = os.path.join(self.cache_dir, f"{key}.pickle")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.debug(f"删除文件缓存: {key}")
                return True
            except Exception as e:
                logger.warning(f"删除缓存文件出错: {e}")
                return False
        
        return True
    
    def clean_expired_cache(self):
        """
        清理过期缓存
        
        返回:
            int: 清理的缓存数量
        """
        # 清理内存缓存
        current_time = time.time()
        expired_keys = []
        for key, timestamp in self.memory_cache_timestamp.items():
            if current_time - timestamp > self.max_age:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.memory_cache_timestamp:
                del self.memory_cache_timestamp[key]
        
        memory_count = len(expired_keys)
        
        # 清理文件缓存
        file_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pickle'):
                file_path = os.path.join(self.cache_dir, filename)
                file_time = os.path.getmtime(file_path)
                if current_time - file_time > self.max_age:
                    try:
                        os.remove(file_path)
                        file_count += 1
                    except:
                        pass
        
        total_count = memory_count + file_count
        if total_count > 0:
            logger.info(f"清理过期缓存: 内存 {memory_count} 条，文件 {file_count} 条")
        
        return total_count
    
    def clear_all(self):
        """
        清空所有缓存
        
        返回:
            bool: 操作是否成功
        """
        # 清空内存缓存
        self.memory_cache.clear()
        self.memory_cache_timestamp.clear()
        
        # 清空文件缓存
        success = True
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pickle'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except:
                    success = False
        
        logger.info("清空所有缓存")
        return success
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        
        返回:
            dict: 缓存统计信息
        """
        # 内存缓存统计
        memory_count = len(self.memory_cache)
        
        # 文件缓存统计
        file_count = 0
        oldest_file = None
        newest_file = None
        now = time.time()
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pickle'):
                file_count += 1
                file_path = os.path.join(self.cache_dir, filename)
                file_time = os.path.getmtime(file_path)
                
                if oldest_file is None or file_time < oldest_file[1]:
                    oldest_file = (filename, file_time)
                
                if newest_file is None or file_time > newest_file[1]:
                    newest_file = (filename, file_time)
        
        stats = {
            'memory_cache_count': memory_count,
            'file_cache_count': file_count,
            'oldest_file': oldest_file[0] if oldest_file else None,
            'oldest_file_age': now - oldest_file[1] if oldest_file else None,
            'newest_file': newest_file[0] if newest_file else None,
            'newest_file_age': now - newest_file[1] if newest_file else None,
            'cache_dir': self.cache_dir,
            'max_age': self.max_age
        }
        
        return stats 