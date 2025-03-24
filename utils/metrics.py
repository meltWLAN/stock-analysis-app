#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能指标监控模块 - 收集、记录和分析系统性能指标
"""

import time
import threading
import datetime
import logging
import os
import psutil
import socket
import platform
import json
from typing import Dict, List, Any, Optional, Union, Callable
import functools
from dataclasses import dataclass, field, asdict
import statistics

from utils.log_manager import get_logger
from utils.config_manager import get_config_manager

@dataclass
class TimingStats:
    """时间性能统计数据"""
    
    name: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def add_timing(self, elapsed: float):
        """添加一次计时记录"""
        self.calls += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.avg_time = self.total_time / self.calls
        self.last_time = elapsed
        
    def reset(self):
        """重置统计数据"""
        self.calls = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.avg_time = 0.0
        self.last_time = 0.0
        self.start_time = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 添加一些有用的衍生度量
        if self.calls > 0:
            result['calls_per_second'] = self.calls / (time.time() - self.start_time) if time.time() > self.start_time else 0
        return result

@dataclass
class CounterStats:
    """计数器统计数据"""
    
    name: str
    count: int = 0
    start_time: float = field(default_factory=time.time)
    
    def increment(self, value: int = 1):
        """增加计数"""
        self.count += value
        
    def decrement(self, value: int = 1):
        """减少计数"""
        self.count -= value
        
    def reset(self):
        """重置统计数据"""
        self.count = 0
        self.start_time = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['rate'] = self.count / (time.time() - self.start_time) if time.time() > self.start_time else 0
        return result

@dataclass
class GaugeStats:
    """仪表盘统计数据"""
    
    name: str
    value: float = 0.0
    timestamp: float = field(default_factory=time.time)
    min_value: float = float('inf')
    max_value: float = float('-inf')
    
    def set(self, value: float):
        """设置值"""
        self.value = value
        self.timestamp = time.time()
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        
    def reset(self):
        """重置统计数据"""
        self.value = 0.0
        self.timestamp = time.time()
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
@dataclass
class HistogramStats:
    """直方图统计数据"""
    
    name: str
    values: List[float] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0
    
    # 添加控制参数
    max_samples: int = 1000  # 最大样本数，限制内存使用
    use_numpy: bool = True   # 是否使用numpy进行计算（如可用）
    _numpy_available: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """初始化后检查是否可用numpy"""
        try:
            import numpy as np
            self._numpy_available = True
        except ImportError:
            self._numpy_available = False
    
    def add(self, value: float):
        """添加一个值"""
        # 维持固定窗口大小以限制内存使用
        if len(self.values) >= self.max_samples:
            # 移除最早的值并调整统计数据
            old_value = self.values.pop(0)
            self.sum -= old_value
            # 总计数不减少，只移除最早的值
        
        self.values.append(value)
        self.count += 1  # 总计数继续增加
        self.sum += value
        
    def reset(self):
        """重置统计数据"""
        self.values = []
        self.count = 0
        self.sum = 0.0
        
    def percentile(self, p: float) -> float:
        """
        计算百分位数
        
        Args:
            p: 百分位 (0-100)
            
        Returns:
            float: 百分位值
        """
        if not self.values:
            return 0.0
        
        # 使用numpy进行高效计算（如果可用）    
        if self.use_numpy and self._numpy_available:
            try:
                import numpy as np
                return float(np.percentile(self.values, p))
            except:
                pass  # 如果numpy计算失败，回退到标准实现
            
        # 标准实现
        sorted_values = sorted(self.values)
        
        # 计算百分位索引
        k = (len(sorted_values) - 1) * (p / 100.0)
        f = int(k)
        c = int(k) + 1 if k > f else f
        
        # 边界检查
        c = min(c, len(sorted_values) - 1)
            
        # 线性插值
        if f == c:
            return sorted_values[f]
        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'name': self.name,
            'count': self.count,  # 总计数（包括已移除的值）
            'sum': self.sum,
            'current_samples': len(self.values)  # 当前保留的样本数量
        }
        
        if self.values:
            # 使用numpy计算统计量（如果可用且启用）
            if self.use_numpy and self._numpy_available:
                try:
                    import numpy as np
                    values_array = np.array(self.values)
                    result.update({
                        'avg': float(np.mean(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'median': float(np.median(values_array)),
                        'p95': float(np.percentile(values_array, 95)),
                        'p99': float(np.percentile(values_array, 99)),
                        'std_dev': float(np.std(values_array))
                    })
                    return result
                except:
                    pass  # 如果numpy计算失败，回退到标准实现
            
            # 标准实现
            if len(self.values) > 0:
                result.update({
                    'avg': self.sum / len(self.values),
                    'min': min(self.values),
                    'max': max(self.values),
                    'median': statistics.median(self.values),
                    'p95': self.percentile(95),
                    'p99': self.percentile(99)
                })
                # 添加标准差（如果样本数量足够）
                if len(self.values) > 1:
                    try:
                        result['std_dev'] = statistics.stdev(self.values)
                    except:
                        pass
        
        return result

class MetricsManager:
    """指标管理器"""
    
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
        """初始化指标管理器"""
        # 获取配置和日志记录器
        self.config_manager = get_config_manager()
        self.logger = get_logger(__name__)
        
        # 指标存储
        self.timers: Dict[str, TimingStats] = {}
        self.counters: Dict[str, CounterStats] = {}
        self.gauges: Dict[str, GaugeStats] = {}
        self.histograms: Dict[str, HistogramStats] = {}
        
        # 定时收集系统指标的线程
        self.system_metrics_thread = None
        self.system_metrics_running = False
        
        # 启动时注册系统信息
        self._register_system_info()
        
        # 如果配置了监控，启动系统指标收集
        if self.config_manager.get('monitoring.enabled', True):
            self.start_system_metrics_collection()
        
    def timer(self, name: str) -> TimingStats:
        """
        获取计时器
        
        Args:
            name: 计时器名称
            
        Returns:
            TimingStats: 计时器统计数据
        """
        if name not in self.timers:
            self.timers[name] = TimingStats(name)
        return self.timers[name]
    
    def counter(self, name: str) -> CounterStats:
        """
        获取计数器
        
        Args:
            name: 计数器名称
            
        Returns:
            CounterStats: 计数器统计数据
        """
        if name not in self.counters:
            self.counters[name] = CounterStats(name)
        return self.counters[name]
    
    def gauge(self, name: str) -> GaugeStats:
        """
        获取仪表盘
        
        Args:
            name: 仪表盘名称
            
        Returns:
            GaugeStats: 仪表盘统计数据
        """
        if name not in self.gauges:
            self.gauges[name] = GaugeStats(name)
        return self.gauges[name]
    
    def histogram(self, name: str) -> HistogramStats:
        """
        获取直方图
        
        Args:
            name: 直方图名称
            
        Returns:
            HistogramStats: 直方图统计数据
        """
        if name not in self.histograms:
            self.histograms[name] = HistogramStats(name)
        return self.histograms[name]
    
    def time_function(self, name: Optional[str] = None):
        """
        函数执行时间装饰器
        
        Args:
            name: 计时器名称，默认为函数名
            
        Returns:
            函数装饰器
        """
        def decorator(func):
            timer_name = name or f"function.{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # 记录时间
                timer = self.timer(timer_name)
                timer.add_timing(elapsed)
                
                return result
            
            return wrapper
        
        return decorator
    
    def count_calls(self, name: Optional[str] = None, value: int = 1):
        """
        函数调用计数装饰器
        
        Args:
            name: 计数器名称，默认为函数名
            value: 每次调用增加的值
            
        Returns:
            函数装饰器
        """
        def decorator(func):
            counter_name = name or f"calls.{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 增加计数
                counter = self.counter(counter_name)
                counter.increment(value)
                
                return func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def measure_histogram(self, name: Optional[str] = None, value_func: Optional[Callable] = None):
        """
        直方图测量装饰器
        
        Args:
            name: 直方图名称，默认为函数名
            value_func: 从函数结果提取值的函数，默认为返回结果本身
            
        Returns:
            函数装饰器
        """
        def decorator(func):
            hist_name = name or f"histogram.{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # 提取值
                value = result
                if value_func:
                    value = value_func(result)
                    
                # 只记录数值类型
                if isinstance(value, (int, float)):
                    hist = self.histogram(hist_name)
                    hist.add(float(value))
                
                return result
            
            return wrapper
        
        return decorator
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标数据
        
        Returns:
            Dict: 所有指标的字典表示
        """
        return {
            'timers': {name: timer.to_dict() for name, timer in self.timers.items()},
            'counters': {name: counter.to_dict() for name, counter in self.counters.items()},
            'gauges': {name: gauge.to_dict() for name, gauge in self.gauges.items()},
            'histograms': {name: hist.to_dict() for name, hist in self.histograms.items()}
        }
    
    def reset_all(self):
        """重置所有指标"""
        for timer in self.timers.values():
            timer.reset()
            
        for counter in self.counters.values():
            counter.reset()
            
        for gauge in self.gauges.values():
            gauge.reset()
            
        for hist in self.histograms.values():
            hist.reset()
            
        self.logger.info("所有指标已重置")
    
    def start_system_metrics_collection(self):
        """启动系统指标收集"""
        if self.system_metrics_thread and self.system_metrics_thread.is_alive():
            self.logger.warning("系统指标收集已在运行中")
            return
            
        self.system_metrics_running = True
        self.system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics_loop,
            daemon=True
        )
        self.system_metrics_thread.start()
        
        self.logger.info("已启动系统指标收集")
    
    def stop_system_metrics_collection(self):
        """停止系统指标收集"""
        if not self.system_metrics_thread or not self.system_metrics_thread.is_alive():
            self.logger.warning("系统指标收集未运行")
            return
            
        self.system_metrics_running = False
        self.system_metrics_thread.join(timeout=1.0)
        
        if self.system_metrics_thread.is_alive():
            self.logger.warning("系统指标收集线程无法正常停止")
        else:
            self.logger.info("已停止系统指标收集")
    
    def _collect_system_metrics_loop(self):
        """系统指标收集循环"""
        interval = self.config_manager.get('monitoring.collect_interval', 60)
        
        while self.system_metrics_running:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"收集系统指标时出错: {str(e)}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        # CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.gauge('system.cpu.percent').set(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.gauge('system.memory.percent').set(memory.percent)
        self.gauge('system.memory.used').set(memory.used / (1024 * 1024))  # MB
        self.gauge('system.memory.available').set(memory.available / (1024 * 1024))  # MB
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        self.gauge('system.disk.percent').set(disk.percent)
        self.gauge('system.disk.used').set(disk.used / (1024 * 1024 * 1024))  # GB
        self.gauge('system.disk.free').set(disk.free / (1024 * 1024 * 1024))  # GB
        
        # 网络 IO 统计
        net_io = psutil.net_io_counters()
        self.gauge('system.network.bytes_sent').set(net_io.bytes_sent / (1024 * 1024))  # MB
        self.gauge('system.network.bytes_recv').set(net_io.bytes_recv / (1024 * 1024))  # MB
        
        # 进程信息
        process = psutil.Process(os.getpid())
        self.gauge('system.process.cpu_percent').set(process.cpu_percent(interval=0.1))
        self.gauge('system.process.memory_percent').set(process.memory_percent())
        self.gauge('system.process.memory_rss').set(process.memory_info().rss / (1024 * 1024))  # MB
        self.gauge('system.process.threads').set(process.num_threads())
        
        # 系统负载（仅在类Unix系统上可用）
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()
            self.gauge('system.load.1min').set(load_avg[0])
            self.gauge('system.load.5min').set(load_avg[1])
            self.gauge('system.load.15min').set(load_avg[2])
    
    def _register_system_info(self):
        """注册系统信息"""
        # 系统信息
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'hostname': socket.gethostname(),
            'cpu_count': psutil.cpu_count(),
            'total_memory': psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            'start_time': datetime.datetime.now().isoformat()
        }
        
        # 添加应用信息
        app_config = self.config_manager.get_app_config()
        system_info.update({
            'app_name': app_config.get('name', 'StockAnalyzer'),
            'app_version': app_config.get('version', '1.0.0')
        })
        
        # 将系统信息存为仪表盘值
        for key, value in system_info.items():
            if isinstance(value, (int, float)):
                self.gauge(f"info.{key}").set(value)
            else:
                # 对于非数值型信息，我们记录到日志
                self.logger.info(f"系统信息 {key}: {value}")
        
        self.logger.info(f"系统信息已注册: {json.dumps(system_info, indent=2)}")

# 创建默认指标管理器实例
metrics_manager = MetricsManager()

def get_metrics_manager() -> MetricsManager:
    """获取指标管理器实例"""
    return metrics_manager 