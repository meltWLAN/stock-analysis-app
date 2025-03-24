#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事件管理器 - 提供事件驱动架构的核心组件
"""

import logging
import threading
import queue
import time
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional, Set, Union
from datetime import datetime
import asyncio
import concurrent.futures

class EventPriority(Enum):
    """事件优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Event:
    """事件对象"""
    name: str  # 事件名称
    data: Any  # 事件数据
    source: Optional[str] = None  # 事件源
    timestamp: datetime = None  # 事件时间戳
    id: str = None  # 事件唯一ID
    priority: EventPriority = EventPriority.NORMAL  # 事件优先级
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.id is None:
            self.id = str(uuid.uuid4())

class EventManager:
    """事件管理器 - 管理事件注册、发布和处理"""
    
    def __init__(self, max_workers=5):
        """
        初始化事件管理器
        
        Args:
            max_workers: 最大工作线程数，用于异步处理事件
        """
        self.logger = logging.getLogger(__name__)
        
        # 事件处理器映射表 {事件名称: [处理器, ...]}
        self.handlers: Dict[str, List[Callable]] = {}
        
        # 异步事件队列 - 使用优先级队列
        self.event_queue = queue.PriorityQueue()
        
        # 事件批处理队列 (同一类型的事件会被批处理)
        self.batch_queues: Dict[str, List[Event]] = {}
        self.batch_config: Dict[str, Dict[str, Any]] = {}  # 批处理配置
        
        # 预处理器映射表 {事件名称: 预处理函数}
        self.preprocessors: Dict[str, Callable] = {}
        
        # 线程池
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 事件处理线程
        self.processing_thread = None
        self._running = False
        
        # 批处理线程
        self.batch_thread = None
        self._batch_running = False
        
        # 处理中的事件ID
        self.processing_events: Set[str] = set()
        
        # 事件统计信息
        self.stats = {
            'total_published': 0,
            'total_processed': 0,
            'total_failed': 0,
            'total_batched': 0,
            'handlers_count': 0,
            'queue_size': 0,
            'batch_queue_size': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info(f"事件管理器初始化完成，工作线程数: {max_workers}")
    
    def register(self, event_name: str, handler: Callable, priority: EventPriority = EventPriority.NORMAL):
        """
        注册事件处理器
        
        Args:
            event_name: 事件名称
            handler: 事件处理函数，接受Event对象作为参数
            priority: 处理器优先级，影响调用顺序
        """
        if event_name not in self.handlers:
            self.handlers[event_name] = []
            
        # 将处理器添加到列表，包含优先级信息
        self.handlers[event_name].append((handler, priority))
        
        # 按优先级排序
        self.handlers[event_name].sort(key=lambda x: x[1].value, reverse=True)
        
        self.stats['handlers_count'] += 1
        self.logger.debug(f"已注册事件处理器: {event_name} -> {handler.__name__}")
    
    def register_preprocessor(self, event_name: str, preprocessor: Callable):
        """
        注册事件预处理器
        
        Args:
            event_name: 事件名称
            preprocessor: 预处理函数，接受Event对象作为参数并返回处理后的Event或None(忽略事件)
        """
        self.preprocessors[event_name] = preprocessor
        self.logger.debug(f"已注册事件预处理器: {event_name} -> {preprocessor.__name__}")
    
    def configure_batch(self, event_name: str, max_batch_size: int = 10, 
                       max_wait_time: float = 0.5, processor: Callable = None):
        """
        配置事件批处理
        
        Args:
            event_name: 事件名称
            max_batch_size: 最大批处理大小
            max_wait_time: 最大等待时间（秒）
            processor: 批处理函数，接受事件列表作为参数并返回处理结果
        """
        self.batch_config[event_name] = {
            'max_size': max_batch_size,
            'max_wait': max_wait_time,
            'processor': processor,
            'last_flush': time.time()
        }
        
        # 创建批处理队列
        if event_name not in self.batch_queues:
            self.batch_queues[event_name] = []
            
        # 确保批处理线程在运行
        self._ensure_batch_thread()
        
        self.logger.debug(f"已配置事件批处理: {event_name}, 最大批量: {max_batch_size}, 最大等待: {max_wait_time}秒")
    
    def unregister(self, event_name: str, handler: Callable = None):
        """
        注销事件处理器
        
        Args:
            event_name: 事件名称
            handler: 要注销的处理器函数，为None时注销该事件的所有处理器
        """
        if event_name not in self.handlers:
            return
            
        if handler is None:
            # 注销所有处理器
            handler_count = len(self.handlers[event_name])
            self.handlers.pop(event_name)
            self.stats['handlers_count'] -= handler_count
            self.logger.debug(f"已注销事件 {event_name} 的所有处理器")
        else:
            # 注销特定处理器
            original_count = len(self.handlers[event_name])
            self.handlers[event_name] = [
                (h, p) for h, p in self.handlers[event_name] if h != handler
            ]
            new_count = len(self.handlers[event_name])
            self.stats['handlers_count'] -= (original_count - new_count)
            
            # 如果没有处理器了，删除该事件名称
            if not self.handlers[event_name]:
                self.handlers.pop(event_name)
                
            self.logger.debug(f"已注销事件处理器: {event_name} -> {handler.__name__}")
    
    def publish(self, event: Union[Event, str], data: Any = None, source: str = None, 
                priority: EventPriority = EventPriority.NORMAL, 
                sync: bool = False,
                batch: bool = False) -> Optional[str]:
        """
        发布事件
        
        Args:
            event: 事件对象或事件名称
            data: 当event为事件名称时，表示事件数据
            source: 当event为事件名称时，表示事件源
            priority: 当event为事件名称时，表示事件优先级
            sync: 是否同步处理事件
            batch: 是否使用批处理（如果配置了）
            
        Returns:
            str: 事件ID（仅在异步模式下返回），用于后续查询事件状态
        """
        # 如果传入的是事件名称，创建事件对象
        if isinstance(event, str):
            event_name = event
            event = Event(
                name=event_name,
                data=data,
                source=source,
                priority=priority
            )
        else:
            event_name = event.name
        
        # 应用预处理器
        if event_name in self.preprocessors:
            try:
                preprocessed_event = self.preprocessors[event_name](event)
                if preprocessed_event is None:
                    # 预处理器决定忽略此事件
                    self.logger.debug(f"事件 {event_name} 被预处理器忽略")
                    return None
                event = preprocessed_event
            except Exception as e:
                self.logger.error(f"事件预处理错误: {e}")
                # 继续处理原始事件
        
        self.stats['total_published'] += 1
        
        # 检查是否应该批处理
        if batch and event_name in self.batch_config:
            return self._add_to_batch(event)
        
        # 如果同步处理，直接调用处理函数
        if sync:
            self._process_event(event)
            return None
            
        # 否则，将事件放入队列
        try:
            # 使用优先级作为队列优先级
            self.event_queue.put((-event.priority.value, event))
            self.stats['queue_size'] = self.event_queue.qsize()
            self.logger.debug(f"事件已加入队列: {event.name} [{event.id}]")
            
            # 确保处理线程在运行
            self._ensure_processing_thread()
            
            return event.id
            
        except Exception as e:
            self.logger.error(f"发布事件失败: {e}")
            return None
    
    def _add_to_batch(self, event: Event) -> str:
        """
        添加事件到批处理队列
        
        Args:
            event: 事件对象
            
        Returns:
            str: 事件ID
        """
        event_name = event.name
        
        with threading.Lock():
            # 添加到批处理队列
            self.batch_queues[event_name].append(event)
            self.stats['batch_queue_size'] = sum(len(q) for q in self.batch_queues.values())
            
            # 检查是否达到最大批处理大小
            batch_config = self.batch_config[event_name]
            if len(self.batch_queues[event_name]) >= batch_config['max_size']:
                # 触发立即处理
                self._process_batch(event_name)
        
        return event.id
    
    def _ensure_processing_thread(self):
        """确保事件处理线程在运行"""
        if not self._running or (self.processing_thread and not self.processing_thread.is_alive()):
            self._running = True
            self.processing_thread = threading.Thread(target=self._event_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.logger.debug("已启动事件处理线程")
    
    def _ensure_batch_thread(self):
        """确保批处理线程在运行"""
        if not self._batch_running or (self.batch_thread and not self.batch_thread.is_alive()):
            self._batch_running = True
            self.batch_thread = threading.Thread(target=self._batch_processing_loop)
            self.batch_thread.daemon = True
            self.batch_thread.start()
            self.logger.debug("已启动事件批处理线程")
    
    def _event_processing_loop(self):
        """事件处理循环"""
        self.logger.info("事件处理循环开始运行")
        
        while self._running:
            try:
                # 从队列获取事件，等待1秒
                try:
                    _, event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # 将事件ID添加到处理集合
                self.processing_events.add(event.id)
                
                # 使用线程池异步处理事件
                self.thread_pool.submit(self._process_event, event)
                
                # 更新队列大小统计
                self.stats['queue_size'] = self.event_queue.qsize()
                
            except Exception as e:
                self.logger.error(f"事件处理循环异常: {e}")
                time.sleep(1)  # 出错时暂停一下
    
    def _batch_processing_loop(self):
        """批处理循环"""
        self.logger.info("事件批处理循环开始运行")
        
        while self._batch_running:
            try:
                # 检查所有批处理队列
                for event_name, events in list(self.batch_queues.items()):
                    if not events:
                        continue
                        
                    batch_config = self.batch_config.get(event_name)
                    if not batch_config:
                        continue
                        
                    # 检查是否达到条件
                    current_time = time.time()
                    time_condition = (current_time - batch_config['last_flush']) > batch_config['max_wait']
                    size_condition = len(events) >= batch_config['max_size']
                    
                    if time_condition or size_condition:
                        self._process_batch(event_name)
                
                # 每100ms检查一次
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"事件批处理循环异常: {e}")
                time.sleep(1)  # 出错时暂停一下
    
    def _process_batch(self, event_name: str):
        """
        处理事件批次
        
        Args:
            event_name: 事件名称
        """
        with threading.Lock():
            # 获取当前批次的所有事件
            events = self.batch_queues[event_name]
            if not events:
                return
                
            # 清空队列
            self.batch_queues[event_name] = []
            
            # 更新统计
            batch_size = len(events)
            self.stats['total_batched'] += batch_size
            self.stats['batch_queue_size'] = sum(len(q) for q in self.batch_queues.values())
            
            # 更新最后刷新时间
            self.batch_config[event_name]['last_flush'] = time.time()
        
        self.logger.debug(f"开始处理事件批次: {event_name} ({batch_size}个事件)")
        
        try:
            # 检查是否有自定义批处理器
            batch_processor = self.batch_config[event_name].get('processor')
            
            if batch_processor:
                # 使用自定义批处理器
                try:
                    batch_processor(events)
                    
                    # 标记所有事件为已处理
                    for event in events:
                        self._finish_event(event.id, success=True)
                        
                except Exception as e:
                    self.logger.error(f"批处理器执行失败: {e}")
                    
                    # 标记所有事件为处理失败
                    for event in events:
                        self._finish_event(event.id, success=False)
            else:
                # 没有自定义处理器，使用标准处理
                for event in events:
                    self._process_event(event)
                    
        except Exception as e:
            self.logger.error(f"批处理过程出错: {e}")
            
            # 标记所有未处理的事件为失败
            for event in events:
                if event.id in self.processing_events:
                    self._finish_event(event.id, success=False)
    
    def _process_event(self, event: Event):
        """
        处理单个事件
        
        Args:
            event: 事件对象
        """
        self.logger.debug(f"开始处理事件: {event.name} [{event.id}]")
        
        # 获取该事件的所有处理器
        event_handlers = self.handlers.get(event.name, [])
        
        if not event_handlers:
            self.logger.debug(f"事件 {event.name} 没有注册的处理器")
            self._finish_event(event.id, success=True)
            return
            
        try:
            # 依次调用处理器
            for handler, _ in event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"事件处理器 {handler.__name__} 执行失败: {e}")
                    self.stats['total_failed'] += 1
            
            self._finish_event(event.id, success=True)
            
        except Exception as e:
            self.logger.error(f"处理事件 {event.name} 时发生异常: {e}")
            self._finish_event(event.id, success=False)
    
    def _finish_event(self, event_id: str, success: bool):
        """
        完成事件处理
        
        Args:
            event_id: 事件ID
            success: 处理是否成功
        """
        # 从处理集合中移除
        if event_id in self.processing_events:
            self.processing_events.remove(event_id)
            
        # 更新统计信息
        self.stats['total_processed'] += 1
        if not success:
            self.stats['total_failed'] += 1
    
    def wait_for_event(self, event_id: str, timeout: float = None) -> bool:
        """
        等待事件处理完成
        
        Args:
            event_id: 事件ID
            timeout: 超时时间（秒）
            
        Returns:
            bool: 事件是否已处理完成
        """
        start_time = time.time()
        
        while event_id in self.processing_events:
            # 检查超时
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            time.sleep(0.1)
            
        return True
    
    def stop(self):
        """停止事件管理器"""
        self._running = False
        self._batch_running = False
        
        # 等待处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            
        # 等待批处理线程结束
        if self.batch_thread and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=2)
            
        # 关闭线程池
        self.thread_pool.shutdown(wait=False)
        
        self.logger.info("事件管理器已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取事件统计信息
        
        Returns:
            Dict: 统计信息
        """
        # 更新队列大小
        self.stats['queue_size'] = self.event_queue.qsize()
        self.stats['batch_queue_size'] = sum(len(q) for q in self.batch_queues.values())
        
        # 计算每类事件的批处理队列大小
        batch_queue_sizes = {name: len(queue) for name, queue in self.batch_queues.items()}
        
        # 添加运行时间
        uptime = datetime.now() - self.stats['start_time']
        self.stats['uptime_seconds'] = uptime.total_seconds()
        
        # 添加每秒处理事件数
        events_per_second = self.stats['total_processed'] / max(1, uptime.total_seconds())
        self.stats['events_per_second'] = round(events_per_second, 2)
        
        # 添加事件处理器分布
        handler_counts = {name: len(handlers) for name, handlers in self.handlers.items()}
        
        # 构建完整统计信息
        stats = dict(self.stats)
        stats['event_handlers'] = handler_counts
        stats['batch_queues'] = batch_queue_sizes
        stats['preprocessors'] = list(self.preprocessors.keys())
        
        return stats


# 创建默认事件管理器实例
default_event_manager = EventManager()

def get_event_manager():
    """获取默认事件管理器实例"""
    return default_event_manager


# 便捷函数，用于其他模块
def register_handler(event_name, handler, priority=EventPriority.NORMAL):
    """注册事件处理器"""
    default_event_manager.register(event_name, handler, priority)

def publish_event(event, data=None, source=None, priority=EventPriority.NORMAL, sync=False):
    """发布事件"""
    return default_event_manager.publish(event, data, source, priority, sync)

def wait_for_event(event_id, timeout=None):
    """等待事件处理完成"""
    return default_event_manager.wait_for_event(event_id, timeout) 