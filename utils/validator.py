#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据验证器 - 提供数据验证功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import datetime
import re
from abc import ABC, abstractmethod

from utils.log_manager import get_logger

class ValidationRule(ABC):
    """验证规则基类"""
    
    def __init__(self, field: Optional[str] = None, message: Optional[str] = None):
        """
        初始化验证规则
        
        Args:
            field: 要验证的字段名
            message: 验证失败时的错误消息
        """
        self.field = field
        self.message = message
        
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证数据
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        pass

class NotEmptyRule(ValidationRule):
    """非空验证规则"""
    
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证数据非空
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if self.field:
            # 验证特定字段
            if not hasattr(data, self.field) and not isinstance(data, dict):
                return False, f"数据不包含字段 '{self.field}'"
                
            if isinstance(data, dict):
                value = data.get(self.field)
            else:
                value = getattr(data, self.field)
                
            is_empty = (value is None or 
                      (isinstance(value, (str, list, dict, tuple)) and len(value) == 0) or
                      (isinstance(value, pd.DataFrame) and value.empty))
                
            if is_empty:
                return False, self.message or f"字段 '{self.field}' 不能为空"
        else:
            # 验证整个数据对象
            is_empty = (data is None or 
                      (isinstance(data, (str, list, dict, tuple)) and len(data) == 0) or
                      (isinstance(data, pd.DataFrame) and data.empty))
                
            if is_empty:
                return False, self.message or "数据不能为空"
                
        return True, ""

class DataFrameColumnRule(ValidationRule):
    """DataFrame列验证规则"""
    
    def __init__(self, columns: List[str], message: Optional[str] = None, require_all: bool = True):
        """
        初始化DataFrame列验证规则
        
        Args:
            columns: 必须存在的列名列表
            message: 验证失败时的错误消息
            require_all: 是否要求所有列都存在
        """
        super().__init__(None, message)
        self.columns = columns
        self.require_all = require_all
        
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证DataFrame包含指定列
        
        Args:
            data: 要验证的DataFrame
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not isinstance(data, pd.DataFrame):
            return False, self.message or "数据类型必须是DataFrame"
            
        if data.empty:
            return False, self.message or "DataFrame不能为空"
            
        # 检查必要的列
        if self.require_all:
            # 所有指定的列都必须存在
            missing_columns = [col for col in self.columns if col not in data.columns]
            if missing_columns:
                return False, self.message or f"DataFrame缺少必要的列: {', '.join(missing_columns)}"
        else:
            # 至少有一个指定的列存在
            if not any(col in data.columns for col in self.columns):
                return False, self.message or f"DataFrame必须至少包含以下列之一: {', '.join(self.columns)}"
                
        return True, ""

class DateRangeRule(ValidationRule):
    """日期范围验证规则"""
    
    def __init__(self, field: str, min_date: Optional[str] = None, max_date: Optional[str] = None, 
                message: Optional[str] = None, date_format: str = '%Y-%m-%d'):
        """
        初始化日期范围验证规则
        
        Args:
            field: 日期字段名
            min_date: 最小日期 (YYYY-MM-DD)
            max_date: 最大日期 (YYYY-MM-DD)
            message: 验证失败时的错误消息
            date_format: 日期格式
        """
        super().__init__(field, message)
        self.min_date = min_date
        self.max_date = max_date
        self.date_format = date_format
        
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证日期在指定范围内
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        # 获取字段值
        if isinstance(data, dict):
            if self.field not in data:
                return False, f"数据不包含字段 '{self.field}'"
            value = data[self.field]
        elif hasattr(data, self.field):
            value = getattr(data, self.field)
        elif isinstance(data, pd.DataFrame):
            if self.field not in data.columns:
                return False, f"DataFrame不包含列 '{self.field}'"
            # 对于DataFrame，我们检查所有日期值
            return self._validate_dataframe(data)
        else:
            return False, f"无法从数据中获取字段 '{self.field}'"
            
        # 验证单个日期值
        return self._validate_date(value)
    
    def _validate_date(self, date_value: Any) -> Tuple[bool, str]:
        """验证单个日期值"""
        if date_value is None:
            return False, f"日期字段 '{self.field}' 不能为空"
            
        # 转换为日期对象
        if isinstance(date_value, str):
            try:
                date_obj = datetime.datetime.strptime(date_value, self.date_format).date()
            except ValueError:
                return False, f"日期格式无效: {date_value}，应为 {self.date_format}"
        elif isinstance(date_value, datetime.datetime):
            date_obj = date_value.date()
        elif isinstance(date_value, datetime.date):
            date_obj = date_value
        else:
            return False, f"无效的日期类型: {type(date_value)}"
            
        # 检查最小日期
        if self.min_date:
            min_date_obj = datetime.datetime.strptime(self.min_date, self.date_format).date()
            if date_obj < min_date_obj:
                return False, self.message or f"日期 {date_obj} 小于最小允许日期 {self.min_date}"
                
        # 检查最大日期
        if self.max_date:
            max_date_obj = datetime.datetime.strptime(self.max_date, self.date_format).date()
            if date_obj > max_date_obj:
                return False, self.message or f"日期 {date_obj} 大于最大允许日期 {self.max_date}"
                
        return True, ""
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """验证DataFrame中的日期列"""
        # 获取日期列
        date_col = df[self.field]
        
        # 检查是否有空值
        if date_col.isna().any():
            return False, f"日期列 '{self.field}' 包含空值"
            
        # 对于DataFrame，我们分别检查最小和最大日期
        if isinstance(date_col.iloc[0], str):
            try:
                date_col = pd.to_datetime(date_col, format=self.date_format)
            except ValueError:
                return False, f"日期列包含无效的日期格式，应为 {self.date_format}"
                
        min_val = date_col.min()
        max_val = date_col.max()
        
        # 转换为日期对象
        if isinstance(min_val, pd.Timestamp):
            min_val = min_val.date()
        if isinstance(max_val, pd.Timestamp):
            max_val = max_val.date()
            
        # 检查最小日期
        if self.min_date:
            min_date_obj = datetime.datetime.strptime(self.min_date, self.date_format).date()
            if min_val < min_date_obj:
                return False, self.message or f"数据包含小于最小允许日期的日期: {min_val} < {self.min_date}"
                
        # 检查最大日期
        if self.max_date:
            max_date_obj = datetime.datetime.strptime(self.max_date, self.date_format).date()
            if max_val > max_date_obj:
                return False, self.message or f"数据包含大于最大允许日期的日期: {max_val} > {self.max_date}"
                
        return True, ""

class NumericRangeRule(ValidationRule):
    """数值范围验证规则"""
    
    def __init__(self, field: str, min_value: Optional[float] = None, max_value: Optional[float] = None, 
                message: Optional[str] = None):
        """
        初始化数值范围验证规则
        
        Args:
            field: 数值字段名
            min_value: 最小值
            max_value: 最大值
            message: 验证失败时的错误消息
        """
        super().__init__(field, message)
        self.min_value = min_value
        self.max_value = max_value
        
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证数值在指定范围内
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        # 获取字段值
        if isinstance(data, dict):
            if self.field not in data:
                return False, f"数据不包含字段 '{self.field}'"
            value = data[self.field]
        elif hasattr(data, self.field):
            value = getattr(data, self.field)
        elif isinstance(data, pd.DataFrame):
            if self.field not in data.columns:
                return False, f"DataFrame不包含列 '{self.field}'"
            # 对于DataFrame，我们检查所有数值
            return self._validate_dataframe(data)
        else:
            return False, f"无法从数据中获取字段 '{self.field}'"
            
        # 验证单个数值
        return self._validate_number(value)
    
    def _validate_number(self, value: Any) -> Tuple[bool, str]:
        """验证单个数值"""
        if value is None:
            return False, f"数值字段 '{self.field}' 不能为空"
            
        # 检查是否为数值类型
        if not isinstance(value, (int, float, np.number)):
            return False, f"字段 '{self.field}' 必须是数值类型，而不是 {type(value)}"
            
        # 检查最小值
        if self.min_value is not None and value < self.min_value:
            return False, self.message or f"值 {value} 小于最小允许值 {self.min_value}"
            
        # 检查最大值
        if self.max_value is not None and value > self.max_value:
            return False, self.message or f"值 {value} 大于最大允许值 {self.max_value}"
            
        return True, ""
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """验证DataFrame中的数值列"""
        # 获取数值列
        num_col = df[self.field]
        
        # 检查是否有空值
        if num_col.isna().any():
            return False, f"数值列 '{self.field}' 包含空值"
            
        # 检查是否全是数值
        if not pd.api.types.is_numeric_dtype(num_col):
            return False, f"列 '{self.field}' 必须是数值类型"
            
        # 检查最小值
        if self.min_value is not None:
            min_val = num_col.min()
            if min_val < self.min_value:
                return False, self.message or f"数据包含小于最小允许值的数值: {min_val} < {self.min_value}"
                
        # 检查最大值
        if self.max_value is not None:
            max_val = num_col.max()
            if max_val > self.max_value:
                return False, self.message or f"数据包含大于最大允许值的数值: {max_val} > {self.max_value}"
                
        return True, ""

class PatternRule(ValidationRule):
    """正则表达式模式验证规则"""
    
    def __init__(self, field: str, pattern: str, message: Optional[str] = None):
        """
        初始化正则表达式验证规则
        
        Args:
            field: 字段名
            pattern: 正则表达式模式
            message: 验证失败时的错误消息
        """
        super().__init__(field, message)
        self.pattern = pattern
        self.regex = re.compile(pattern)
        
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证字段值符合正则表达式模式
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        # 获取字段值
        if isinstance(data, dict):
            if self.field not in data:
                return False, f"数据不包含字段 '{self.field}'"
            value = data[self.field]
        elif hasattr(data, self.field):
            value = getattr(data, self.field)
        else:
            return False, f"无法从数据中获取字段 '{self.field}'"
            
        # 检查值是否为字符串
        if not isinstance(value, str):
            return False, f"字段 '{self.field}' 必须是字符串"
            
        # 检查是否匹配模式
        if not self.regex.match(value):
            return False, self.message or f"字段 '{self.field}' 值 '{value}' 不符合模式 '{self.pattern}'"
            
        return True, ""

class CustomRule(ValidationRule):
    """自定义验证规则"""
    
    def __init__(self, validator: Callable[[Any], Tuple[bool, str]], field: Optional[str] = None, 
                message: Optional[str] = None):
        """
        初始化自定义验证规则
        
        Args:
            validator: 自定义验证函数，接收数据参数，返回(is_valid, message)元组
            field: 字段名（可选）
            message: 验证失败时的错误消息（可选）
        """
        super().__init__(field, message)
        self.validator_func = validator
        
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        使用自定义函数验证数据
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        # 如果指定了字段，则验证该字段的值
        if self.field:
            if isinstance(data, dict):
                if self.field not in data:
                    return False, f"数据不包含字段 '{self.field}'"
                value = data[self.field]
            elif hasattr(data, self.field):
                value = getattr(data, self.field)
            else:
                return False, f"无法从数据中获取字段 '{self.field}'"
                
            # 使用自定义函数验证字段值
            is_valid, message = self.validator_func(value)
        else:
            # 验证整个数据对象
            is_valid, message = self.validator_func(data)
            
        # 如果验证失败且提供了默认消息，则使用默认消息
        if not is_valid and self.message:
            message = self.message
            
        return is_valid, message

class Validator:
    """数据验证器"""
    
    def __init__(self, name: str = "default"):
        """
        初始化验证器
        
        Args:
            name: 验证器名称
        """
        self.name = name
        self.rules: List[ValidationRule] = []
        self.logger = get_logger(__name__)
        
    def add_rule(self, rule: ValidationRule) -> 'Validator':
        """
        添加验证规则
        
        Args:
            rule: 验证规则
            
        Returns:
            Validator: 验证器自身，用于链式调用
        """
        self.rules.append(rule)
        return self
        
    def validate(self, data: Any) -> Tuple[bool, str]:
        """
        验证数据
        
        Args:
            data: 要验证的数据
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not self.rules:
            self.logger.warning(f"验证器 '{self.name}' 没有定义规则")
            return True, ""
            
        for rule in self.rules:
            is_valid, message = rule.validate(data)
            if not is_valid:
                self.logger.debug(f"验证失败: {message}")
                return False, message
                
        return True, ""
        
    def validate_all(self, data: Any) -> Dict[str, str]:
        """
        验证所有规则并返回所有错误
        
        Args:
            data: 要验证的数据
            
        Returns:
            Dict[str, str]: 错误字典 {规则索引: 错误消息}
        """
        errors = {}
        
        if not self.rules:
            self.logger.warning(f"验证器 '{self.name}' 没有定义规则")
            return errors
            
        for i, rule in enumerate(self.rules):
            is_valid, message = rule.validate(data)
            if not is_valid:
                rule_key = f"rule_{i}"
                if rule.field:
                    rule_key = rule.field
                errors[rule_key] = message
                
        return errors
    
    def clear_rules(self):
        """清除所有规则"""
        self.rules = []
        
    @staticmethod
    def create_stock_data_validator() -> 'Validator':
        """
        创建股票数据验证器
        
        Returns:
            Validator: 股票数据验证器
        """
        validator = Validator("stock_data")
        
        # 添加通用规则
        validator.add_rule(NotEmptyRule(message="股票数据不能为空"))
        validator.add_rule(DataFrameColumnRule(
            columns=["date", "open", "high", "low", "close", "volume"],
            message="股票数据必须包含标准OHLCV列"
        ))
        
        # 添加日期规则
        validator.add_rule(DateRangeRule(
            field="date",
            min_date="1990-01-01",
            max_date=datetime.datetime.now().strftime('%Y-%m-%d'),
            message="股票日期必须在有效范围内"
        ))
        
        # 添加价格和交易量规则
        validator.add_rule(NumericRangeRule(
            field="open", min_value=0, 
            message="开盘价必须大于等于0"
        ))
        validator.add_rule(NumericRangeRule(
            field="high", min_value=0, 
            message="最高价必须大于等于0"
        ))
        validator.add_rule(NumericRangeRule(
            field="low", min_value=0, 
            message="最低价必须大于等于0"
        ))
        validator.add_rule(NumericRangeRule(
            field="close", min_value=0, 
            message="收盘价必须大于等于0"
        ))
        validator.add_rule(NumericRangeRule(
            field="volume", min_value=0, 
            message="交易量必须大于等于0"
        ))
        
        # 添加自定义规则：检查价格一致性
        def check_price_consistency(df):
            if not isinstance(df, pd.DataFrame):
                return False, "数据必须是DataFrame类型"
                
            # 检查最高价 >= 开盘价, 收盘价, 最低价
            inconsistent = (
                (df["high"] < df["open"]) | 
                (df["high"] < df["close"]) | 
                (df["high"] < df["low"])
            )
            
            if inconsistent.any():
                inconsistent_rows = df[inconsistent].index.tolist()
                return False, f"发现价格不一致：最高价小于其他价格，问题行索引: {inconsistent_rows[:5]}..."
                
            # 检查最低价 <= 开盘价, 收盘价, 最高价
            inconsistent = (
                (df["low"] > df["open"]) | 
                (df["low"] > df["close"]) | 
                (df["low"] > df["high"])
            )
            
            if inconsistent.any():
                inconsistent_rows = df[inconsistent].index.tolist()
                return False, f"发现价格不一致：最低价大于其他价格，问题行索引: {inconsistent_rows[:5]}..."
                
            return True, ""
            
        validator.add_rule(CustomRule(
            validator=check_price_consistency,
            message="股票价格数据不一致"
        ))
        
        return validator
    
    @staticmethod
    def create_financial_data_validator() -> 'Validator':
        """
        创建财务数据验证器
        
        Returns:
            Validator: 财务数据验证器
        """
        validator = Validator("financial_data")
        
        # 添加通用规则
        validator.add_rule(NotEmptyRule(message="财务数据不能为空"))
        validator.add_rule(DataFrameColumnRule(
            columns=["report_date", "code", "total_assets", "total_liability", "net_profit"],
            message="财务数据必须包含基本财务指标列"
        ))
        
        # 添加日期规则
        validator.add_rule(DateRangeRule(
            field="report_date",
            min_date="1990-01-01",
            max_date=datetime.datetime.now().strftime('%Y-%m-%d'),
            message="报告日期必须在有效范围内"
        ))
        
        # 添加财务数据规则
        validator.add_rule(NumericRangeRule(
            field="total_assets", min_value=0, 
            message="总资产必须大于等于0"
        ))
        
        # 添加代码格式验证
        validator.add_rule(PatternRule(
            field="code",
            pattern=r"^\d{6}$",
            message="股票代码必须是6位数字"
        ))
        
        return validator 