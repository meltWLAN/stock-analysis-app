#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据流水线模块 - 负责组织和执行数据处理流程
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from .data_validator import FinancialDataValidator

# 配置日志
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    数据流水线 - 组织和执行数据处理流程
    """
    
    def __init__(self, name="default"):
        """
        初始化数据流水线
        
        参数:
            name (str): 流水线名称
        """
        self.name = name
        self.stages = []  # 流水线阶段列表
        self.validator = FinancialDataValidator()  # 初始化数据验证器
        logger.info(f"初始化数据流水线: {name}")
    
    def add_stage(self, processor, **kwargs):
        """
        添加处理阶段
        
        参数:
            processor (callable): 处理函数或对象
            **kwargs: 传递给处理器的参数
            
        返回:
            DataPipeline: 流水线对象自身，便于链式调用
        """
        self.stages.append({
            'processor': processor,
            'params': kwargs
        })
        logger.debug(f"流水线 {self.name} 添加处理阶段: {processor.__name__ if hasattr(processor, '__name__') else 'unnamed'}")
        return self
    
    def execute(self, data, context=None):
        """
        执行流水线处理
        
        参数:
            data: 初始数据
            context (dict, optional): 执行上下文
            
        返回:
            处理后的数据
        """
        if context is None:
            context = {}
        
        result = data
        stage_count = len(self.stages)
        
        logger.info(f"开始执行流水线 {self.name}，共 {stage_count} 个阶段")
        
        for i, stage in enumerate(self.stages):
            processor = stage['processor']
            params = stage['params']
            
            try:
                # 记录执行信息
                stage_name = processor.__name__ if hasattr(processor, '__name__') else 'unnamed_stage'
                logger.debug(f"执行阶段 {i+1}/{stage_count}: {stage_name}")
                
                # 执行处理
                start_time = datetime.now()
                result = processor(result, **params, context=context)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # 记录执行结果
                logger.debug(f"阶段 {i+1}/{stage_count} {stage_name} 完成，耗时 {elapsed:.3f}s")
                
            except Exception as e:
                logger.error(f"流水线 {self.name} 在阶段 {i+1}/{stage_count} 执行出错: {e}")
                # 记录执行上下文
                logger.debug(f"错误上下文: data={type(data)}, context={context}")
                # 如果是DataFrame，记录部分数据
                if isinstance(result, pd.DataFrame):
                    logger.debug(f"错误时的数据样例: shape={result.shape}, columns={result.columns}")
                # 出错时保留现有结果而不中断流水线
                break
        
        logger.info(f"流水线 {self.name} 执行完成")
        return result
    
    def copy(self):
        """
        创建流水线的副本
        
        返回:
            DataPipeline: 流水线的副本
        """
        new_pipeline = DataPipeline(f"{self.name}_copy")
        new_pipeline.stages = self.stages.copy()
        return new_pipeline
    
    def get_summary(self):
        """
        获取流水线摘要
        
        返回:
            dict: 流水线结构摘要
        """
        return {
            'name': self.name,
            'stages': [
                {
                    'name': s['processor'].__name__ if hasattr(s['processor'], '__name__') else 'unnamed_stage',
                    'params': s['params']
                } for s in self.stages
            ],
            'total_stages': len(self.stages)
        }

    def add_validation_stage(self, validation_type='price'):
        """
        添加数据验证阶段
        
        参数:
            validation_type (str): 验证类型，可选 'price' 或 'financial'
            
        返回:
            DataPipeline: 流水线对象自身，便于链式调用
        """
        if validation_type == 'price':
            self.add_stage(validate_price_data)
        elif validation_type == 'financial':
            self.add_stage(validate_financial_data)
        else:
            logger.warning(f"未知的验证类型: {validation_type}")
        
        return self

# 常用数据预处理函数

def remove_duplicates(df, **kwargs):
    """
    移除数据中的重复行
    
    参数:
        df (pandas.DataFrame): 输入数据
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("remove_duplicates: 输入不是DataFrame，跳过处理")
        return df
    
    original_len = len(df)
    if original_len == 0:
        return df
    
    df = df.drop_duplicates()
    logger.debug(f"移除了 {original_len - len(df)} 条重复记录")
    return df

def handle_missing_values(df, method='ffill', **kwargs):
    """
    处理缺失值
    
    参数:
        df (pandas.DataFrame): 输入数据
        method (str): 处理方法，可选 'ffill', 'bfill', 'mean', 'median', 'drop'
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("handle_missing_values: 输入不是DataFrame，跳过处理")
        return df
    
    original_len = len(df)
    if original_len == 0:
        return df
    
    # 统计缺失值
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        return df
    
    logger.debug(f"处理前缺失值总数: {missing_count}")
    
    # 根据方法处理缺失值
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'mean':
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
    elif method == 'median':
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
    elif method == 'drop':
        df = df.dropna()
        logger.debug(f"删除含有缺失值的行，剩余记录数: {len(df)}")
    else:
        logger.warning(f"未知的缺失值处理方法: {method}")
    
    # 统计处理后的缺失值
    remaining_missing = df.isnull().sum().sum()
    logger.debug(f"处理后缺失值总数: {remaining_missing}")
    
    return df

def normalize_columns(df, columns=None, method='zscore', **kwargs):
    """
    对数值列进行标准化
    
    参数:
        df (pandas.DataFrame): 输入数据
        columns (list): 要处理的列，为None则处理所有数值列
        method (str): 标准化方法，可选 'zscore', 'minmax', 'robust'
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("normalize_columns: 输入不是DataFrame，跳过处理")
        return df
    
    if len(df) == 0:
        return df
    
    # 确定要处理的列
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if not columns:
        logger.debug("没有需要标准化的列")
        return df
    
    # 创建结果DataFrame的副本
    result = df.copy()
    
    # 根据方法进行标准化
    for col in columns:
        if col not in df.columns:
            logger.warning(f"列 '{col}' 不存在，跳过处理")
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"列 '{col}' 不是数值类型，跳过处理")
            continue
        
        if method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                result[col] = 0
            else:
                result[col] = (df[col] - mean) / std
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val == min_val:
                result[col] = 0
            else:
                result[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                result[col] = 0
            else:
                result[col] = (df[col] - median) / iqr
        else:
            logger.warning(f"未知的标准化方法: {method}")
    
    logger.debug(f"已完成 {len(columns)} 列的标准化，方法: {method}")
    return result

def add_date_features(df, date_column='date', **kwargs):
    """
    添加日期特征
    
    参数:
        df (pandas.DataFrame): 输入数据
        date_column (str): 日期列名
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("add_date_features: 输入不是DataFrame，跳过处理")
        return df
    
    if len(df) == 0:
        return df
    
    # 检查日期列是否存在
    if date_column not in df.columns:
        logger.warning(f"日期列 '{date_column}' 不存在，尝试查找其他日期列")
        # 尝试查找其他可能的日期列
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_column = col
                logger.debug(f"找到日期列: {date_column}")
                break
        else:
            logger.warning("未找到有效的日期列，跳过处理")
            return df
    
    # 确保日期列为datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            logger.warning(f"无法将列 '{date_column}' 转换为日期类型，跳过处理")
            return df
    
    # 创建结果DataFrame的副本
    result = df.copy()
    
    # 添加日期特征
    result['year'] = df[date_column].dt.year
    result['month'] = df[date_column].dt.month
    result['day'] = df[date_column].dt.day
    result['dayofweek'] = df[date_column].dt.dayofweek
    result['quarter'] = df[date_column].dt.quarter
    result['is_month_end'] = df[date_column].dt.is_month_end
    result['is_month_start'] = df[date_column].dt.is_month_start
    result['is_quarter_end'] = df[date_column].dt.is_quarter_end
    result['is_quarter_start'] = df[date_column].dt.is_quarter_start
    result['is_year_end'] = df[date_column].dt.is_year_end
    result['is_year_start'] = df[date_column].dt.is_year_start
    
    logger.debug(f"已添加日期特征，共 {len(result.columns) - len(df.columns)} 个新列")
    return result

def calculate_returns(df, price_column='close', date_column='date', periods=[1, 5, 20], **kwargs):
    """
    计算收益率
    
    参数:
        df (pandas.DataFrame): 输入数据
        price_column (str): 价格列名
        date_column (str): 日期列名
        periods (list): 计算周期列表
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 处理后的数据
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("calculate_returns: 输入不是DataFrame，跳过处理")
        return df
    
    if len(df) == 0:
        return df
    
    # 检查价格列是否存在
    if price_column not in df.columns:
        logger.warning(f"价格列 '{price_column}' 不存在，跳过处理")
        return df
    
    # 创建结果DataFrame的副本
    result = df.copy()
    
    # 确保按日期排序
    if date_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                result[date_column] = pd.to_datetime(df[date_column])
            except:
                logger.warning(f"无法将列 '{date_column}' 转换为日期类型")
        result = result.sort_values(date_column)
    
    # 计算各周期收益率
    for period in periods:
        result[f'return_{period}d'] = result[price_column].pct_change(periods=period)
    
    logger.debug(f"已计算 {len(periods)} 个周期的收益率")
    return result

def create_default_pipeline():
    """
    创建默认的数据处理流水线
    
    返回:
        DataPipeline: 默认流水线
    """
    pipeline = DataPipeline("default_pipeline")
    
    # 添加处理阶段
    pipeline.add_stage(remove_duplicates)
    pipeline.add_stage(handle_missing_values, method='ffill')
    pipeline.add_stage(calculate_returns, periods=[1, 5, 10, 20, 60])
    
    logger.info("创建默认数据处理流水线完成")
    return pipeline 

def validate_price_data(df, **kwargs):
    """
    验证价格数据
    
    参数:
        df (pandas.DataFrame): 输入数据
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 验证后的数据
    """
    context = kwargs.get('context', {})
    validator = kwargs.get('validator', FinancialDataValidator())
    
    if not isinstance(df, pd.DataFrame):
        logger.warning("validate_price_data: 输入不是DataFrame，跳过验证")
        return df
    
    # 执行验证
    result = validator.validate_price_data(df)
    
    # 将验证结果添加到上下文
    context['validation_result'] = result
    
    # 记录验证信息
    if not result['valid']:
        logger.warning(f"价格数据验证失败: {', '.join(result['errors'])}")
    else:
        logger.info(f"价格数据验证通过，质量评分: {result['data_quality_score']:.1f}/100")
    
    if result.get('warnings', []):
        for warning in result['warnings']:
            logger.warning(f"数据警告: {warning}")
    
    # 返回原始数据，以便继续处理
    return df

def validate_financial_data(df, **kwargs):
    """
    验证财务数据
    
    参数:
        df (pandas.DataFrame): 输入数据
        **kwargs: 额外参数
        
    返回:
        pandas.DataFrame: 验证后的数据
    """
    context = kwargs.get('context', {})
    validator = kwargs.get('validator', FinancialDataValidator())
    
    if not isinstance(df, pd.DataFrame):
        logger.warning("validate_financial_data: 输入不是DataFrame，跳过验证")
        return df
    
    # 执行验证
    result = validator.validate_financial_data(df)
    
    # 将验证结果添加到上下文
    context['validation_result'] = result
    
    # 记录验证信息
    if not result['valid']:
        logger.warning(f"财务数据验证失败: {', '.join(result['errors'])}")
    else:
        logger.info(f"财务数据验证通过，质量评分: {result['data_quality_score']:.1f}/100")
    
    if result.get('warnings', []):
        for warning in result['warnings']:
            logger.warning(f"数据警告: {warning}")
    
    # 返回原始数据，以便继续处理
    return df 