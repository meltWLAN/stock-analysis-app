#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据验证器模块
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 直接导入数据验证器模块
try:
    # 直接导入模块，跳过__init__.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data.data_validator import FinancialDataValidator
    
    # 手动导入数据流水线相关功能
    class DataPipeline:
        def __init__(self, name="default"):
            self.name = name
            self.stages = []
            self.validator = FinancialDataValidator()
            
        def add_stage(self, processor, **kwargs):
            self.stages.append({
                'processor': processor,
                'params': kwargs
            })
            return self
            
        def execute(self, data, context=None):
            if context is None:
                context = {}
                
            result = data
            for stage in self.stages:
                processor = stage['processor']
                params = stage['params']
                try:
                    result = processor(result, **params, context=context)
                except Exception as e:
                    logger.error(f"执行流水线错误: {e}")
                    break
            return result
            
        def add_validation_stage(self, validation_type='price'):
            if validation_type == 'price':
                self.add_stage(validate_price_data)
            elif validation_type == 'financial':
                self.add_stage(validate_financial_data)
            return self
            
    # 定义验证函数        
    def validate_price_data(df, **kwargs):
        context = kwargs.get('context', {})
        validator = kwargs.get('validator', FinancialDataValidator())
        
        if not isinstance(df, pd.DataFrame):
            return df
            
        result = validator.validate_price_data(df)
        context['validation_result'] = result
        return df
        
    def validate_financial_data(df, **kwargs):
        context = kwargs.get('context', {})
        validator = kwargs.get('validator', FinancialDataValidator())
        
        if not isinstance(df, pd.DataFrame):
            return df
            
        result = validator.validate_financial_data(df)
        context['validation_result'] = result
        return df
        
    logger.info("成功导入数据验证器模块")
    
except ImportError as e:
    logger.error(f"导入数据验证器模块失败: {e}")
    sys.exit(1)

def create_test_price_data(with_errors=False):
    """
    创建测试价格数据
    
    参数:
        with_errors (bool): 是否包含错误数据
        
    返回:
        pd.DataFrame: 测试数据
    """
    # 创建日期范围
    dates = pd.date_range(start='2020-01-01', end='2020-01-10')
    
    # 创建正常数据
    data = {
        'date': dates,
        'open': np.random.uniform(10, 20, len(dates)),
        'high': np.random.uniform(15, 25, len(dates)),
        'low': np.random.uniform(8, 15, len(dates)),
        'close': np.random.uniform(10, 20, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }
    
    # 确保high > low
    for i in range(len(dates)):
        if data['high'][i] < data['low'][i]:
            data['high'][i], data['low'][i] = data['low'][i], data['high'][i]
        # 确保open和close在high和low之间
        data['open'][i] = data['low'][i] + (data['high'][i] - data['low'][i]) * np.random.random()
        data['close'][i] = data['low'][i] + (data['high'][i] - data['low'][i]) * np.random.random()
    
    df = pd.DataFrame(data)
    
    if with_errors:
        # 添加一些错误数据
        # 1. 负值
        df.loc[2, 'close'] = -5.0
        
        # 2. 高低价逻辑错误
        df.loc[4, 'high'] = 10.0
        df.loc[4, 'low'] = 15.0
        
        # 3. 开收盘价超出高低价范围
        df.loc[6, 'open'] = df.loc[6, 'high'] + 2.0
        
        # 4. 异常价格跳跃
        df.loc[8, 'close'] = df.loc[7, 'close'] * 2.5
    
    df.set_index('date', inplace=True)
    return df

def create_test_financial_data(with_errors=False):
    """
    创建测试财务数据
    
    参数:
        with_errors (bool): 是否包含错误数据
        
    返回:
        pd.DataFrame: 测试数据
    """
    # 创建报告日期
    report_dates = ['2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31']
    
    # 创建正常数据
    data = {
        'report_date': report_dates,
        'revenue': np.random.uniform(1e8, 5e8, len(report_dates)),
        'net_profit': np.random.uniform(1e7, 5e7, len(report_dates)),
        'pe_ratio': np.random.uniform(10, 30, len(report_dates)),
        'pb_ratio': np.random.uniform(1, 5, len(report_dates)),
        'debt_to_assets': np.random.uniform(0.2, 0.6, len(report_dates))
    }
    
    # 确保净利润<营收
    for i in range(len(report_dates)):
        data['net_profit'][i] = min(data['net_profit'][i], data['revenue'][i] * 0.3)
    
    df = pd.DataFrame(data)
    
    if with_errors:
        # 添加一些错误数据
        # 1. 异常PE值
        df.loc[1, 'pe_ratio'] = 1500.0
        
        # 2. 异常PB值
        df.loc[2, 'pb_ratio'] = -0.5
        
        # 3. 异常的利润/营收比率
        df.loc[3, 'net_profit'] = df.loc[3, 'revenue'] * 0.6
        
        # 4. 资产负债率异常
        df.loc[0, 'debt_to_assets'] = 1.2
    
    return df

def test_validator_with_pipeline():
    """测试数据验证器在流水线中的使用"""
    logger.info("=== 测试数据验证器在数据流水线中的使用 ===")
    
    # 创建数据流水线
    pipeline = DataPipeline("test_validation_pipeline")
    
    # 添加验证阶段
    pipeline.add_validation_stage(validation_type='price')
    
    # 创建测试数据 - 有错误版本
    test_data = create_test_price_data(with_errors=True)
    logger.info(f"创建的测试数据维度: {test_data.shape}")
    
    # 执行验证流程
    context = {}
    logger.info("开始执行验证流程...")
    result = pipeline.execute(test_data, context)
    
    # 检查验证结果
    validation_result = context.get('validation_result', {})
    logger.info(f"验证结果: {validation_result}")
    
    # 测试财务数据验证
    logger.info("\n=== 测试财务数据验证 ===")
    pipeline = DataPipeline("test_financial_validation")
    pipeline.add_validation_stage(validation_type='financial')
    
    test_financial_data = create_test_financial_data(with_errors=True)
    logger.info(f"创建的财务测试数据维度: {test_financial_data.shape}")
    
    context = {}
    result = pipeline.execute(test_financial_data, context)
    
    validation_result = context.get('validation_result', {})
    logger.info(f"财务数据验证结果: {validation_result}")
    
    return True

def test_validator_direct_usage():
    """测试数据验证器的直接使用"""
    logger.info("\n=== 测试数据验证器的直接使用 ===")
    
    # 创建验证器实例
    validator = FinancialDataValidator()
    
    # 测试股票代码验证
    codes_to_test = [
        '600000.SH',  # 有效A股
        '000001.SZ',  # 有效A股
        '00700.HK',   # 有效港股
        'AAPL',       # 有效美股
        '600.SH',     # 无效代码
        '12345',      # 无效代码
        None          # 无效输入
    ]
    
    logger.info("测试股票代码验证:")
    for code in codes_to_test:
        result = validator.validate_stock_code(code)
        logger.info(f"代码 {code}: {'✓ 有效' if result else '✗ 无效'}")
    
    # 测试日期范围验证
    date_ranges = [
        ('2020-01-01', '2020-12-31'),    # 有效范围
        ('2020-12-31', '2020-01-01'),    # 无效范围（开始>结束）
        ('1980-01-01', '2020-01-01'),    # 无效范围（开始太早）
        ('2020-01-01', '2030-01-01'),    # 无效范围（结束太晚）
    ]
    
    logger.info("\n测试日期范围验证:")
    for start, end in date_ranges:
        result = validator.validate_date_range(start, end)
        logger.info(f"日期范围 {start} 到 {end}: {'✓ 有效' if result else '✗ 无效'}")
    
    # 测试价格数据验证
    logger.info("\n测试价格数据验证 - 正常数据:")
    normal_data = create_test_price_data(with_errors=False)
    result = validator.validate_price_data(normal_data)
    logger.info(f"结果: {result}")
    
    logger.info("\n测试价格数据验证 - 异常数据:")
    error_data = create_test_price_data(with_errors=True)
    result = validator.validate_price_data(error_data)
    logger.info(f"结果: {result}")
    
    # 测试财务数据验证
    logger.info("\n测试财务数据验证 - 正常数据:")
    normal_financial = create_test_financial_data(with_errors=False)
    result = validator.validate_financial_data(normal_financial)
    logger.info(f"结果: {result}")
    
    logger.info("\n测试财务数据验证 - 异常数据:")
    error_financial = create_test_financial_data(with_errors=True)
    result = validator.validate_financial_data(error_financial)
    logger.info(f"结果: {result}")
    
    return True

if __name__ == "__main__":
    logger.info("开始测试数据验证器...")
    
    # 测试数据验证器在流水线中的使用
    test_validator_with_pipeline()
    
    # 测试数据验证器的直接使用
    test_validator_direct_usage()
    
    logger.info("数据验证器测试完成!") 