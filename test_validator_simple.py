#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版数据验证器测试
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import importlib.util

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 直接导入模块文件
def import_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # 直接导入验证器模块
    validator_module = import_file(
        "data_validator", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/data_validator.py")
    )
    FinancialDataValidator = validator_module.FinancialDataValidator
    validator = FinancialDataValidator()
    logger.info("成功导入数据验证器模块")
except Exception as e:
    logger.error(f"导入数据验证器模块失败: {e}")
    sys.exit(1)

def create_test_price_data(with_errors=False):
    """创建测试价格数据"""
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

def test_validator_direct_usage():
    """测试数据验证器的直接使用"""
    logger.info("\n=== 测试数据验证器的直接使用 ===")
    
    # 测试股票代码验证
    codes_to_test = [
        '600000.SH',  # 有效A股
        '000001.SZ',  # 有效A股
        '00700.HK',   # 有效港股
        'AAPL',       # 有效美股
        '600.SH',     # 无效代码
        '12345',      # 无效代码
    ]
    
    logger.info("测试股票代码验证:")
    for code in codes_to_test:
        result = validator.validate_stock_code(code)
        logger.info(f"代码 {code}: {'✓ 有效' if result else '✗ 无效'}")
    
    # 测试价格数据验证
    logger.info("\n测试价格数据验证 - 正常数据:")
    normal_data = create_test_price_data(with_errors=False)
    result = validator.validate_price_data(normal_data)
    logger.info(f"结果: {result}")
    
    logger.info("\n测试价格数据验证 - 异常数据:")
    error_data = create_test_price_data(with_errors=True)
    result = validator.validate_price_data(error_data)
    logger.info(f"结果: {result}")
    
    return True

if __name__ == "__main__":
    logger.info("开始测试数据验证器...")
    
    # 测试数据验证器的直接使用
    test_validator_direct_usage()
    
    logger.info("数据验证器测试完成!") 