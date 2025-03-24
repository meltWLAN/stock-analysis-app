#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试改进后的JoinQuantDataSource
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import sys

# 将当前目录添加到Python路径
sys.path.append('.')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入自定义模块
try:
    from data.data_source_manager import JoinQuantDataSource, DataValidator
except ImportError as e:
    logger.error(f"导入自定义模块失败: {e}")
    sys.exit(1)

def test_basic_functions(jq_source):
    """测试基本功能"""
    logger.info("=== 测试基本功能 ===")
    
    # 测试股票列表获取
    logger.info("获取股票列表...")
    stocks_df = jq_source.get_data("get_all_securities", types=['stock'])
    logger.info(f"获取股票数量: {len(stocks_df)}")
    logger.info(f"样例股票数据:\n{stocks_df.head(3)}")
    
    # 测试指数成分股获取 - 使用有效日期
    logger.info("获取上证50成分股...")
    date_str = '2024-03-01'  # 确保在允许范围内的日期
    index_stocks = jq_source.get_data("get_index_stocks", index_symbol='000016.XSHG', date=date_str)
    
    # 检查结果类型并正确处理
    if isinstance(index_stocks, pd.DataFrame):
        if index_stocks.empty:
            logger.warning("获取的成分股DataFrame为空")
            stock_count = 0
        else:
            stock_count = len(index_stocks)
    elif isinstance(index_stocks, (list, tuple)):
        stock_count = len(index_stocks)
        if stock_count > 0:
            logger.info(f"样例成分股: {index_stocks[:3]}")
    else:
        logger.warning(f"意外的返回类型: {type(index_stocks)}")
        stock_count = 0
    
    logger.info(f"上证50成分股数量: {stock_count}")
    
    # 测试基本面数据
    logger.info("获取基本面数据...")
    date_str = '2024-03-01'  # 使用确定在范围内的日期
    logger.info(f"使用基准日期: {date_str}")
    # 注意：这里需要特殊处理，因为query对象无法直接传递
    # 在实际应用中，需要在源代码中构建query对象
    # 这里我们进行简化测试
    try:
        import jqdatasdk as jq
        q = jq.query(jq.valuation.code, jq.valuation.market_cap, jq.valuation.pe_ratio)
        fundamentals = jq_source.get_data("get_fundamentals", query_object=q, date=date_str)
        logger.info(f"获取基本面数据条数: {len(fundamentals)}")
        logger.info(f"样例基本面数据:\n{fundamentals.head(3)}")
    except Exception as e:
        logger.error(f"获取基本面数据失败: {e}")

def test_date_range_adjustment(jq_source):
    """测试日期范围自动调整"""
    logger.info("=== 测试日期范围自动调整 ===")
    
    # 获取一个用于测试的股票代码
    # 为了避免依赖get_index_stocks，这里使用一个硬编码的股票代码
    stock_code = '600000.XSHG'  # 浦发银行
    logger.info(f"使用股票: {stock_code}")
    
    # 测试超出范围的日期
    logger.info("测试超出范围的日期...")
    start_date = '2020-01-01'  # 远早于允许的日期
    end_date = '2025-01-01'    # 远晚于允许的日期
    
    price_data = jq_source.get_data(
        "get_price", 
        security=stock_code,
        start_date=start_date,
        end_date=end_date,
        frequency='daily',
        fields=['open', 'close', 'high', 'low', 'volume']
    )
    
    if isinstance(price_data, pd.DataFrame) and not price_data.empty:
        logger.info(f"成功获取价格数据，行数: {len(price_data)}")
        logger.info(f"实际日期范围: {price_data.index.min()} 至 {price_data.index.max()}")
        logger.info(f"样例价格数据:\n{price_data.head(3)}")
    else:
        logger.warning("价格数据为空或非DataFrame类型")
    
    # 测试正确范围内的日期
    logger.info("测试正确范围内的日期...")
    start_date = '2024-01-15'  # 应该在允许范围内
    end_date = '2024-03-15'    # 应该在允许范围内
    
    price_data = jq_source.get_data(
        "get_price", 
        security=stock_code,
        start_date=start_date,
        end_date=end_date,
        frequency='daily',
        fields=['open', 'close', 'high', 'low', 'volume']
    )
    
    if isinstance(price_data, pd.DataFrame) and not price_data.empty:
        logger.info(f"成功获取价格数据，行数: {len(price_data)}")
        logger.info(f"实际日期范围: {price_data.index.min()} 至 {price_data.index.max()}")
        logger.info(f"样例价格数据:\n{price_data.head(3)}")
    else:
        logger.warning("价格数据为空或非DataFrame类型")

def main():
    """主函数"""
    logger.info("开始测试改进后的JoinQuantDataSource...")
    
    # 从环境变量获取JoinQuant凭据
    username = os.environ.get('JOINQUANT_USERNAME')
    password = os.environ.get('JOINQUANT_PASSWORD')
    
    if not username or not password:
        logger.error("未设置JoinQuant账号密码环境变量")
        return
    
    # 创建JoinQuantDataSource实例
    jq_source = JoinQuantDataSource(username=username, password=password)
    
    # 检查数据源状态
    logger.info(f"数据源状态: {jq_source.status}")
    if jq_source.status != "up":
        logger.error("JoinQuant数据源初始化失败")
        return
    
    # 测试基本功能
    test_basic_functions(jq_source)
    
    # 测试日期范围自动调整
    test_date_range_adjustment(jq_source)
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 