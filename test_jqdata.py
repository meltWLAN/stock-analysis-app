#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试JQData连接和数据获取
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量获取JoinQuant凭据
jq_username = os.environ.get('JOINQUANT_USERNAME')
jq_password = os.environ.get('JOINQUANT_PASSWORD')

if not jq_username or not jq_password:
    logger.error("未设置JoinQuant账号密码环境变量 JOINQUANT_USERNAME 和 JOINQUANT_PASSWORD")
    exit(1)

logger.info(f"使用JoinQuant账号: {jq_username}")

try:
    import jqdatasdk as jq
    logger.info(f"jqdatasdk版本: {jq.__version__}")
except ImportError:
    logger.error("未安装jqdatasdk，请先安装: pip install jqdatasdk")
    exit(1)

def test_jqdata_connection():
    """测试JQData连接"""
    try:
        logger.info("尝试连接JQData...")
        jq.auth(jq_username, jq_password)
        logger.info("JQData连接成功!")
        return True
    except Exception as e:
        logger.error(f"JQData连接失败: {e}")
        return False

def test_get_stock_list():
    """测试获取股票列表"""
    try:
        logger.info("获取股票列表...")
        stocks = jq.get_all_securities(['stock'])
        logger.info(f"成功获取股票数量: {len(stocks)}")
        logger.info(f"样例股票数据:\n{stocks.head()}")
        return stocks
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return pd.DataFrame()

def test_jqdata_quota():
    """测试JQData配额信息"""
    try:
        logger.info("获取JQData配额信息...")
        quota_info = jq.get_query_count()
        logger.info(f"配额信息: {quota_info}")
        return quota_info
    except Exception as e:
        logger.error(f"获取配额信息失败: {e}")
        return None

def test_get_price_data():
    """测试获取价格数据"""
    try:
        # 获取上证50成分股
        logger.info("获取上证50成分股...")
        stocks = jq.get_index_stocks('000016.XSHG')
        logger.info(f"上证50成分股数量: {len(stocks)}")
        
        # 获取第一个股票的价格数据
        stock_code = stocks[0]
        # 使用非常短的日期范围
        end_date = '2024-03-01'
        start_date = '2024-02-01'
        
        logger.info(f"获取股票 {stock_code} 在 {start_date} 至 {end_date} 的价格数据...")
        price_data = jq.get_price(
            stock_code, 
            start_date=start_date, 
            end_date=end_date,
            frequency='daily',
            fields=['open', 'close', 'high', 'low', 'volume']
        )
        
        logger.info(f"成功获取价格数据条数: {len(price_data)}")
        logger.info(f"样例价格数据:\n{price_data.head()}")
        return price_data
    except Exception as e:
        logger.error(f"获取价格数据失败: {e}")
        return pd.DataFrame()

def test_get_fundamentals():
    """测试获取基本面数据"""
    try:
        logger.info("获取基本面数据...")
        q = jq.query(jq.valuation.code, jq.valuation.market_cap, jq.valuation.pe_ratio)
        # 使用账户允许的日期
        date = '2024-05-01'
        logger.info(f"获取 {date} 的基本面数据...")
        df = jq.get_fundamentals(q, date=date)
        logger.info(f"成功获取基本面数据条数: {len(df)}")
        logger.info(f"样例基本面数据:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"获取基本面数据失败: {e}")
        return pd.DataFrame()

def main():
    """主函数"""
    logger.info("开始测试JQData...")
    
    # 测试连接
    if not test_jqdata_connection():
        logger.error("JQData连接测试失败，退出测试")
        return
    
    # 测试配额信息
    test_jqdata_quota()
    
    # 测试获取股票列表
    test_get_stock_list()
    
    # 测试获取价格数据
    test_get_price_data()
    
    # 测试获取基本面数据
    test_get_fundamentals()
    
    logger.info("JQData测试完成")

if __name__ == "__main__":
    main() 