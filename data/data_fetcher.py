#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取模块 - 负责从各种数据源获取股票数据
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Lock

# 配置日志
logger = logging.getLogger(__name__)

class DataFetcher:
    """
    数据获取器 - 单例模式实现
    负责从各种数据源获取股票数据
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
    
    def __init__(self):
        """初始化数据获取器"""
        self.stock_names = {}  # 股票代码与名称的映射
        self.industry_stocks = {}  # 行业与股票列表的映射
        self._load_stock_info()
    
    def _load_stock_info(self):
        """加载股票基本信息"""
        logger.info("加载股票基本信息...")
        try:
            # 模拟加载股票信息
            # 实际项目中应从数据源或本地文件加载
            self.stock_names = {
                '000001.SZ': '平安银行',
                '000002.SZ': '万科A',
                '000063.SZ': '中兴通讯',
                '000333.SZ': '美的集团',
                '000651.SZ': '格力电器',
                '000725.SZ': '京东方A',
                '000858.SZ': '五粮液',
                '002304.SZ': '洋河股份',
                '002415.SZ': '海康威视',
                '600000.SH': '浦发银行',
                '600019.SH': '宝钢股份',
                '600028.SH': '中国石化',
                '600036.SH': '招商银行',
                '600276.SH': '恒瑞医药',
                '600309.SH': '万华化学',
                '600519.SH': '贵州茅台',
                '600887.SH': '伊利股份',
                '601318.SH': '中国平安',
                '601857.SH': '中国石油',
                '601988.SH': '中国银行',
                '603288.SH': '海天味业',
                '603501.SH': '韦尔股份',
                '603986.SH': '兆易创新'
            }
            
            # 模拟行业分类
            self.industry_stocks = {
                '银行': ['000001.SZ', '600000.SH', '600036.SH', '601988.SH'],
                '房地产': ['000002.SZ'],
                '科技': ['000063.SZ', '000725.SZ', '002415.SZ', '603501.SH', '603986.SH'],
                '家电': ['000333.SZ', '000651.SZ'],
                '白酒': ['000858.SZ', '002304.SZ', '600519.SH'],
                '医药': ['600276.SH'],
                '化工': ['600309.SH'],
                '保险': ['601318.SH'],
                '能源': ['600028.SH', '601857.SH'],
                '钢铁': ['600019.SH'],
                '食品': ['600887.SH', '603288.SH']
            }
            
            logger.info(f"成功加载 {len(self.stock_names)} 只股票信息")
        except Exception as e:
            logger.error(f"加载股票信息出错: {e}")
    
    def get_stock_list(self, industry=None):
        """
        获取股票列表
        
        参数:
            industry (str, optional): 行业名称，为None则返回所有股票
            
        返回:
            list: 股票代码列表
        """
        if industry is None:
            return list(self.stock_names.keys())
        else:
            return self.industry_stocks.get(industry, [])
    
    def get_stock_name(self, stock_code):
        """
        获取股票名称
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            str: 股票名称
        """
        return self.stock_names.get(stock_code, "未知股票")
    
    def get_k_line_data(self, stock_code, start_date=None, end_date=None):
        """
        获取股票K线数据
        
        参数:
            stock_code (str): 股票代码
            start_date (str, optional): 开始日期，格式为'YYYY-MM-DD'
            end_date (str, optional): 结束日期，格式为'YYYY-MM-DD'
            
        返回:
            pandas.DataFrame: K线数据，包含OHLCV
        """
        logger.info(f"获取股票 {stock_code} 的K线数据...")
        
        try:
            # 使用模拟数据
            days = 120
            
            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_date = datetime.now() - timedelta(days=days)
                
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date = datetime.now()
            
            # 生成日期范围
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # 生成模拟数据
            np.random.seed(int(stock_code[-5:]) % 10000)  # 使用股票代码作为随机种子
            
            # 生成起始价格 - 不同股票不同初始价格
            start_price = 10 + (int(stock_code[-4:]) % 100) * 2
            
            # 生成价格数据
            price_changes = np.random.normal(0, 1, len(date_range)) * 0.02  # 每日变化率
            prices = start_price * np.cumprod(1 + price_changes)
            
            # 生成OHLCV数据
            data = {
                'date': date_range,
                'open': prices * (1 + np.random.normal(0, 0.005, len(date_range))),
                'high': prices * (1 + np.random.normal(0.01, 0.01, len(date_range))),
                'low': prices * (1 - np.random.normal(0.01, 0.01, len(date_range))),
                'close': prices,
                'volume': np.random.normal(1000000, 300000, len(date_range)) * (start_price / 10)
            }
            
            # 确保high >= open, close, low 且 low <= open, close
            for i in range(len(date_range)):
                data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
                data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            logger.info(f"成功获取 {stock_code} 的K线数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 的K线数据出错: {e}")
            return None
    
    def get_stock_price(self, stock_code):
        """
        获取股票最新价格数据
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            pandas.DataFrame: 价格数据
        """
        # 简单地返回最近几天的K线数据
        df = self.get_k_line_data(stock_code, 
                                  start_date=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'),
                                  end_date=datetime.now().strftime('%Y-%m-%d'))
        return df
    
    def get_stock_fundamentals(self, stock_code):
        """
        获取股票基本面数据
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            dict: 基本面数据
        """
        logger.info(f"获取股票 {stock_code} 的基本面数据...")
        
        try:
            # 模拟基本面数据
            np.random.seed(int(stock_code[-4:]) % 1000)
            
            pe = np.random.normal(20, 10)
            pb = np.random.normal(2, 1)
            ps = np.random.normal(3, 1.5)
            dividend_yield = np.random.normal(2, 1) / 100
            roe = np.random.normal(15, 5)
            
            fundamentals = {
                'pe_ratio': abs(pe),
                'pb_ratio': abs(pb),
                'ps_ratio': abs(ps),
                'dividend_yield': abs(dividend_yield),
                'roe': roe,
                'total_assets': np.random.normal(5000, 3000) * 1e8,
                'revenue': np.random.normal(2000, 1000) * 1e8,
                'net_profit': np.random.normal(500, 300) * 1e8,
                'debt_to_assets': np.random.normal(0.4, 0.1),
                'growth_rate': np.random.normal(0.15, 0.1)
            }
            
            logger.info(f"成功获取 {stock_code} 的基本面数据")
            return fundamentals
            
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 的基本面数据出错: {e}")
            return None
    
    def get_index_data(self, index_code):
        """
        获取指数数据
        
        参数:
            index_code (str): 指数代码，如'000001.SH'表示上证指数
            
        返回:
            pandas.DataFrame: 指数数据
        """
        # 类似于股票K线数据的处理方式
        return self.get_k_line_data(index_code)
    
    def get_stock_industry(self, stock_code):
        """
        获取股票所属行业
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            str: 行业名称
        """
        for industry, stocks in self.industry_stocks.items():
            if stock_code in stocks:
                return industry
        return "其他"
