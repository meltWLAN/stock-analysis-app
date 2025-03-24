#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场环境筛选策略模块 - 实现基于市场环境的选股策略
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MarketStrategy:
    """市场环境筛选策略类"""
    
    def __init__(self):
        """初始化市场环境策略"""
        self.logger = logging.getLogger(__name__)
    
    def check_index_uptrend(self, index_df, days=3):
        """
        检查指数连续上涨
        
        Args:
            index_df: 指数数据DataFrame
            days: 连续上涨的天数，默认为3天
            
        Returns:
            bool: 是否满足指数连续上涨条件
        """
        if index_df is None or index_df.empty:
            self.logger.warning("指数数据为空，无法检查指数走势")
            return False
        
        # 确保数据量足够
        if len(index_df) < days:
            self.logger.warning(f"指数数据不足{days}天")
            return False
        
        # 获取最近N天数据
        recent_df = index_df.tail(days).copy()
        
        # 计算每日涨跌幅
        recent_df['pct_change'] = recent_df['close'].pct_change()
        
        # 检查是否连续上涨（第一天的涨跌幅为NaN，从第二天开始检查）
        is_uptrend = (recent_df['pct_change'].iloc[1:] > 0).all()
        
        return is_uptrend
    
    def check_market_sentiment(self, index_df, up_down_df=None, days=5):
        """
        检查市场情绪
        
        Args:
            index_df: 指数数据DataFrame
            up_down_df: 涨跌家数数据DataFrame，可选
            days: 检查的天数范围，默认为5天
            
        Returns:
            bool: 市场情绪是否处于拐点或回暖
        """
        if index_df is None or index_df.empty:
            self.logger.warning("指数数据为空，无法检查市场情绪")
            return False
        
        # 确保数据量足够
        if len(index_df) < days + 5:  # 需要额外的数据来判断趋势
            self.logger.warning(f"指数数据不足{days+5}天")
            return False
        
        # 获取最近数据
        recent_df = index_df.tail(days + 5).copy()
        
        # 计算5日均线
        recent_df['ma5'] = recent_df['close'].rolling(window=5).mean()
        
        # 去除NaN值
        valid_df = recent_df.dropna()
        
        # 判断指数是否站上5日均线
        above_ma5 = valid_df['close'].iloc[-1] > valid_df['ma5'].iloc[-1]
        
        # 判断5日均线是否开始向上
        ma5_turning_up = valid_df['ma5'].diff().iloc[-1] > 0
        
        # 如果提供了涨跌家数数据，判断市场宽度
        market_breadth_good = True
        if up_down_df is not None and not up_down_df.empty:
            recent_up_down = up_down_df.tail(days).copy()
            
            # 根据数据源的不同，字段名可能不同
            if 'up_count' in recent_up_down.columns and 'down_count' in recent_up_down.columns:
                # 计算近期涨跌比
                recent_up_down['up_down_ratio'] = recent_up_down['up_count'] / recent_up_down['down_count']
                # 判断涨跌比是否改善
                market_breadth_good = recent_up_down['up_down_ratio'].iloc[-1] > recent_up_down['up_down_ratio'].iloc[0]
        
        # 综合判断市场情绪：指数站上5日均线且5日均线开始向上或市场宽度改善
        return (above_ma5 and ma5_turning_up) or market_breadth_good
    
    def check_extreme_market(self, index_df, days=5, threshold=-0.05):
        """
        检查是否处于市场极端行情
        
        Args:
            index_df: 指数数据DataFrame
            days: 检查的天数范围，默认为5天
            threshold: 极端下跌阈值，默认为-5%
            
        Returns:
            bool: 是否处于极端行情（返回True表示安全，False表示极端行情）
        """
        if index_df is None or index_df.empty:
            self.logger.warning("指数数据为空，无法检查极端行情")
            return False
        
        # 确保数据量足够
        if len(index_df) < days:
            self.logger.warning(f"指数数据不足{days}天")
            return False
        
        # 获取最近N天数据
        recent_df = index_df.tail(days).copy()
        
        # 计算累计涨跌幅
        cumulative_return = recent_df['close'].iloc[-1] / recent_df['close'].iloc[0] - 1
        
        # 计算连续下跌天数
        recent_df['pct_change'] = recent_df['close'].pct_change()
        consecutive_down_days = (recent_df['pct_change'] < 0).sum()
        
        # 判断是否处于极端行情：累计跌幅超过阈值或连续下跌天数过多
        is_extreme = cumulative_return < threshold or consecutive_down_days >= days - 1
        
        # 返回是否安全（非极端行情）
        return not is_extreme
    
    def check_market_environment(self, index_df, up_down_df=None):
        """
        综合检查市场环境
        
        Args:
            index_df: 指数数据DataFrame
            up_down_df: 涨跌家数数据DataFrame，可选
            
        Returns:
            bool: 市场环境是否适合交易
            float: 市场环境评分（0-100）
        """
        # 检查指数上涨趋势
        index_uptrend = self.check_index_uptrend(index_df)
        
        # 检查市场情绪
        market_sentiment_good = self.check_market_sentiment(index_df, up_down_df)
        
        # 检查是否处于极端行情
        not_extreme = self.check_extreme_market(index_df)
        
        # 计算市场环境评分
        score = 0
        if index_uptrend:
            score += 40
        if market_sentiment_good:
            score += 30
        if not_extreme:
            score += 30
        
        # 判断市场环境是否适合交易：非极端行情且（指数上涨或市场情绪良好）
        is_suitable = not_extreme and (index_uptrend or market_sentiment_good)
        
        return is_suitable, score
    
    def screen(self, stock_list):
        """
        根据市场环境策略筛选股票
        
        Args:
            stock_list: 待筛选的股票列表DataFrame
            
        Returns:
            DataFrame: 筛选后的股票列表
        """
        if stock_list is None or stock_list.empty:
            self.logger.warning("输入的股票列表为空，无法进行筛选")
            return pd.DataFrame()
        
        self.logger.info(f"开始使用市场环境策略筛选，共 {len(stock_list)} 只股票")
        
        # 获取市场环境数据
        index_df = self._get_index_data()
        up_down_df = self._get_up_down_data()
        
        if index_df is None or index_df.empty:
            self.logger.error("无法获取指数数据，市场环境策略筛选失败")
            return pd.DataFrame()
        
        # 检查市场环境
        market_good = self.check_market_environment(index_df, up_down_df)
        if not market_good:
            self.logger.warning("当前市场环境不佳，不推荐进行选股")
            return pd.DataFrame()
        
        # 在市场环境良好的情况下，返回原始股票列表
        self.logger.info(f"市场环境良好，符合选股条件，保留所有 {len(stock_list)} 只股票")
        return stock_list
    
    def _get_index_data(self):
        """获取指数数据（实际实现中应该从数据源获取）"""
        self.logger.warning("使用模拟指数数据，实际使用时请实现真实数据获取")
        
        # 创建模拟数据
        dates = pd.date_range(end=pd.Timestamp.now().date(), periods=30)
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.normal(3500, 100, 30),
            'high': np.random.normal(3550, 100, 30),
            'low': np.random.normal(3450, 100, 30),
            'close': np.random.normal(3500, 100, 30),
            'pct_chg': np.random.normal(0.002, 0.01, 30)  # 涨跌幅
        })
        df.set_index('date', inplace=True)
        
        return df
    
    def _get_up_down_data(self):
        """获取市场涨跌数据（实际实现中应该从数据源获取）"""
        self.logger.warning("使用模拟涨跌数据，实际使用时请实现真实数据获取")
        
        # 创建模拟数据
        dates = pd.date_range(end=pd.Timestamp.now().date(), periods=30)
        df = pd.DataFrame({
            'date': dates,
            'up_count': np.random.randint(1000, 2500, 30),  # 上涨股票数
            'down_count': np.random.randint(1000, 2500, 30),  # 下跌股票数
            'unchanged_count': np.random.randint(0, 300, 30)  # 平盘股票数
        })
        df.set_index('date', inplace=True)
        
        return df