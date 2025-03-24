#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
趋势策略模块 - 基于技术指标的趋势分析策略
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

class TrendStrategy:
    """
    趋势策略 - 使用多种技术指标分析股票趋势
    """
    
    def __init__(self):
        """初始化趋势策略"""
        from data.data_fetcher import DataFetcher
        self.data_fetcher = DataFetcher.get_instance()
    
    def calculate_ma(self, df, windows=[5, 10, 20, 30, 60, 120, 250]):
        """
        计算移动平均线
        
        参数:
            df (pandas.DataFrame): K线数据
            windows (list): 移动平均窗口列表
            
        返回:
            pandas.DataFrame: 包含MA的DataFrame
        """
        result_df = df.copy()
        
        for window in windows:
            result_df[f'ma{window}'] = df['close'].rolling(window=window).mean()
            
        return result_df
    
    def calculate_ema(self, df, windows=[5, 10, 20, 30, 60]):
        """
        计算指数移动平均线
        
        参数:
            df (pandas.DataFrame): K线数据
            windows (list): EMA窗口列表
            
        返回:
            pandas.DataFrame: 包含EMA的DataFrame
        """
        result_df = df.copy()
        
        for window in windows:
            result_df[f'ema{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
        return result_df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        计算MACD指标
        
        参数:
            df (pandas.DataFrame): K线数据
            fast (int): 快线周期
            slow (int): 慢线周期
            signal (int): 信号线周期
            
        返回:
            pandas.DataFrame: 包含MACD的DataFrame
        """
        result_df = df.copy()
        
        # 计算MACD
        result_df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        result_df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        result_df['macd'] = result_df['ema_fast'] - result_df['ema_slow']
        result_df['macd_signal'] = result_df['macd'].ewm(span=signal, adjust=False).mean()
        result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
        
        return result_df
    
    def calculate_rsi(self, df, window=14):
        """
        计算RSI指标
        
        参数:
            df (pandas.DataFrame): K线数据
            window (int): RSI窗口大小
            
        返回:
            pandas.DataFrame: 包含RSI的DataFrame
        """
        result_df = df.copy()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        result_df['rsi'] = 100 - (100 / (1 + rs))
        
        return result_df
    
    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """
        计算布林带
        
        参数:
            df (pandas.DataFrame): K线数据
            window (int): 移动平均窗口
            num_std (float): 标准差倍数
            
        返回:
            pandas.DataFrame: 包含布林带的DataFrame
        """
        result_df = df.copy()
        
        # 计算布林带
        result_df['middle'] = df['close'].rolling(window=window).mean()
        result_df['std'] = df['close'].rolling(window=window).std()
        result_df['upper'] = result_df['middle'] + (result_df['std'] * num_std)
        result_df['lower'] = result_df['middle'] - (result_df['std'] * num_std)
        
        return result_df
    
    def calculate_kdj(self, df, n=9, m1=3, m2=3):
        """
        计算KDJ指标
        
        参数:
            df (pandas.DataFrame): K线数据
            n (int): RSV计算周期
            m1 (int): K值计算周期
            m2 (int): D值计算周期
            
        返回:
            pandas.DataFrame: 包含KDJ的DataFrame
        """
        result_df = df.copy()
        
        # 计算KDJ
        low_list = df['low'].rolling(window=n).min()
        high_list = df['high'].rolling(window=n).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        
        result_df['kdj_k'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
        result_df['kdj_d'] = result_df['kdj_k'].ewm(alpha=1/m2, adjust=False).mean()
        result_df['kdj_j'] = 3 * result_df['kdj_k'] - 2 * result_df['kdj_d']
        
        return result_df
    
    def analyze_stock(self, stock_code):
        """
        分析股票趋势
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            dict: 分析结果
        """
        logger.info(f"使用趋势策略分析股票: {stock_code}")
        
        try:
            # 获取K线数据
            k_line_data = self.data_fetcher.get_k_line_data(stock_code)
            
            if k_line_data is None or k_line_data.empty:
                logger.warning(f"无法获取股票 {stock_code} 的K线数据")
                return {
                    'code': stock_code,
                    'name': self.data_fetcher.get_stock_name(stock_code),
                    'score': 0,
                    'trend': '未知',
                    'signal': '无法分析',
                    'details': {'错误': '无法获取数据'}
                }
            
            # 计算技术指标
            ma_df = self.calculate_ma(k_line_data)
            ema_df = self.calculate_ema(k_line_data)
            macd_df = self.calculate_macd(k_line_data)
            rsi_df = self.calculate_rsi(k_line_data)
            bb_df = self.calculate_bollinger_bands(k_line_data)
            kdj_df = self.calculate_kdj(k_line_data)
            
            # 获取最新数据
            last_data = k_line_data.iloc[-1]
            last_ma = ma_df.iloc[-1]
            last_ema = ema_df.iloc[-1]
            last_macd = macd_df.iloc[-1]
            last_rsi = rsi_df.iloc[-1]
            last_bb = bb_df.iloc[-1]
            last_kdj = kdj_df.iloc[-1]
            
            # 趋势判断逻辑
            trend_signals = []
            
            # 1. 价格相对于MA的位置
            price_above_ma20 = last_data['close'] > last_ma['ma20'] if 'ma20' in last_ma else False
            price_above_ma60 = last_data['close'] > last_ma['ma60'] if 'ma60' in last_ma else False
            
            # 2. MA趋势
            ma20_trend = ma_df['ma20'].diff().iloc[-5:].mean() > 0 if 'ma20' in ma_df.columns else False
            ma60_trend = ma_df['ma60'].diff().iloc[-10:].mean() > 0 if 'ma60' in ma_df.columns else False
            
            # 3. MA系统
            ma_system = (last_ma['ma5'] > last_ma['ma10'] > last_ma['ma20'] > last_ma['ma60']) if all(f'ma{i}' in last_ma for i in [5, 10, 20, 60]) else False
            ma_system_bearish = (last_ma['ma5'] < last_ma['ma10'] < last_ma['ma20'] < last_ma['ma60']) if all(f'ma{i}' in last_ma for i in [5, 10, 20, 60]) else False
            
            # 4. MACD信号
            macd_positive = last_macd['macd'] > 0 if 'macd' in last_macd else False
            macd_above_signal = last_macd['macd'] > last_macd['macd_signal'] if all(i in last_macd for i in ['macd', 'macd_signal']) else False
            macd_hist_positive = last_macd['macd_hist'] > 0 if 'macd_hist' in last_macd else False
            
            # 5. RSI信号
            rsi_bullish = last_rsi['rsi'] > 50 if 'rsi' in last_rsi else False
            rsi_overbought = last_rsi['rsi'] > 70 if 'rsi' in last_rsi else False
            rsi_oversold = last_rsi['rsi'] < 30 if 'rsi' in last_rsi else False
            
            # 6. 布林带位置
            in_upper_bb = last_data['close'] > last_bb['upper'] if 'upper' in last_bb else False
            in_lower_bb = last_data['close'] < last_bb['lower'] if 'lower' in last_bb else False
            
            # 7. KDJ信号
            kdj_golden_cross = (last_kdj['kdj_j'] > last_kdj['kdj_d'] > last_kdj['kdj_k']) if all(f'kdj_{i}' in last_kdj for i in ['j', 'd', 'k']) else False
            kdj_death_cross = (last_kdj['kdj_j'] < last_kdj['kdj_d'] < last_kdj['kdj_k']) if all(f'kdj_{i}' in last_kdj for i in ['j', 'd', 'k']) else False
            
            # 8. 价格突破
            # 计算最近N天的高点和低点
            recent_high = k_line_data['high'].iloc[-20:].max()
            recent_low = k_line_data['low'].iloc[-20:].min()
            
            break_high = last_data['close'] > recent_high * 0.98
            break_low = last_data['close'] < recent_low * 1.02
            
            # 评分计算
            score = 5.0  # 初始评分
            
            # 上涨信号加分
            if price_above_ma20: score += 0.5
            if price_above_ma60: score += 0.5
            if ma20_trend: score += 0.5
            if ma60_trend: score += 0.5
            if ma_system: score += 1.0
            if macd_positive: score += 0.5
            if macd_above_signal: score += 0.5
            if macd_hist_positive: score += 0.5
            if rsi_bullish: score += 0.5
            if kdj_golden_cross: score += 0.5
            if break_high: score += 1.0
            
            # 下跌信号减分
            if ma_system_bearish: score -= 1.0
            if not macd_positive: score -= 0.5
            if not macd_above_signal: score -= 0.5
            if not macd_hist_positive: score -= 0.5
            if not rsi_bullish: score -= 0.5
            if kdj_death_cross: score -= 0.5
            if break_low: score -= 1.0
            
            # 极端情况处理
            if rsi_overbought: 
                score -= 0.5  # 超买可能导致回调
            if rsi_oversold: 
                score += 0.5  # 超卖可能导致反弹
            if in_upper_bb: 
                score -= 0.3  # 接近上轨可能导致回调
            if in_lower_bb: 
                score += 0.3  # 接近下轨可能导致反弹
            
            # 限制评分范围在0-10之间
            score = max(0, min(10, score))
            
            # 判断趋势
            if score >= 7.0:
                trend = "上升"
                signal = "买入"
            elif score >= 5.5:
                trend = "偏强"
                signal = "观望"
            elif score >= 4.5:
                trend = "震荡"
                signal = "观望"
            elif score >= 3.0:
                trend = "偏弱"
                signal = "观望"
            else:
                trend = "下降"
                signal = "卖出"
            
            # 构建分析结果
            analysis_result = {
                'code': stock_code,
                'name': self.data_fetcher.get_stock_name(stock_code),
                'score': score,
                'trend': trend,
                'signal': signal,
                'details': {
                    '价格高于MA20': price_above_ma20,
                    '价格高于MA60': price_above_ma60,
                    'MA20上升趋势': ma20_trend,
                    'MA60上升趋势': ma60_trend,
                    'MA系统多头排列': ma_system,
                    'MA系统空头排列': ma_system_bearish,
                    'MACD为正': macd_positive,
                    'MACD在信号线之上': macd_above_signal,
                    'MACD柱状体为正': macd_hist_positive,
                    'RSI多头': rsi_bullish,
                    'RSI超买': rsi_overbought,
                    'RSI超卖': rsi_oversold,
                    '价格在布林上轨之上': in_upper_bb,
                    '价格在布林下轨之下': in_lower_bb,
                    'KDJ金叉': kdj_golden_cross,
                    'KDJ死叉': kdj_death_cross,
                    '突破近期高点': break_high,
                    '跌破近期低点': break_low,
                    '最新收盘价': last_data['close'],
                    '最新RSI值': last_rsi['rsi'] if 'rsi' in last_rsi else None,
                    '最新MACD值': last_macd['macd'] if 'macd' in last_macd else None
                }
            }
            
            logger.info(f"股票 {stock_code} 分析完成，趋势: {trend}, 评分: {score:.1f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            return {
                'code': stock_code,
                'name': self.data_fetcher.get_stock_name(stock_code),
                'score': 0,
                'trend': '未知',
                'signal': '分析出错',
                'details': {'错误': str(e)}
            }