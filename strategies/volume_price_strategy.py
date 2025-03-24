#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量价策略模块 - 基于成交量与价格关系的分析策略
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

class VolumePriceStrategy:
    """
    量价策略 - 分析成交量与价格的关系来预测趋势
    """
    
    def __init__(self):
        """初始化量价策略"""
        from data.data_fetcher import DataFetcher
        self.data_fetcher = DataFetcher.get_instance()
    
    def calculate_volume_indicators(self, df, windows=[5, 10, 20, 30]):
        """
        计算成交量相关指标
        
        参数:
            df (pandas.DataFrame): K线数据
            windows (list): 移动窗口列表
            
        返回:
            pandas.DataFrame: 包含成交量指标的DataFrame
        """
        result_df = df.copy()
        
        # 计算成交量移动平均
        for window in windows:
            result_df[f'volume_ma{window}'] = df['volume'].rolling(window=window).mean()
        
        # 计算成交量变化率
        result_df['volume_change'] = df['volume'].pct_change()
        
        # 计算相对成交量（与均值的比例）
        result_df['relative_volume'] = df['volume'] / result_df['volume_ma20']
        
        # 计算价量相关性
        result_df['price_change'] = df['close'].pct_change()
        
        # 计算OBV (On-Balance Volume)
        result_df['obv'] = 0
        for i in range(1, len(result_df)):
            if result_df['close'].iloc[i] > result_df['close'].iloc[i-1]:
                result_df['obv'].iloc[i] = result_df['obv'].iloc[i-1] + result_df['volume'].iloc[i]
            elif result_df['close'].iloc[i] < result_df['close'].iloc[i-1]:
                result_df['obv'].iloc[i] = result_df['obv'].iloc[i-1] - result_df['volume'].iloc[i]
            else:
                result_df['obv'].iloc[i] = result_df['obv'].iloc[i-1]
        
        # 计算成交量震荡指标
        result_df['volume_oscillator'] = (
            result_df['volume_ma5'] - result_df['volume_ma20']
        ) / result_df['volume_ma20'] * 100
        
        return result_df
    
    def calculate_price_volume_correlation(self, df, window=20):
        """
        计算价格与成交量的相关性
        
        参数:
            df (pandas.DataFrame): K线数据
            window (int): 窗口大小
            
        返回:
            pandas.DataFrame: 包含相关性的DataFrame
        """
        result_df = df.copy()
        
        # 计算价格变化
        result_df['price_change'] = df['close'].pct_change()
        
        # 计算成交量变化
        result_df['volume_change'] = df['volume'].pct_change()
        
        # 计算滚动相关性
        result_df['price_volume_corr'] = (
            result_df['price_change'].rolling(window=window)
            .corr(result_df['volume_change'])
        )
        
        return result_df
    
    def calculate_volume_price_ratio(self, df, windows=[5, 10, 20]):
        """
        计算成交量/价格比率指标
        
        参数:
            df (pandas.DataFrame): K线数据
            windows (list): 移动窗口列表
            
        返回:
            pandas.DataFrame: 包含成交量价格比的DataFrame
        """
        result_df = df.copy()
        
        # 计算成交量/价格比
        result_df['vpr'] = df['volume'] / df['close']
        
        # 计算VPR的移动平均
        for window in windows:
            result_df[f'vpr_ma{window}'] = result_df['vpr'].rolling(window=window).mean()
        
        # 计算VPR变化率
        result_df['vpr_change'] = result_df['vpr'].pct_change()
        
        return result_df
    
    def detect_volume_breakout(self, df, threshold=2.0, window=20):
        """
        检测成交量突破
        
        参数:
            df (pandas.DataFrame): K线数据
            threshold (float): 突破阈值
            window (int): 参考窗口
            
        返回:
            pandas.DataFrame: 带有突破标记的DataFrame
        """
        result_df = df.copy()
        
        # 计算成交量均值和标准差
        volume_mean = result_df['volume'].rolling(window=window).mean()
        volume_std = result_df['volume'].rolling(window=window).std()
        
        # 检测成交量突破
        result_df['volume_breakout'] = (
            result_df['volume'] > volume_mean + threshold * volume_std
        )
        
        # 检测放量上涨和放量下跌
        result_df['volume_price_up'] = (
            (result_df['close'] > result_df['open']) & 
            (result_df['volume'] > volume_mean * 1.5)
        )
        
        result_df['volume_price_down'] = (
            (result_df['close'] < result_df['open']) & 
            (result_df['volume'] > volume_mean * 1.5)
        )
        
        return result_df
    
    def detect_divergence(self, df, price_window=14, volume_window=14):
        """
        检测价量背离
        
        参数:
            df (pandas.DataFrame): K线数据
            price_window (int): 价格趋势窗口
            volume_window (int): 成交量趋势窗口
            
        返回:
            pandas.DataFrame: 带有背离标记的DataFrame
        """
        result_df = df.copy()
        
        # 计算价格趋势
        price_trend = result_df['close'].rolling(window=price_window).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
        
        # 计算成交量趋势
        volume_trend = result_df['volume'].rolling(window=volume_window).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
        
        # 检测背离
        result_df['bullish_divergence'] = (
            (price_trend == -1) & (volume_trend == 1)
        )
        
        result_df['bearish_divergence'] = (
            (price_trend == 1) & (volume_trend == -1)
        )
        
        return result_df
    
    def analyze_stock(self, stock_code):
        """
        分析股票量价关系
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            dict: 分析结果
        """
        logger.info(f"使用量价策略分析股票: {stock_code}")
        
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
            
            # 计算各种量价指标
            volume_df = self.calculate_volume_indicators(k_line_data)
            corr_df = self.calculate_price_volume_correlation(k_line_data)
            vpr_df = self.calculate_volume_price_ratio(k_line_data)
            breakout_df = self.detect_volume_breakout(k_line_data)
            divergence_df = self.detect_divergence(k_line_data)
            
            # 获取最新数据
            last_volume = volume_df.iloc[-1]
            last_corr = corr_df.iloc[-1]
            last_vpr = vpr_df.iloc[-1]
            last_breakout = breakout_df.iloc[-1]
            last_divergence = divergence_df.iloc[-1]
            
            # 计算最近几个交易日的各项指标
            recent_days = 10
            recent_data = k_line_data.iloc[-recent_days:]
            recent_volume = volume_df.iloc[-recent_days:]
            
            # 量价信号分析
            # 1. 成交量趋势
            volume_increasing = recent_volume['volume'].iloc[-5:].mean() > recent_volume['volume'].iloc[:5].mean()
            
            # 2. 成交量与均线关系
            volume_above_ma20 = last_volume['volume'] > last_volume['volume_ma20'] if 'volume_ma20' in last_volume else False
            
            # 3. 相对成交量
            high_relative_volume = last_volume['relative_volume'] > 1.5 if 'relative_volume' in last_volume else False
            low_relative_volume = last_volume['relative_volume'] < 0.7 if 'relative_volume' in last_volume else False
            
            # 4. OBV趋势
            obv_increasing = volume_df['obv'].diff().iloc[-5:].mean() > 0 if 'obv' in volume_df.columns else False
            
            # 5. 成交量震荡指标
            vo_positive = last_volume['volume_oscillator'] > 0 if 'volume_oscillator' in last_volume else False
            
            # 6. 价量相关性
            price_volume_positive_corr = last_corr['price_volume_corr'] > 0.5 if 'price_volume_corr' in last_corr else False
            price_volume_negative_corr = last_corr['price_volume_corr'] < -0.5 if 'price_volume_corr' in last_corr else False
            
            # 7. 成交量价格比
            vpr_decreasing = vpr_df['vpr'].diff().iloc[-5:].mean() < 0 if 'vpr' in vpr_df.columns else False
            
            # 8. 成交量突破
            volume_breakout = last_breakout['volume_breakout'] if 'volume_breakout' in last_breakout else False
            
            # 9. 放量上涨和放量下跌
            volume_price_up = any(breakout_df['volume_price_up'].iloc[-3:]) if 'volume_price_up' in breakout_df.columns else False
            volume_price_down = any(breakout_df['volume_price_down'].iloc[-3:]) if 'volume_price_down' in breakout_df.columns else False
            
            # 10. 价量背离
            bullish_divergence = any(divergence_df['bullish_divergence'].iloc[-5:]) if 'bullish_divergence' in divergence_df.columns else False
            bearish_divergence = any(divergence_df['bearish_divergence'].iloc[-5:]) if 'bearish_divergence' in divergence_df.columns else False
            
            # 11. 近期价格走势
            price_trend = recent_data['close'].iloc[-1] > recent_data['close'].iloc[0]
            
            # 12. 量价配合度
            volume_price_alignment = (
                (price_trend and volume_increasing) or 
                (not price_trend and not volume_increasing)
            )
            
            # 评分计算
            score = 5.0  # 初始评分
            
            # 积极信号加分
            if volume_increasing and price_trend: score += 0.8
            if volume_above_ma20 and price_trend: score += 0.5
            if high_relative_volume and price_trend: score += 0.6
            if obv_increasing: score += 0.7
            if vo_positive: score += 0.4
            if price_volume_positive_corr and price_trend: score += 0.5
            if vpr_decreasing and price_trend: score += 0.4
            if volume_breakout and price_trend: score += 0.7
            if volume_price_up: score += 0.8
            if bullish_divergence: score += 1.0
            if volume_price_alignment: score += 0.5
            
            # 消极信号减分
            if volume_increasing and not price_trend: score -= 0.5
            if volume_above_ma20 and not price_trend: score -= 0.6
            if high_relative_volume and not price_trend: score -= 0.7
            if not obv_increasing: score -= 0.5
            if not vo_positive: score -= 0.3
            if price_volume_negative_corr and price_trend: score -= 0.6
            if not vpr_decreasing and price_trend: score -= 0.3
            if volume_breakout and not price_trend: score -= 0.6
            if volume_price_down: score -= 0.8
            if bearish_divergence: score -= 1.0
            if not volume_price_alignment: score -= 0.5
            
            # 极端情况调整
            if low_relative_volume: score -= 0.3  # 交投不活跃
            
            # 限制评分范围
            score = max(0, min(10, score))
            
            # 判断趋势和信号
            if score >= 7.0:
                trend = "强势"
                signal = "买入"
            elif score >= 6.0:
                trend = "偏强"
                signal = "观望"
            elif score >= 4.0:
                trend = "中性"
                signal = "观望"
            elif score >= 3.0:
                trend = "偏弱"
                signal = "减仓"
            else:
                trend = "弱势"
                signal = "卖出"
            
            # 构建分析结果
            analysis_result = {
                'code': stock_code,
                'name': self.data_fetcher.get_stock_name(stock_code),
                'score': score,
                'trend': trend,
                'signal': signal,
                'details': {
                    '成交量上升': volume_increasing,
                    '成交量高于20日均线': volume_above_ma20,
                    '相对成交量偏高': high_relative_volume,
                    '相对成交量偏低': low_relative_volume,
                    'OBV趋势上升': obv_increasing,
                    '成交量震荡指标为正': vo_positive,
                    '价量正相关': price_volume_positive_corr,
                    '价量负相关': price_volume_negative_corr,
                    '成交量价格比下降': vpr_decreasing,
                    '成交量突破': volume_breakout,
                    '放量上涨': volume_price_up,
                    '放量下跌': volume_price_down,
                    '底部背离(看涨)': bullish_divergence,
                    '顶部背离(看跌)': bearish_divergence,
                    '价格上涨': price_trend,
                    '量价配合': volume_price_alignment,
                    '最新成交量': last_volume['volume'],
                    '相对成交量': last_volume['relative_volume'] if 'relative_volume' in last_volume else None,
                    '价量相关性': last_corr['price_volume_corr'] if 'price_volume_corr' in last_corr else None
                }
            }
            
            logger.info(f"股票 {stock_code} 量价分析完成，趋势: {trend}, 评分: {score:.1f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"量价分析股票 {stock_code} 时出错: {e}")
            return {
                'code': stock_code,
                'name': self.data_fetcher.get_stock_name(stock_code),
                'score': 0,
                'trend': '未知',
                'signal': '分析出错',
                'details': {'错误': str(e)}
            }