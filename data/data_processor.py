#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理器模块 - 负责高级数据处理操作
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import concurrent.futures
from functools import lru_cache

# 导入自定义模块
from data.data_validator import FinancialDataValidator, adjust_price_for_splits, adjust_dividend_impact
from utils.cache_manager import CacheManager, cached
from data.data_source_manager import DataSourceManager

logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """
    增强版数据处理器 - 提供高级数据处理功能
    """
    
    def __init__(self):
        """初始化数据处理器"""
        logger.info("初始化增强版数据处理器")
        self.validator = FinancialDataValidator()
        self.data_source_manager = DataSourceManager.get_instance()
        self.cache_manager = CacheManager.get_instance()
        
        # 公司行为数据缓存
        self._corporate_actions = {}
        
    def clean_outliers(self, df, column, method='iqr', threshold=1.5):
        """
        清理异常值
        
        参数:
            df (pandas.DataFrame): 输入数据
            column (str): 要处理的列名
            method (str): 方法，'iqr'或'zscore'
            threshold (float): 阈值
            
        返回:
            pandas.DataFrame: 处理后的数据
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return df
        
        if column not in df.columns:
            logger.warning(f"列 '{column}' 不存在")
            return df
        
        result = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            result = result[mask]
            
            outliers_count = len(df) - len(result)
            logger.info(f"IQR法移除 {outliers_count} 条异常值, 下界: {lower_bound:.2f}, 上界: {upper_bound:.2f}")
            
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            
            z_scores = (df[column] - mean) / std
            mask = z_scores.abs() <= threshold
            result = result[mask]
            
            outliers_count = len(df) - len(result)
            logger.info(f"Z分数法移除 {outliers_count} 条异常值, 阈值: {threshold}")
            
        else:
            logger.warning(f"未知的异常值处理方法: {method}")
        
        return result
    
    def calculate_technical_indicators(self, df, ohlcv_columns=None):
        """
        计算技术指标
        
        参数:
            df (pandas.DataFrame): K线数据
            ohlcv_columns (dict): OHLCV列名映射
            
        返回:
            pandas.DataFrame: 包含技术指标的DataFrame
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return df
        
        # 默认列名映射
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # 检查所需列是否存在
        for col_type, col_name in ohlcv_columns.items():
            if col_name not in df.columns:
                logger.warning(f"列 '{col_name}' 不存在，无法计算技术指标")
                return df
        
        # 创建结果DataFrame的副本
        result = df.copy()
        
        # 获取OHLCV列
        open_col = ohlcv_columns['open']
        high_col = ohlcv_columns['high']
        low_col = ohlcv_columns['low']
        close_col = ohlcv_columns['close']
        volume_col = ohlcv_columns['volume']
        
        # 计算移动平均线 (MA)
        for window in [5, 10, 20, 60, 120]:
            result[f'ma{window}'] = df[close_col].rolling(window=window).mean()
        
        # 计算指数移动平均线 (EMA)
        for window in [5, 10, 20, 60]:
            result[f'ema{window}'] = df[close_col].ewm(span=window, adjust=False).mean()
        
        # 计算相对强弱指数 (RSI)
        delta = df[close_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        ema12 = df[close_col].ewm(span=12, adjust=False).mean()
        ema26 = df[close_col].ewm(span=26, adjust=False).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # 计算布林带
        result['bb_middle'] = df[close_col].rolling(window=20).mean()
        result['bb_std'] = df[close_col].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        
        # 计算平均真实范围 (ATR)
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - df[close_col].shift()).abs()
        tr3 = (df[low_col] - df[close_col].shift()).abs()
        result['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr'] = result['tr'].rolling(window=14).mean()
        
        # 计算价格变化率
        for period in [1, 5, 10, 20]:
            result[f'change_{period}d'] = df[close_col].pct_change(periods=period)
        
        # 计算累积涨跌幅
        result['cumulative_return'] = (1 + df[close_col].pct_change()).cumprod() - 1
        
        # 计算成交量变化率
        result['volume_change'] = df[volume_col].pct_change()
        
        # 计算成交量移动平均线
        for window in [5, 10, 20]:
            result[f'volume_ma{window}'] = df[volume_col].rolling(window=window).mean()
        
        # 计算相对成交量
        result['relative_volume'] = df[volume_col] / df[volume_col].rolling(window=20).mean()
        
        logger.info(f"计算技术指标完成，共添加 {len(result.columns) - len(df.columns)} 个指标")
        return result
    
    def detect_patterns(self, df, ohlcv_columns=None):
        """
        检测K线形态
        
        参数:
            df (pandas.DataFrame): K线数据
            ohlcv_columns (dict): OHLCV列名映射
            
        返回:
            pandas.DataFrame: 包含形态检测结果的DataFrame
        """
        if not isinstance(df, pd.DataFrame) or len(df) < 5:
            return df
        
        # 默认列名映射
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # 检查所需列是否存在
        for col_type, col_name in ohlcv_columns.items():
            if col_name not in df.columns:
                logger.warning(f"列 '{col_name}' 不存在，无法检测K线形态")
                return df
        
        # 创建结果DataFrame的副本
        result = df.copy()
        
        # 获取OHLCV列
        open_col = ohlcv_columns['open']
        high_col = ohlcv_columns['high']
        low_col = ohlcv_columns['low']
        close_col = ohlcv_columns['close']
        
        # 计算实体和影线
        result['body'] = (df[close_col] - df[open_col]).abs()
        result['upper_shadow'] = df[high_col] - df[[open_col, close_col]].max(axis=1)
        result['lower_shadow'] = df[[open_col, close_col]].min(axis=1) - df[low_col]
        
        # 计算K线颜色（阳线或阴线）
        result['is_bullish'] = df[close_col] > df[open_col]
        
        # 检测十字星形态
        body_threshold = df[high_col].rolling(window=20).mean() * 0.001  # 0.1%的波动作为实体阈值
        result['is_doji'] = result['body'] <= body_threshold
        
        # 检测锤子线形态
        result['is_hammer'] = (
            (result['lower_shadow'] > 2 * result['body']) &
            (result['upper_shadow'] < 0.5 * result['body']) &
            (result['body'] > 0)
        )
        
        # 检测上吊线形态
        result['is_hanging_man'] = (
            (result['lower_shadow'] > 2 * result['body']) &
            (result['upper_shadow'] < 0.5 * result['body']) &
            (result['body'] > 0) &
            (~result['is_bullish'])
        )
        
        # 检测吞没形态
        bullish_engulfing = (
            (~df[close_col].shift(1).isna()) &
            (df[close_col].shift(1) < df[open_col].shift(1)) &  # 前一天是阴线
            (df[close_col] > df[open_col]) &  # 当天是阳线
            (df[open_col] <= df[close_col].shift(1)) &  # 当天开盘低于等于前一天收盘
            (df[close_col] >= df[open_col].shift(1))  # 当天收盘高于等于前一天开盘
        )
        
        bearish_engulfing = (
            (~df[close_col].shift(1).isna()) &
            (df[close_col].shift(1) > df[open_col].shift(1)) &  # 前一天是阳线
            (df[close_col] < df[open_col]) &  # 当天是阴线
            (df[open_col] >= df[close_col].shift(1)) &  # 当天开盘高于等于前一天收盘
            (df[close_col] <= df[open_col].shift(1))  # 当天收盘低于等于前一天开盘
        )
        
        result['is_bullish_engulfing'] = bullish_engulfing
        result['is_bearish_engulfing'] = bearish_engulfing
        
        # 检测孕线形态
        bullish_harami = (
            (~df[close_col].shift(1).isna()) &
            (df[close_col].shift(1) < df[open_col].shift(1)) &  # 前一天是阴线
            (df[close_col] > df[open_col]) &  # 当天是阳线
            (df[open_col] > df[close_col].shift(1)) &  # 当天开盘高于前一天收盘
            (df[close_col] < df[open_col].shift(1))  # 当天收盘低于前一天开盘
        )
        
        bearish_harami = (
            (~df[close_col].shift(1).isna()) &
            (df[close_col].shift(1) > df[open_col].shift(1)) &  # 前一天是阳线
            (df[close_col] < df[open_col]) &  # 当天是阴线
            (df[open_col] < df[close_col].shift(1)) &  # 当天开盘低于前一天收盘
            (df[close_col] > df[open_col].shift(1))  # 当天收盘高于前一天开盘
        )
        
        result['is_bullish_harami'] = bullish_harami
        result['is_bearish_harami'] = bearish_harami
        
        # 检测星线形态
        morning_star = (
            (~df[close_col].shift(2).isna()) &
            (df[close_col].shift(2) < df[open_col].shift(2)) &  # 前前天是阴线
            (result['is_doji'].shift(1)) &  # 前一天是十字星
            (df[close_col] > df[open_col]) &  # 当天是阳线
            (df[close_col] > df[close_col].shift(2) + (df[open_col].shift(2) - df[close_col].shift(2)) / 2)  # 当天收盘至少回升到前前天实体的一半
        )
        
        evening_star = (
            (~df[close_col].shift(2).isna()) &
            (df[close_col].shift(2) > df[open_col].shift(2)) &  # 前前天是阳线
            (result['is_doji'].shift(1)) &  # 前一天是十字星
            (df[close_col] < df[open_col]) &  # 当天是阴线
            (df[close_col] < df[close_col].shift(2) - (df[close_col].shift(2) - df[open_col].shift(2)) / 2)  # 当天收盘至少下跌到前前天实体的一半
        )
        
        result['is_morning_star'] = morning_star
        result['is_evening_star'] = evening_star
        
        # 检测三兵/三卒形态
        three_white_soldiers = (
            (~df[close_col].shift(2).isna()) &
            (df[close_col].shift(2) > df[open_col].shift(2)) &  # 前前天是阳线
            (df[close_col].shift(1) > df[open_col].shift(1)) &  # 前一天是阳线
            (df[close_col] > df[open_col]) &  # 当天是阳线
            (df[close_col].shift(1) > df[close_col].shift(2)) &  # 连续上涨
            (df[close_col] > df[close_col].shift(1))
        )
        
        three_black_crows = (
            (~df[close_col].shift(2).isna()) &
            (df[close_col].shift(2) < df[open_col].shift(2)) &  # 前前天是阴线
            (df[close_col].shift(1) < df[open_col].shift(1)) &  # 前一天是阴线
            (df[close_col] < df[open_col]) &  # 当天是阴线
            (df[close_col].shift(1) < df[close_col].shift(2)) &  # 连续下跌
            (df[close_col] < df[close_col].shift(1))
        )
        
        result['is_three_white_soldiers'] = three_white_soldiers
        result['is_three_black_crows'] = three_black_crows
        
        logger.info(f"K线形态检测完成，共检测 {len(result.columns) - len(df.columns)} 种形态")
        return result
    
    def extract_features(self, df, base_features=None, window_sizes=[5, 10, 20, 60]):
        """
        提取特征
        
        参数:
            df (pandas.DataFrame): 输入数据
            base_features (list): 基础特征列表
            window_sizes (list): 窗口大小列表
            
        返回:
            pandas.DataFrame: 提取特征后的DataFrame
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return df
        
        # 默认使用所有数值列作为基础特征
        if base_features is None:
            base_features = df.select_dtypes(include=np.number).columns.tolist()
        
        # 创建结果DataFrame的副本
        result = df.copy()
        
        # 为每个基础特征计算统计量
        for feature in base_features:
            if feature not in df.columns:
                logger.warning(f"特征 '{feature}' 不存在，跳过")
                continue
            
            if not pd.api.types.is_numeric_dtype(df[feature]):
                logger.warning(f"特征 '{feature}' 不是数值类型，跳过")
                continue
            
            # 计算各窗口的统计量
            for window in window_sizes:
                # 移动平均
                result[f'{feature}_ma{window}'] = df[feature].rolling(window=window).mean()
                
                # 移动标准差
                result[f'{feature}_std{window}'] = df[feature].rolling(window=window).std()
                
                # 移动最大值和最小值
                result[f'{feature}_max{window}'] = df[feature].rolling(window=window).max()
                result[f'{feature}_min{window}'] = df[feature].rolling(window=window).min()
                
                # Z-Score标准化
                result[f'{feature}_zscore{window}'] = (
                    (df[feature] - result[f'{feature}_ma{window}']) / 
                    result[f'{feature}_std{window}']
                )
                
                # 相对位置
                result[f'{feature}_pos{window}'] = (
                    (df[feature] - result[f'{feature}_min{window}']) / 
                    (result[f'{feature}_max{window}'] - result[f'{feature}_min{window}'])
                )
                
                # 滞后特征
                for lag in range(1, min(6, window)):
                    result[f'{feature}_lag{lag}'] = df[feature].shift(lag)
                
                # 差分特征
                result[f'{feature}_diff{window}'] = df[feature].diff(window)
                
                # 变化率
                result[f'{feature}_pct{window}'] = df[feature].pct_change(window)
                
                # 滚动偏度和峰度
                result[f'{feature}_skew{window}'] = df[feature].rolling(window=window).skew()
                result[f'{feature}_kurt{window}'] = df[feature].rolling(window=window).kurt()
                
                # 滚动分位数
                result[f'{feature}_q25_{window}'] = df[feature].rolling(window=window).quantile(0.25)
                result[f'{feature}_q75_{window}'] = df[feature].rolling(window=window).quantile(0.75)
                
                # 滚动累积和与滚动累积积
                result[f'{feature}_cumsum{window}'] = df[feature].rolling(window=window).sum()
                
                # 滚动自相关性
                result[f'{feature}_autocorr{window}'] = df[feature].rolling(window=window).apply(
                    lambda x: x.autocorr() if len(x.dropna()) >= 2 else np.nan,
                    raw=False
                )
        
        logger.info(f"特征提取完成，从 {len(base_features)} 个基础特征生成了 {len(result.columns) - len(df.columns)} 个新特征")
        return result
    
    def process_with_metadata(self, df, metadata=None):
        """
        使用元数据处理数据
        
        参数:
            df (pandas.DataFrame): 输入数据
            metadata (dict): 数据元信息
            
        返回:
            pandas.DataFrame: 处理后的数据
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return df
        
        if metadata is None:
            logger.warning("未提供元数据，跳过处理")
            return df
        
        # 创建结果DataFrame的副本
        result = df.copy()
        
        # 根据元数据处理数据
        # 例如添加行业、市值等信息
        if 'industry' in metadata:
            result['industry'] = metadata['industry']
        
        if 'market_cap' in metadata:
            result['market_cap'] = metadata['market_cap']
        
        if 'stock_name' in metadata:
            result['stock_name'] = metadata['stock_name']
        
        # 添加其他元数据信息
        for key, value in metadata.items():
            if key not in result.columns:
                result[key] = value
        
        logger.info(f"使用元数据处理完成，添加了 {len(metadata)} 个元数据字段")
        return result
    
    def get_adjusted_price_data(self, stock_code, start_date=None, end_date=None, 
                                adjust_type='qfq', validate=True) -> pd.DataFrame:
        """
        获取前复权/后复权的价格数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust_type: 调整类型，'qfq'(前复权),'hfq'(后复权),'none'(不调整)
            validate: 是否验证数据
            
        Returns:
            pd.DataFrame: 调整后的价格数据
        """
        # 参数处理
        today = datetime.now().strftime("%Y%m%d")
        start_date = start_date or "20150101"
        end_date = end_date or today
        
        # 缓存键
        cache_key = f"price_{stock_code}_{start_date}_{end_date}_{adjust_type}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # 从数据源获取原始价格数据
        raw_data = self.data_source_manager.get_data(
            'daily_data',
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if raw_data.empty:
            self.logger.warning(f"无法获取股票{stock_code}的价格数据")
            return pd.DataFrame()
        
        # 验证数据
        if validate:
            valid, error_msg, raw_data = self.validator.validate_price_data(raw_data, strict=False)
            if not valid:
                self.logger.warning(f"股票{stock_code}数据验证警告: {error_msg}")
                
        # 获取除权除息和拆分数据
        corporate_actions = self._get_corporate_actions(stock_code)
        
        # 根据调整类型进行处理
        if adjust_type == 'qfq':  # 前复权
            if 'splits' in corporate_actions:
                raw_data = adjust_price_for_splits(raw_data, corporate_actions['splits'])
            if 'dividends' in corporate_actions:
                raw_data = adjust_dividend_impact(raw_data, corporate_actions['dividends'])
        elif adjust_type == 'hfq':  # 后复权
            # 后复权需要逆序处理adjustment_factor
            if 'splits' in corporate_actions:
                split_data = [(date, 1/ratio) for date, ratio in corporate_actions['splits']]
                raw_data = adjust_price_for_splits(raw_data, split_data)
            if 'dividends' in corporate_actions:
                dividend_data = [(date, -amount) for date, amount in corporate_actions['dividends']]
                raw_data = adjust_dividend_impact(raw_data, dividend_data)
        
        # 缓存结果
        self.cache_manager.set(cache_key, raw_data, expire_in=86400)  # 缓存一天
        
        return raw_data
        
    def get_multi_source_price_data(self, stock_code, start_date=None, end_date=None) -> pd.DataFrame:
        """
        从多个数据源获取价格数据并进行交叉验证
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 交叉验证后的价格数据
        """
        # 参数处理
        today = datetime.now().strftime("%Y%m%d")
        start_date = start_date or "20150101"
        end_date = end_date or today
        
        # 缓存键
        cache_key = f"multi_price_{stock_code}_{start_date}_{end_date}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        # 并行从多个数据源获取数据
        data_dict = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 从Tushare获取数据
            tushare_future = executor.submit(
                self._get_source_data, 
                'tushare', 
                stock_code, 
                start_date, 
                end_date
            )
            
            # 从AkShare获取数据
            akshare_future = executor.submit(
                self._get_source_data, 
                'akshare', 
                stock_code, 
                start_date, 
                end_date
            )
            
            # 收集结果
            for future, source_name in [(tushare_future, 'tushare'), (akshare_future, 'akshare')]:
                try:
                    data = future.result(timeout=10)
                    if not data.empty:
                        data_dict[source_name] = data
                except Exception as e:
                    self.logger.error(f"从{source_name}获取数据失败: {e}")
                    
        # 如果没有获取到任何数据
        if not data_dict:
            self.logger.warning(f"无法从任何数据源获取股票{stock_code}的数据")
            return pd.DataFrame()
            
        # 交叉验证数据
        merged_data, reliability_scores = self.validator.cross_validate_sources(data_dict)
        self.logger.info(f"数据源可信度评分: {reliability_scores}")
        
        # 缓存结果
        self.cache_manager.set(cache_key, merged_data, expire_in=86400)  # 缓存一天
        
        return merged_data
        
    def get_accurate_history_data(self, stock_code, field='price', start_date=None, end_date=None, 
                                adjust_type='qfq', validate=True) -> pd.DataFrame:
        """
        获取经过多重验证和调整的高精度历史数据
        
        Args:
            stock_code: 股票代码
            field: 数据字段，'price'(价格),'finance'(财务),'dividend'(分红)等
            start_date: 开始日期
            end_date: 结束日期
            adjust_type: 调整类型
            validate: 是否验证数据
            
        Returns:
            pd.DataFrame: 高精度历史数据
        """
        # 缓存键
        cache_key = f"accurate_{field}_{stock_code}_{start_date}_{end_date}_{adjust_type}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        # 根据字段类型选择不同的处理方法
        if field == 'price':
            # 获取交叉验证后的价格数据
            raw_data = self.get_multi_source_price_data(stock_code, start_date, end_date)
            
            # 验证并修正数据
            if validate:
                valid, error_msg, raw_data = self.validator.validate_price_data(raw_data, strict=False)
                if not valid:
                    self.logger.warning(f"股票{stock_code}数据验证警告: {error_msg}")
                    
            # 应用公司行为调整
            if adjust_type != 'none':
                corporate_actions = self._get_corporate_actions(stock_code)
                
                if adjust_type == 'qfq':  # 前复权
                    if 'splits' in corporate_actions:
                        raw_data = adjust_price_for_splits(raw_data, corporate_actions['splits'])
                    if 'dividends' in corporate_actions:
                        raw_data = adjust_dividend_impact(raw_data, corporate_actions['dividends'])
                elif adjust_type == 'hfq':  # 后复权
                    if 'splits' in corporate_actions:
                        split_data = [(date, 1/ratio) for date, ratio in corporate_actions['splits']]
                        raw_data = adjust_price_for_splits(raw_data, split_data)
                    if 'dividends' in corporate_actions:
                        dividend_data = [(date, -amount) for date, amount in corporate_actions['dividends']]
                        raw_data = adjust_dividend_impact(raw_data, dividend_data)
                        
        elif field == 'finance':
            # 获取财务数据并验证
            raw_data = self.data_source_manager.get_data(
                'financial_data',
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if validate and not raw_data.empty:
                valid, error_msg, raw_data = self.validator.validate_financial_indicator(
                    raw_data, indicator_type='profit'
                )
                if not valid:
                    self.logger.warning(f"股票{stock_code}财务数据验证警告: {error_msg}")
                    
        elif field == 'dividend':
            # 获取分红数据
            raw_data = self._get_dividend_data(stock_code, start_date, end_date)
            
        else:
            self.logger.warning(f"未知的数据字段类型: {field}")
            return pd.DataFrame()
            
        # 缓存结果
        self.cache_manager.set(cache_key, raw_data, expire_in=86400)  # 缓存一天
        
        return raw_data
        
    def get_real_trading_dates(self, exchange='SSE', start_date=None, end_date=None) -> List[str]:
        """
        获取真实的交易日历
        
        Args:
            exchange: 交易所，'SSE'(上交所),'SZSE'(深交所)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[str]: 交易日列表
        """
        # 参数处理
        today = datetime.now().strftime("%Y%m%d")
        start_date = start_date or "20150101"
        end_date = end_date or today
        
        # 缓存键
        cache_key = f"trading_dates_{exchange}_{start_date}_{end_date}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        # 从数据源获取交易日历
        calendar_data = self.data_source_manager.get_data(
            'trade_cal',
            exchange=exchange,
            start_date=start_date,
            end_date=end_date
        )
        
        if calendar_data.empty:
            self.logger.warning(f"无法获取交易所{exchange}的交易日历")
            return []
            
        # 提取是交易日的日期
        date_col = next((col for col in calendar_data.columns if 'date' in col.lower()), None)
        is_open_col = next((col for col in calendar_data.columns if 'open' in col.lower() or 'is_trading' in col.lower()), None)
        
        if not date_col or not is_open_col:
            self.logger.warning("交易日历数据缺少必要的列")
            return []
            
        trading_dates = calendar_data[calendar_data[is_open_col] == 1][date_col].tolist()
        
        # 格式化日期
        if isinstance(trading_dates[0], pd.Timestamp):
            trading_dates = [d.strftime("%Y%m%d") for d in trading_dates]
            
        # 缓存结果
        self.cache_manager.set(cache_key, trading_dates, expire_in=604800)  # 缓存一周
        
        return trading_dates
        
    def fill_missing_trading_dates(self, df: pd.DataFrame, date_col='date', method='ffill'):
        """
        填充缺失的交易日数据
        
        Args:
            df: 数据DataFrame
            date_col: 日期列名
            method: 填充方法，'ffill'(前向填充),'bfill'(后向填充),'interpolate'(插值)
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        if df.empty:
            return df
            
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            
        # 获取起止日期
        start_date = df[date_col].min().strftime("%Y%m%d")
        end_date = df[date_col].max().strftime("%Y%m%d")
        
        # 获取该时间段的所有交易日
        trading_dates = self.get_real_trading_dates(start_date=start_date, end_date=end_date)
        trading_dates = pd.to_datetime(trading_dates)
        
        # 创建一个包含所有交易日的DataFrame
        date_df = pd.DataFrame({date_col: trading_dates})
        
        # 与原始数据合并
        merged_df = pd.merge(date_df, df, on=date_col, how='left')
        
        # 按指定方法填充缺失值
        if method == 'ffill':
            merged_df = merged_df.ffill()
        elif method == 'bfill':
            merged_df = merged_df.bfill()
        elif method == 'interpolate':
            numeric_cols = merged_df.select_dtypes(include=np.number).columns
            merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
            # 非数值列用前向填充
            non_numeric_cols = [col for col in merged_df.columns if col not in numeric_cols and col != date_col]
            if non_numeric_cols:
                merged_df[non_numeric_cols] = merged_df[non_numeric_cols].ffill().bfill()
                
        return merged_df
        
    def _get_corporate_actions(self, stock_code):
        """获取股票的公司行为数据（除权除息、拆分等）"""
        if stock_code in self._corporate_actions:
            return self._corporate_actions[stock_code]
            
        # 获取除权除息数据
        dividend_data = self._get_dividend_data(stock_code)
        
        # 获取股票拆分数据
        split_data = self._get_split_data(stock_code)
        
        # 整合数据
        corporate_actions = {}
        if not dividend_data.empty:
            # 提取日期和分红金额
            date_col = next((col for col in dividend_data.columns if 'date' in col.lower() or 'div' in col.lower()), None)
            amount_col = next((col for col in dividend_data.columns if 'cash' in col.lower() or 'amount' in col.lower()), None)
            
            if date_col and amount_col:
                corporate_actions['dividends'] = [
                    (row[date_col], row[amount_col]) 
                    for _, row in dividend_data.iterrows() 
                    if pd.notna(row[amount_col]) and row[amount_col] > 0
                ]
                
        if not split_data.empty:
            # 提取日期和拆分比例
            date_col = next((col for col in split_data.columns if 'date' in col.lower() or 'ann' in col.lower()), None)
            ratio_col = next((col for col in split_data.columns if 'ratio' in col.lower() or 'split' in col.lower()), None)
            
            if date_col and ratio_col:
                corporate_actions['splits'] = [
                    (row[date_col], row[ratio_col]) 
                    for _, row in split_data.iterrows() 
                    if pd.notna(row[ratio_col]) and row[ratio_col] != 1
                ]
                
        # 缓存结果
        self._corporate_actions[stock_code] = corporate_actions
        
        return corporate_actions
        
    def _get_dividend_data(self, stock_code, start_date=None, end_date=None):
        """获取股票的分红数据"""
        # 参数处理
        today = datetime.now().strftime("%Y%m%d")
        start_date = start_date or "20100101"
        end_date = end_date or today
        
        # 缓存键
        cache_key = f"dividend_{stock_code}_{start_date}_{end_date}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        # 从数据源获取分红数据
        dividend_data = self.data_source_manager.get_data(
            'dividend',
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # 缓存结果
        self.cache_manager.set(cache_key, dividend_data, expire_in=604800)  # 缓存一周
        
        return dividend_data
        
    def _get_split_data(self, stock_code, start_date=None, end_date=None):
        """获取股票的拆分数据"""
        # 参数处理
        today = datetime.now().strftime("%Y%m%d")
        start_date = start_date or "20100101"
        end_date = end_date or today
        
        # 缓存键
        cache_key = f"split_{stock_code}_{start_date}_{end_date}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        # 从数据源获取拆分数据
        split_data = self.data_source_manager.get_data(
            'stock_splits',
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # 缓存结果
        self.cache_manager.set(cache_key, split_data, expire_in=604800)  # 缓存一周
        
        return split_data
        
    def _get_source_data(self, source_name, stock_code, start_date, end_date):
        """从指定数据源获取数据"""
        try:
            if source_name == 'tushare':
                data = self.data_source_manager.data_sources['tushare'].get_data_safe(
                    api_name='daily',
                    ts_code=stock_code,
                    start_date=start_date,
                    end_date=end_date
                )
                return data
            elif source_name == 'akshare':
                # AkShare接口适配
                data = self.data_source_manager.data_sources['akshare'].get_data_safe(
                    func_name='stock_zh_a_daily',
                    symbol=stock_code.replace('.SH', '').replace('.SZ', '')
                )
                return data
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"从{source_name}获取{stock_code}数据失败: {e}")
            return pd.DataFrame() 