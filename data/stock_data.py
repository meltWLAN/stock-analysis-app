#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票数据获取模块 - 负责从各数据源获取股票行情数据
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 尝试导入数据源包
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("Tushare未安装，部分功能将不可用")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logging.warning("AkShare未安装，部分功能将不可用")


class StockDataFetcher:
    """股票数据获取类"""
    
    def __init__(self, token=None, cache_expire=3600):
        """
        初始化数据获取器
        
        Args:
            token: Tushare API token，如果为None则尝试从环境变量获取
            cache_expire: 缓存过期时间(秒)，默认1小时
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化Tushare
        if TUSHARE_AVAILABLE:
            self.token = token or os.environ.get('TUSHARE_TOKEN')
            if self.token:
                try:
                    ts.set_token(self.token)
                    self.pro = ts.pro_api()
                    self.logger.info("Tushare API初始化成功")
                except Exception as e:
                    self.logger.error(f"Tushare API初始化失败: {e}")
            else:
                self.logger.warning("未设置Tushare Token，相关功能将不可用")
        
        # 缓存数据和过期时间设置
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_expire = cache_expire
    
    def get_stock_list(self):
        """
        获取A股股票列表
        
        Returns:
            DataFrame: 股票代码、名称等基本信息
        """
        if TUSHARE_AVAILABLE and hasattr(self, 'pro'):
            try:
                df = self.pro.stock_basic(exchange='', list_status='L', 
                                         fields='ts_code,symbol,name,area,industry,list_date')
                return df
            except Exception as e:
                self.logger.error(f"获取股票列表失败: {e}")
        
        if AKSHARE_AVAILABLE:
            try:
                df = ak.stock_info_a_code_name()
                return df
            except Exception as e:
                self.logger.error(f"通过AkShare获取股票列表失败: {e}")
        
        self.logger.error("无法获取股票列表，请检查数据源配置")
        return pd.DataFrame()
    
    def _is_cache_valid(self, cache_key):
        """
        检查缓存是否有效
        
        Args:
            cache_key: 缓存键
            
        Returns:
            bool: 缓存是否有效
        """
        if cache_key not in self.cache:
            return False
            
        if cache_key not in self.cache_timestamps:
            return False
            
        # 检查缓存是否过期
        timestamp = self.cache_timestamps[cache_key]
        now = datetime.now().timestamp()
        return (now - timestamp) < self.cache_expire
    
    def _update_cache(self, cache_key, data):
        """
        更新缓存
        
        Args:
            cache_key: 缓存键
            data: 缓存数据
        """
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now().timestamp()
    
    def get_daily_data(self, stock_code, start_date=None, end_date=None):
        """
        获取股票日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，默认为60天前
            end_date: 结束日期，默认为今天
            
        Returns:
            DataFrame: 日线数据，包含OHLCV等信息
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        # 检查股票代码格式
        if not stock_code:
            self.logger.error("股票代码不能为空")
            return pd.DataFrame()
        
        # 缓存键
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"使用缓存数据: {cache_key}")
            return self.cache[cache_key]
        
        # 尝试从Tushare获取
        if TUSHARE_AVAILABLE and hasattr(self, 'pro'):
            try:
                df = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    # 按日期排序
                    df = df.sort_values('trade_date')
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从Tushare获取{stock_code}日线数据失败: {e}")
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 转换股票代码格式
                if len(stock_code) == 9:  # Tushare格式 (000001.SZ)
                    symbol = stock_code.split('.')[0]
                    market = 'sh' if stock_code.endswith('SH') else 'sz'
                    ak_code = f"{market}{symbol}"
                else:
                    ak_code = stock_code
                
                df = ak.stock_zh_a_hist(symbol=ak_code, period="daily", 
                                       start_date=start_date, end_date=end_date)
                if not df.empty:
                    # 重命名列以保持一致性
                    df = df.rename(columns={
                        '日期': 'trade_date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'vol',
                        '成交额': 'amount'
                    })
                    # 确保日期格式一致
                    if 'trade_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从AkShare获取{stock_code}日线数据失败: {e}")
        
        self.logger.error(f"无法获取{stock_code}的日线数据，请检查数据源配置")
        return pd.DataFrame()
    
    def get_index_data(self, index_code, start_date=None, end_date=None):
        """
        获取指数日线数据
        
        Args:
            index_code: 指数代码，如'000300.SH'(沪深300),'000905.SH'(中证500)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 指数日线数据
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 缓存键
        cache_key = f"IDX_{index_code}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"使用缓存的指数数据: {cache_key}")
            return self.cache[cache_key]
        
        # 尝试从Tushare获取
        if TUSHARE_AVAILABLE and hasattr(self, 'pro'):
            try:
                df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df = df.sort_values('trade_date')
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从Tushare获取{index_code}指数数据失败: {e}")
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 转换指数代码
                if index_code == '000300.SH':
                    ak_index = '000300'
                elif index_code == '000905.SH':
                    ak_index = '000905'
                else:
                    ak_index = index_code.split('.')[0]
                
                df = ak.stock_zh_index_daily(symbol=ak_index)
                if not df.empty:
                    # 筛选日期
                    df['date'] = pd.to_datetime(df['date'])
                    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
                    df = df[mask]
                    
                    # 重命名列
                    df = df.rename(columns={
                        'date': 'trade_date',
                        'open': 'open',
                        'close': 'close',
                        'high': 'high',
                        'low': 'low',
                        'volume': 'vol'
                    })
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从AkShare获取{index_code}指数数据失败: {e}")
        
        self.logger.error(f"无法获取{index_code}的指数数据，请检查数据源配置")
        return pd.DataFrame()
    
    def get_north_money_flow(self, start_date=None, end_date=None):
        """
        获取北向资金流入数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 北向资金流入数据
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 缓存键
        cache_key = f"NORTH_FLOW_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"使用缓存的北向资金数据: {cache_key}")
            return self.cache[cache_key]
        
        # 尝试从Tushare获取
        if TUSHARE_AVAILABLE and hasattr(self, 'pro'):
            try:
                df = self.pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
                if not df.empty:
                    df = df.sort_values('trade_date')
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从Tushare获取北向资金数据失败: {e}")
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 使用新的AkShare API获取北向资金数据
                # 替换旧的stock_em_hsgt_north_net_flow_in为新的stock_hsgt_fund_flow_summary_em
                df = ak.stock_hsgt_fund_flow_summary_em()
                
                if not df.empty:
                    # 转换并处理数据
                    df['日期'] = pd.to_datetime(df['日期'])
                    mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
                    df = df[mask]
                    
                    # 重命名列，确保与原代码兼容
                    df = df.rename(columns={
                        '日期': 'trade_date',
                        '北上资金': 'north_money',
                        '沪股通': 'value_sh',
                        '深股通': 'value_sz'
                    })
                    
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从AkShare获取北向资金数据失败: {e}")
        
        self.logger.error("无法获取北向资金数据，请检查数据源配置")
        return pd.DataFrame()
    
    def get_money_flow(self, stock_code, start_date=None, end_date=None):
        """
        获取个股资金流向数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 资金流向数据
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        # 缓存键
        cache_key = f"FLOW_{stock_code}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"使用缓存的资金流向数据: {cache_key}")
            return self.cache[cache_key]
        
        # 尝试从Tushare获取
        if TUSHARE_AVAILABLE and hasattr(self, 'pro'):
            try:
                df = self.pro.moneyflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    df = df.sort_values('trade_date')
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从Tushare获取{stock_code}资金流向数据失败: {e}")
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 转换股票代码格式
                if len(stock_code) == 9:  # Tushare格式 (000001.SZ)
                    symbol = stock_code.split('.')[0]
                else:
                    symbol = stock_code
                
                df = ak.stock_individual_fund_flow(stock=symbol)
                if not df.empty:
                    # 转换日期并筛选
                    df['日期'] = pd.to_datetime(df['日期'])
                    mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
                    df = df[mask]
                    
                    # 重命名列
                    df = df.rename(columns={
                        '日期': 'trade_date',
                        '开盘': 'open',
                        '收盘': 'close',
                        '最高': 'high',
                        '最低': 'low',
                        '成交量': 'vol',
                        '成交额': 'amount'
                    })
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从AkShare获取{stock_code}资金流向数据失败: {e}")
        
        self.logger.error(f"无法获取{stock_code}的资金流向数据，请检查数据源配置")
        return pd.DataFrame()
    
    def get_stock_kline(self, stock_code, start_date=None, end_date=None, frequency='daily'):
        """
        获取个股K线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期，默认为近180天
            end_date: 结束日期，默认为当天
            frequency: K线周期，daily-日线，weekly-周线，monthly-月线，默认为日线
            
        Returns:
            DataFrame: K线数据，包含开高低收、成交量等
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        
        # 缓存键
        cache_key = f"KLINE_{stock_code}_{frequency}_{start_date}_{end_date}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"使用缓存的K线数据: {cache_key}")
            return self.cache[cache_key]
        
        # 尝试从Tushare获取
        if TUSHARE_AVAILABLE and hasattr(self, 'pro'):
            try:
                if frequency == 'daily':
                    df = self.pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
                elif frequency == 'weekly':
                    df = self.pro.weekly(ts_code=stock_code, start_date=start_date, end_date=end_date)
                elif frequency == 'monthly':
                    df = self.pro.monthly(ts_code=stock_code, start_date=start_date, end_date=end_date)
                else:
                    self.logger.error(f"不支持的K线周期: {frequency}")
                    return pd.DataFrame()
                
                if not df.empty:
                    df = df.sort_values('trade_date')
                    # 确保日期是datetime类型
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    # 设置日期为索引
                    df.set_index('trade_date', inplace=True)
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从Tushare获取{stock_code}K线数据失败: {e}")
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 转换股票代码格式
                if len(stock_code) == 9:  # Tushare格式 (000001.SZ)
                    code = stock_code.split('.')[0]
                    market = 'sh' if stock_code.endswith('.SH') else 'sz'
                    symbol = f"{market}{code}"
                else:
                    symbol = stock_code
                
                # 根据不同周期获取数据
                if frequency == 'daily':
                    df = ak.stock_zh_a_hist(symbol=symbol, start_date=start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:], 
                                           end_date=end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:])
                elif frequency == 'weekly':
                    df = ak.stock_zh_a_hist_min_em(symbol=symbol, period='weekly', start_date=start_date, end_date=end_date)
                elif frequency == 'monthly':
                    df = ak.stock_zh_a_hist_min_em(symbol=symbol, period='monthly', start_date=start_date, end_date=end_date)
                else:
                    self.logger.error(f"不支持的K线周期: {frequency}")
                    return pd.DataFrame()
                
                if not df.empty:
                    # 重命名列
                    if '日期' in df.columns:
                        df = df.rename(columns={
                            '日期': 'trade_date',
                            '开盘': 'open',
                            '收盘': 'close',
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'vol',
                            '成交额': 'amount'
                        })
                    
                    # 确保日期是datetime类型
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    # 设置日期为索引
                    df.set_index('trade_date', inplace=True)
                    self._update_cache(cache_key, df)
                    return df
            except Exception as e:
                self.logger.error(f"从AkShare获取{stock_code}K线数据失败: {e}")
        
        self.logger.error(f"无法获取{stock_code}的K线数据，请检查数据源配置")
        return pd.DataFrame()
    
    def clear_cache(self, pattern=None):
        """
        清理缓存数据
        
        Args:
            pattern: 缓存键匹配模式，如果为None则清理所有缓存
        
        Returns:
            int: 清理的缓存数量
        """
        count = 0
        if pattern is None:
            # 清理所有缓存
            count = len(self.cache)
            self.cache.clear()
            self.cache_timestamps.clear()
            self.logger.info(f"已清理全部缓存数据: {count}条")
        else:
            # 按模式清理
            keys_to_remove = []
            for key in self.cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                count += 1
            
            self.logger.info(f"已清理匹配'{pattern}'的缓存数据: {count}条")
        
        return count
    
    def get_industry_list(self):
        """
        获取东方财富行业分类列表
        
        Returns:
            list: 行业分类列表
        """
        # 尝试从缓存获取
        cache_key = "INDUSTRY_LIST"
        if self._is_cache_valid(cache_key):
            self.logger.debug("使用缓存的行业分类数据")
            return self.cache[cache_key]
            
        industry_list = ["全部"]  # 默认第一项
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 获取东方财富行业分类
                df = ak.stock_board_industry_name_em()
                if not df.empty and '板块名称' in df.columns:
                    # 添加所有行业名称
                    industry_list.extend(df['板块名称'].tolist())
                    
                    # 缓存结果
                    self._update_cache(cache_key, industry_list)
                    return industry_list
            except Exception as e:
                self.logger.error(f"从AkShare获取行业分类失败: {e}")
        
        # 如果获取失败，返回一些常见行业作为备选
        backup_industries = [
            "全部", "银行", "证券", "保险", "房地产", "医药制造", "医疗器械", 
            "电子元件", "计算机设备", "软件开发", "互联网服务", "通信设备", 
            "电力", "化工", "钢铁", "有色金属", "汽车制造", "机械设备", 
            "家用电器", "食品饮料", "纺织服装", "建筑材料", "商业贸易"
        ]
        self.logger.warning("无法获取东方财富行业分类，使用备用行业列表")
        
        # 缓存备用行业列表
        self._update_cache(cache_key, backup_industries)
        return backup_industries
        
    def get_industry_stocks(self, industry_name):
        """
        获取指定行业的所有股票
        
        Args:
            industry_name: 行业名称
            
        Returns:
            DataFrame: 该行业的股票列表
        """
        if industry_name == "全部":
            return self.get_stock_list()
            
        # 缓存键
        cache_key = f"INDUSTRY_STOCKS_{industry_name}"
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"使用缓存的行业股票数据: {industry_name}")
            return self.cache[cache_key]
        
        # 尝试从AkShare获取
        if AKSHARE_AVAILABLE:
            try:
                # 获取东方财富行业板块成分股
                df = ak.stock_board_industry_cons_em(symbol=industry_name)
                if not df.empty:
                    # 处理返回的数据
                    result_df = df.rename(columns={
                        '代码': 'symbol',
                        '名称': 'name'
                    })
                    
                    # 添加ts_code列（Tushare格式）
                    result_df['ts_code'] = result_df['symbol'].apply(
                        lambda x: x + '.SH' if x.startswith('6') else x + '.SZ'
                    )
                    
                    # 缓存结果
                    self._update_cache(cache_key, result_df)
                    return result_df
            except Exception as e:
                self.logger.error(f"从AkShare获取行业股票列表失败: {industry_name}, 错误: {e}")
        
        # 如果获取失败，返回空DataFrame
        self.logger.error(f"无法获取行业'{industry_name}'的股票列表")
        return pd.DataFrame()
        
    def get_hot_industries_data(self):
        """
        获取热门行业数据，包含行业涨跌幅、上涨家数、下跌家数及领涨股等指标
        
        Returns:
            list: 热门行业数据列表
        """
        try:
            self.logger.info("开始获取热门行业数据")
            
            # 获取行业列表
            industry_list = self.get_industry_list()
            if not industry_list:
                self.logger.error("获取行业列表失败")
                return []
            
            # 保留行业数量，不超过30个
            sample_size = min(30, len(industry_list))
            result = []
            
            # 1. 获取大盘指标作为对比基准
            market_index = self.get_stock_kline('000001.SH')  # 上证指数
            market_change = 0
            
            if market_index is not None and not market_index.empty and len(market_index) > 1:
                market_change = (market_index['close'].iloc[-1] / market_index['close'].iloc[-5] - 1) * 100
                self.logger.info(f"大盘5日涨跌幅: {market_change:.2f}%")
            
            # 2. 获取行业热度数据
            # 为每个行业获取多因子数据
            for industry in industry_list[:sample_size]:  # 取样本行业
                try:
                    # 获取行业成分股
                    stocks = self.get_industry_stocks(industry)
                    if stocks is not None and not stocks.empty:
                        # 计算行业整体表现
                        total_change = 0           # 涨跌幅因子
                        total_volume_change = 0    # 成交量变化因子
                        total_turnover = 0         # 换手率因子
                        up_stocks = 0              # 上涨股票数量
                        down_stocks = 0            # 下跌股票数量
                        leading_stock = ""         # 龙头股
                        lead_stock_change = 0      # 龙头股涨幅
                        max_change = -100
                        
                        # 取样本股票分析，避免处理时间过长
                        max_stocks = min(20, len(stocks))  # 最多处理20个股票
                        processed_stocks = 0
                        
                        # 用于计算行业Beta和动量的数据
                        price_changes_5d = []
                        price_changes_10d = []
                        price_changes_20d = []
                        volume_changes = []
                        
                        for idx, (_, stock) in enumerate(stocks.iterrows()):
                            if idx >= max_stocks:
                                break
                                
                            try:
                                if 'ts_code' in stock:
                                    stock_code = stock['ts_code']
                                    # 获取K线数据
                                    kline = self.get_stock_kline(stock_code)
                                    if kline is not None and not kline.empty and len(kline) > 5:  # 至少需要5天数据
                                        # 计算短期、中期、长期涨跌幅
                                        change_5d = (kline['close'].iloc[-1] / kline['close'].iloc[-5] - 1) * 100
                                        price_changes_5d.append(change_5d)
                                        
                                        if len(kline) > 10:
                                            change_10d = (kline['close'].iloc[-1] / kline['close'].iloc[-10] - 1) * 100
                                            price_changes_10d.append(change_10d)
                                        
                                        if len(kline) > 20:
                                            change_20d = (kline['close'].iloc[-1] / kline['close'].iloc[-20] - 1) * 100
                                            price_changes_20d.append(change_20d)
                                        
                                        # 计算成交量变化
                                        if 'vol' in kline.columns and len(kline) > 10:
                                            vol_change = (kline['vol'].iloc[-5:].mean() / kline['vol'].iloc[-10:-5].mean() - 1) * 100
                                            volume_changes.append(vol_change)
                                            total_volume_change += vol_change
                                        
                                        # 统计短期涨跌幅
                                        total_change += change_5d
                                        
                                        # 统计上涨/下跌股票数量
                                        if change_5d > 0:
                                            up_stocks += 1
                                        else:
                                            down_stocks += 1
                                        
                                        # 计算换手率（如果有数据）
                                        if 'turnover' in kline.columns:
                                            total_turnover += kline['turnover'].iloc[-5:].mean()
                                        
                                        # 识别龙头股
                                        if change_5d > max_change:
                                            max_change = change_5d
                                            leading_stock = stock.get('name', stock_code)
                                            lead_stock_change = change_5d
                                        
                                        processed_stocks += 1
                            except Exception as e:
                                self.logger.warning(f"处理行业股票 {stock.get('ts_code', '未知')} 数据时出错: {str(e)}")
                                continue
                        
                        # 只有有足够数据的行业才进行分析
                        if processed_stocks > 0:
                            # 计算多因子得分
                            # 1. 价格动量因子 (5日、10日、20日权重递减)
                            momentum_score = 0
                            if price_changes_5d:
                                momentum_score += np.mean(price_changes_5d) * 0.5
                            if price_changes_10d:
                                momentum_score += np.mean(price_changes_10d) * 0.3
                            if price_changes_20d:
                                momentum_score += np.mean(price_changes_20d) * 0.2
                            
                            # 2. 成交量因子
                            volume_score = np.mean(volume_changes) if volume_changes else 0
                            
                            # 3. 上涨比例因子
                            up_ratio = (up_stocks / processed_stocks) * 100 if processed_stocks > 0 else 0
                            
                            # 4. 换手率因子
                            turnover_score = total_turnover / processed_stocks if processed_stocks > 0 else 0
                            
                            # 计算综合得分 (根据重要性设置权重)
                            composite_score = (
                                momentum_score * 0.4 +  # 涨跌幅权重40%
                                volume_score * 0.3 +    # 成交量变化权重30%
                                up_ratio * 0.2 +        # 上涨比例权重20%
                                turnover_score * 0.1    # 换手率权重10%
                            )
                            
                            # 计算行业相对强度 (相对于大盘)
                            relative_strength = momentum_score - market_change
                            
                            # 计算行业关注度指数 (50-100之间的值)
                            attention_score = min(100, max(50, 50 + relative_strength + volume_score * 0.5))
                            
                            # 构建行业结果数据
                            industry_item = {
                                'name': industry,
                                'change': f"{momentum_score:.2f}",  # 使用动量评分作为行业涨跌幅
                                'change_num': float(f"{momentum_score:.2f}"),  # 用于排序
                                'volume_change': f"{volume_score:.2f}",
                                'up_ratio': f"{up_ratio:.2f}",
                                'up_count': up_stocks,  # 上涨家数
                                'down_count': down_stocks,  # 下跌家数
                                'turnover': f"{turnover_score:.2f}",
                                'relative_strength': f"{relative_strength:.2f}",
                                'composite_score': composite_score,
                                'lead_stock': leading_stock,  # 领涨股
                                'lead_stock_change': f"{lead_stock_change:.2f}",  # 领涨股涨幅
                                'attention_index': int(attention_score),
                                'momentum_score': momentum_score,
                                'stocks_count': processed_stocks
                            }
                            
                            result.append(industry_item)
                            
                except Exception as e:
                    self.logger.error(f"处理行业 {industry} 数据时出错: {str(e)}")
                    continue
            
            # 3. 根据综合评分对行业排序
            result.sort(key=lambda x: x['change_num'], reverse=True)
            
            self.logger.info(f"成功获取 {len(result)} 个热门行业数据")
            return result
            
        except Exception as e:
            self.logger.error(f"获取热门行业数据失败: {str(e)}")
            return []