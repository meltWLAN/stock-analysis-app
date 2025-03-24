#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据验证器模块 - 负责验证数据的完整性和准确性
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

class FinancialDataValidator:
    """
    金融数据验证器 - 验证金融数据的质量和一致性
    """
    
    def __init__(self):
        """初始化数据验证器"""
        logger.info("初始化金融数据验证器")
    
    def validate_stock_code(self, stock_code):
        """
        验证股票代码格式
        
        参数:
            stock_code (str): 股票代码
            
        返回:
            bool: 是否有效
        """
        if not isinstance(stock_code, str):
            logger.warning(f"股票代码必须是字符串: {stock_code}")
            return False
        
        # 检查基本格式
        valid_formats = [
            # A股标准格式（6位数字+交易所后缀）
            r'^\d{6}\.(SH|SZ|BJ)$',
            # 港股格式（0开头5位数+交易所后缀或00开头4位数+交易所后缀）
            r'^0\d{4}\.HK$|^00\d{3}\.HK$',
            # 美股格式（字母+可能的点号和数字）
            r'^[A-Z]+(\.[A-Z])?$'
        ]
        
        import re
        for pattern in valid_formats:
            if re.match(pattern, stock_code):
                return True
        
        logger.warning(f"股票代码格式不正确: {stock_code}")
        return False
    
    def validate_date_range(self, start_date, end_date):
        """
        验证日期范围
        
        参数:
            start_date (str, datetime): 开始日期
            end_date (str, datetime): 结束日期
            
        返回:
            bool: 是否有效
        """
        try:
            # 转换为datetime类型
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 检查日期范围
            if start_date > end_date:
                logger.warning(f"开始日期 {start_date} 晚于结束日期 {end_date}")
                return False
            
            # 检查是否超过合理范围
            earliest_date = datetime(1990, 1, 1)  # 中国股市始于1990年
            if start_date < earliest_date:
                logger.warning(f"开始日期 {start_date} 早于有效期 {earliest_date}")
                return False
            
            if end_date > datetime.now() + timedelta(days=1):
                logger.warning(f"结束日期 {end_date} 晚于当前日期")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"日期验证失败: {e}")
            return False
    
    def validate_price_data(self, df):
        """
        验证价格数据的有效性
        
        参数:
            df (pandas.DataFrame): 价格数据
            
        返回:
            dict: 验证结果
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return {'valid': False, 'errors': ['数据为空或格式不正确']}
        
        errors = []
        warnings = []
        
        # 检查必要列是否存在
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"缺少必要列: {', '.join(missing_columns)}")
        
        # 价格数据基本检查
        if 'close' in df.columns:
            # 检查负值
            if (df['close'] < 0).any():
                errors.append("收盘价存在负值")
            
            # 检查异常值
            mean_price = df['close'].mean()
            std_price = df['close'].std()
            max_price = df['close'].max()
            min_price = df['close'].min()
            
            if max_price > mean_price + 5 * std_price:
                warnings.append(f"存在疑似异常高价: {max_price:.2f}")
            
            if min_price < mean_price - 5 * std_price and min_price > 0:
                warnings.append(f"存在疑似异常低价: {min_price:.2f}")
            
            # 检查价格跳跃
            price_change = df['close'].pct_change().abs()
            max_change = price_change.max()
            if max_change > 0.2:  # 单日涨跌幅超过20%
                warnings.append(f"存在异常价格跳跃: {max_change*100:.2f}%")
        
        # 价格逻辑关系检查
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 检查高低价关系
            invalid_hl = ((df['high'] < df['low']) | (df['high'] < 0) | (df['low'] < 0)).sum()
            if invalid_hl > 0:
                errors.append(f"存在 {invalid_hl} 条高低价逻辑错误")
            
            # 检查开盘价和收盘价是否在高低价之间
            invalid_oc = ((df['open'] > df['high']) | (df['open'] < df['low']) | 
                          (df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if invalid_oc > 0:
                errors.append(f"存在 {invalid_oc} 条开收盘价超出高低价范围")
        
        # 检查日期连续性
        if df.index.name == 'date' or 'date' in df.columns:
            date_col = df.index if df.index.name == 'date' else df['date']
            if pd.api.types.is_datetime64_any_dtype(date_col):
                # 检查日期是否有重复
                if date_col.duplicated().any():
                    errors.append("存在重复日期")
                
                # 检查日期是否按升序排列
                if not date_col.is_monotonic_increasing:
                    errors.append("日期未按升序排列")
                
                # 检查日期间隔是否异常
                if len(date_col) > 1:
                    if isinstance(date_col, pd.DatetimeIndex):
                        date_diff = date_col[1:] - date_col[:-1]
                    else:
                        date_diff = pd.Series(date_col).diff()[1:]
                    
                    max_gap = date_diff.max()
                    if isinstance(max_gap, timedelta) and max_gap > timedelta(days=30):
                        warnings.append(f"存在异常日期间隔: {max_gap.days} 天")
        
        # 检查数据完整性
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            for col in missing_data.index:
                if missing_data[col] > 0:
                    warnings.append(f"列 '{col}' 有 {missing_data[col]} 条缺失值")
        
        # 汇总结果
        valid = len(errors) == 0
        result = {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'data_quality_score': self._calculate_quality_score(df, errors, warnings),
        }
        
        if not valid:
            logger.warning(f"数据验证失败: {', '.join(errors)}")
        elif warnings:
            logger.info(f"数据验证通过，有警告: {', '.join(warnings)}")
        else:
            logger.info("数据验证通过")
        
        return result
    
    def validate_financial_data(self, df):
        """
        验证财务数据的有效性
        
        参数:
            df (pandas.DataFrame): 财务数据
            
        返回:
            dict: 验证结果
        """
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return {'valid': False, 'errors': ['数据为空或格式不正确']}
        
        errors = []
        warnings = []
        
        # 检查财务指标的合理性
        if 'pe_ratio' in df.columns:
            # 检查异常PE值
            pe_values = df['pe_ratio'].dropna()
            if len(pe_values) > 0:
                if (pe_values < 0).any():
                    warnings.append("存在负PE值")
                if (pe_values > 1000).any():
                    warnings.append("存在异常高PE值 > 1000")
        
        if 'pb_ratio' in df.columns:
            # 检查异常PB值
            pb_values = df['pb_ratio'].dropna()
            if len(pb_values) > 0:
                if (pb_values < 0).any():
                    warnings.append("存在负PB值")
                if (pb_values > 100).any():
                    warnings.append("存在异常高PB值 > 100")
        
        if 'debt_to_assets' in df.columns:
            # 检查资产负债率
            dta_values = df['debt_to_assets'].dropna()
            if len(dta_values) > 0:
                if (dta_values < 0).any():
                    errors.append("存在负资产负债率")
                if (dta_values > 1).any():
                    warnings.append("资产负债率超过100%")
        
        # 检查财务数据的时间有效性
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            for date_col in date_columns:
                try:
                    dates = pd.to_datetime(df[date_col])
                    if dates.max() > datetime.now():
                        errors.append(f"列 '{date_col}' 存在未来日期")
                except:
                    warnings.append(f"列 '{date_col}' 日期格式无法解析")
        
        # 检查收入和利润的合理关系
        if 'revenue' in df.columns and 'net_profit' in df.columns:
            revenue = df['revenue'].dropna()
            net_profit = df['net_profit'].dropna()
            
            if len(revenue) > 0 and len(net_profit) > 0:
                if (net_profit > revenue).any():
                    errors.append("存在净利润大于营收的记录")
                
                profit_margin = net_profit / revenue.replace(0, np.nan)
                if (profit_margin > 0.5).any():
                    warnings.append("存在利润率异常高的记录 > 50%")
                
                if (profit_margin < -0.5).any():
                    warnings.append("存在利润率异常低的记录 < -50%")
        
        # 汇总结果
        valid = len(errors) == 0
        result = {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'data_quality_score': self._calculate_quality_score(df, errors, warnings),
        }
        
        if not valid:
            logger.warning(f"财务数据验证失败: {', '.join(errors)}")
        elif warnings:
            logger.info(f"财务数据验证通过，有警告: {', '.join(warnings)}")
        else:
            logger.info("财务数据验证通过")
        
        return result
    
    def _calculate_quality_score(self, df, errors, warnings):
        """
        计算数据质量评分
        
        参数:
            df (pandas.DataFrame): 数据
            errors (list): 错误列表
            warnings (list): 警告列表
            
        返回:
            float: 质量评分（0-100）
        """
        # 基础分数
        base_score = 100
        
        # 错误扣分（每个错误扣除15分）
        error_penalty = len(errors) * 15
        
        # 警告扣分（每个警告扣除5分）
        warning_penalty = len(warnings) * 5
        
        # 数据完整性分数
        if len(df) > 0:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            missing_penalty = missing_ratio * 30  # 缺失比例乘以30
        else:
            missing_penalty = 30
        
        # 汇总分数
        score = base_score - error_penalty - warning_penalty - missing_penalty
        score = max(0, min(100, score))  # 限制在0-100范围内
        
        return score 