#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
资金筛选策略模块 - 实现基于资金流向的选股策略
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FundStrategy:
    """资金筛选策略类"""
    
    def __init__(self):
        """初始化资金策略"""
        self.logger = logging.getLogger(__name__)
    
    def check_main_fund_inflow(self, money_flow_df, days=5, threshold=1000):
        """
        检查主力资金净流入
        
        Args:
            money_flow_df: 资金流向数据DataFrame
            days: 检查的天数，默认为5天
            threshold: 资金流入阈值，单位为万元，默认为1000万
            
        Returns:
            bool: 是否满足主力资金净流入条件
        """
        # 确保数据量足够
        if len(money_flow_df) < days:
            self.logger.warning(f"资金流向数据不足{days}天")
            return False
        
        # 获取最近N天数据
        recent_df = money_flow_df.iloc[-days:].copy()
        
        # 根据数据源的不同，字段名可能不同
        if 'net_mf_amount' in recent_df.columns:  # Tushare格式
            # 计算主力资金净流入总和（单位：万元）
            total_inflow = recent_df['net_mf_amount'].sum() / 10000
        elif '主力净流入' in recent_df.columns:  # AkShare格式
            # 直接计算主力净流入总和
            total_inflow = recent_df['主力净流入'].sum()
        else:
            self.logger.error("无法识别资金流向数据格式")
            return False
        
        # 判断是否满足阈值条件
        return total_inflow >= threshold
    
    def check_institution_buy(self, dragon_tiger_df, days=3):
        """
        检查龙虎榜机构买入情况
        
        Args:
            dragon_tiger_df: 龙虎榜数据DataFrame
            days: 检查的天数，默认为3天
            
        Returns:
            bool: 是否有机构买入
        """
        # 确保数据量足够
        if dragon_tiger_df is None or dragon_tiger_df.empty:
            self.logger.warning("龙虎榜数据为空")
            return False
        
        # 获取最近N天数据
        recent_df = dragon_tiger_df
        if len(dragon_tiger_df) > days:
            recent_df = dragon_tiger_df.iloc[-days:].copy()
        
        # 检查是否有机构席位买入
        has_institution = False
        
        # 根据数据源的不同，字段名可能不同
        if 'exalter' in recent_df.columns:  # Tushare格式
            # 检查买入席位中是否包含机构或知名游资
            for _, row in recent_df.iterrows():
                if '机构' in str(row['exalter']) or '证券' in str(row['exalter']) or '基金' in str(row['exalter']):
                    has_institution = True
                    break
        elif '买方机构' in recent_df.columns:  # AkShare格式
            # 直接检查买方机构列
            has_institution = not recent_df['买方机构'].isna().all()
        else:
            self.logger.error("无法识别龙虎榜数据格式")
            return False
        
        return has_institution
    
    def check_north_money_inflow(self, north_flow_df, days=5):
        """
        检查北向资金持续流入
        
        Args:
            north_flow_df: 北向资金流向数据DataFrame
            days: 检查的天数，默认为5天
            
        Returns:
            bool: 是否满足北向资金持续流入条件
        """
        # 确保数据量足够
        if north_flow_df is None or north_flow_df.empty:
            self.logger.warning("北向资金数据为空")
            return False
        
        # 获取最近N天数据
        recent_df = north_flow_df
        if len(north_flow_df) > days:
            recent_df = north_flow_df.iloc[-days:].copy()
        
        # 根据数据源的不同，字段名可能不同
        if 'north_money' in recent_df.columns:  # 自定义合并格式
            # 计算流入天数占比
            inflow_days = (recent_df['north_money'] > 0).sum()
            inflow_ratio = inflow_days / len(recent_df)
            
            # 计算累计净流入
            total_inflow = recent_df['north_money'].sum()
            
            # 判断条件：流入天数超过60%且累计为净流入
            return inflow_ratio >= 0.6 and total_inflow > 0
            
        elif 'net_northbound' in recent_df.columns:
            # 使用北向资金净流入作为筛选条件
            if north_money_threshold > 0:
                recent_df = recent_df[recent_df['net_northbound'] > north_money_threshold]
            
            # 计算流入天数占比
            inflow_days = (recent_df['net_northbound'] > 0).sum()
            inflow_ratio = inflow_days / len(recent_df)
            
            # 计算累计净流入
            total_inflow = recent_df['net_northbound'].sum()
            
            # 判断条件：流入天数超过60%且累计为净流入
            return inflow_ratio >= 0.6 and total_inflow > 0
    
    def screen(self, stock_list):
        """
        根据资金策略筛选股票
        
        Args:
            stock_list: 待筛选的股票列表DataFrame
            
        Returns:
            DataFrame: 筛选后的股票列表
        """
        if stock_list is None or stock_list.empty:
            self.logger.warning("输入的股票列表为空，无法进行筛选")
            return pd.DataFrame()
        
        self.logger.info(f"开始使用资金策略筛选，共 {len(stock_list)} 只股票")
        
        # 结果列表
        results = []
        
        # 遍历股票进行筛选
        for _, stock in stock_list.iterrows():
            try:
                # 获取股票代码
                stock_code = stock['ts_code'] if 'ts_code' in stock else stock['code']
                stock_name = stock['name'] if 'name' in stock else '未知'
                
                # 获取资金流向数据
                money_flow_df = self._get_money_flow_data(stock_code)
                dragon_tiger_df = self._get_dragon_tiger_data(stock_code)
                north_flow_df = self._get_north_flow_data(stock_code)
                
                if money_flow_df is None or money_flow_df.empty:
                    continue
                
                # 执行策略
                has_main_inflow = self.check_main_fund_inflow(money_flow_df)
                institution_buying = dragon_tiger_df is not None and self.check_institution_buy(dragon_tiger_df)
                north_inflow = north_flow_df is not None and self.check_north_money_inflow(north_flow_df)
                
                # 如果满足主力资金流入条件，或者同时满足机构买入和北向资金流入条件
                if has_main_inflow or (institution_buying and north_inflow):
                    results.append(stock)
            except Exception as e:
                self.logger.error(f"筛选股票 {stock_code} 时出错: {str(e)}")
                continue
        
        # 转换为DataFrame
        result_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        self.logger.info(f"资金策略筛选完成，找到 {len(result_df)} 只符合条件的股票")
        return result_df
    
    def _get_money_flow_data(self, stock_code):
        """获取资金流向数据（实际实现中应该从数据源获取）"""
        self.logger.warning(f"使用模拟资金流向数据，实际使用时请实现真实数据获取: {stock_code}")
        
        # 创建模拟数据
        dates = pd.date_range(end=pd.Timestamp.now().date(), periods=30)
        df = pd.DataFrame({
            'date': dates,
            'main_net_inflow': np.random.normal(500, 1500, 30),  # 主力净流入（万元）
        })
        df.set_index('date', inplace=True)
        
        return df
    
    def _get_dragon_tiger_data(self, stock_code):
        """获取龙虎榜数据（实际实现中应该从数据源获取）"""
        self.logger.warning(f"使用模拟龙虎榜数据，实际使用时请实现真实数据获取: {stock_code}")
        
        # 创建模拟数据
        dates = pd.date_range(end=pd.Timestamp.now().date(), periods=10)
        df = pd.DataFrame({
            'date': dates,
            'buy_amount': np.random.normal(5000, 2000, 10),  # 买入金额（万元）
            'sell_amount': np.random.normal(4000, 2000, 10),  # 卖出金额（万元）
            'institution_buy': np.random.choice([True, False], 10, p=[0.3, 0.7])  # 机构是否买入
        })
        df.set_index('date', inplace=True)
        
        return df
    
    def _get_north_flow_data(self, stock_code):
        """获取北向资金数据（实际实现中应该从数据源获取）"""
        self.logger.warning(f"使用模拟北向资金数据，实际使用时请实现真实数据获取: {stock_code}")
        
        # 创建模拟数据
        dates = pd.date_range(end=pd.Timestamp.now().date(), periods=30)
        df = pd.DataFrame({
            'date': dates,
            'north_net_inflow': np.random.normal(300, 1000, 30),  # 北向资金净流入（万元）
        })
        df.set_index('date', inplace=True)
        
        return df