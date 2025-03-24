#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票筛选器应用类 - 提供Streamlit界面的主要功能实现
"""

import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta

# 导入自定义模块
from data.stock_data import StockDataFetcher
from strategies.fund_strategy import FundStrategy
from strategies.market_strategy import MarketStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.volume_price_strategy import VolumePriceStrategy
from models.prediction import StockPredictor


class StockScreenerApp:
    """股票筛选器应用类"""
    
    def __init__(self):
        """初始化应用"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化股票筛选器应用")
        
        # 初始化数据获取器
        self.data_fetcher = StockDataFetcher()
        
        # 初始化策略
        self.strategies = {
            "基本面策略": FundStrategy(),
            "市场表现策略": MarketStrategy(),
            "趋势策略": TrendStrategy(),
            "量价策略": VolumePriceStrategy()
        }
        
        # 初始化预测模型
        self.predictor = StockPredictor()
        
        # 缓存数据
        self.cache = {}
    
    def run(self):
        """运行应用"""
        st.set_page_config(page_title="至简交易选股系统", page_icon="📈", layout="wide")
        
        # 设置侧边栏
        self._setup_sidebar()
        
        # 设置主页面
        st.title("至简交易选股系统")
        st.subheader("基于简放交易理念 + AI + 量化因子的股票筛选系统")
        
        # 创建标签页
        tabs = st.tabs(["选股", "个股分析", "市场概览", "AI预测"])
        
        # 选股标签页
        with tabs[0]:
            self._stock_screening_tab()
        
        # 个股分析标签页
        with tabs[1]:
            self._stock_analysis_tab()
        
        # 市场概览标签页
        with tabs[2]:
            self._market_overview_tab()
        
        # AI预测标签页
        with tabs[3]:
            self._ai_prediction_tab()
    
    def _setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.title("系统设置")
        
        # 数据源设置
        st.sidebar.subheader("数据源设置")
        data_source = st.sidebar.selectbox("选择数据源", ["Tushare", "AkShare"])
        
        # 刷新数据按钮
        if st.sidebar.button("刷新数据"):
            st.sidebar.success("数据已刷新！")
            self.cache = {}
        
        # 显示系统信息
        st.sidebar.subheader("系统信息")
        st.sidebar.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 关于信息
        st.sidebar.subheader("关于")
        st.sidebar.markdown("**至简交易选股系统** 是一个基于简放交易理念，结合AI和量化因子的股票筛选工具。")
    
    def _stock_screening_tab(self):
        """选股标签页"""
        st.header("股票筛选")
        
        # 获取股票列表
        stock_list = self._get_stock_list()
        if stock_list.empty:
            st.error("无法获取股票列表，请检查数据源配置")
            return
        
        # 筛选条件
        col1, col2 = st.columns(2)
        
        with col1:
            # 行业筛选
            industries = ["全部"] + sorted(stock_list["industry"].dropna().unique().tolist())
            selected_industry = st.selectbox("选择行业", industries)
            
            # 策略选择
            selected_strategies = st.multiselect(
                "选择策略",
                list(self.strategies.keys()),
                default=["趋势策略", "量价策略"]
            )
        
        with col2:
            # 市值范围
            market_cap_range = st.slider(
                "市值范围（亿元）",
                min_value=0, max_value=5000,
                value=(50, 1000)
            )
            
            # 最小交易量
            min_volume = st.number_input("最小日均交易量（万手）", min_value=0, value=100)
        
        # 执行筛选按钮
        if st.button("开始筛选"):
            with st.spinner("正在筛选股票..."):
                # 根据行业筛选
                if selected_industry != "全部":
                    filtered_stocks = stock_list[stock_list["industry"] == selected_industry]
                else:
                    filtered_stocks = stock_list
                
                # 应用选择的策略
                results = []
                for strategy_name in selected_strategies:
                    strategy = self.strategies[strategy_name]
                    strategy_result = strategy.screen(filtered_stocks)
                    results.append(strategy_result)
                
                # 合并结果
                if results:
                    final_result = pd.concat(results).drop_duplicates()
                    st.success(f"筛选完成，共找到 {len(final_result)} 只符合条件的股票")
                    st.dataframe(final_result)
                else:
                    st.warning("请至少选择一种策略")
    
    def _stock_analysis_tab(self):
        """个股分析标签页"""
        st.header("个股分析")
        
        # 股票选择
        stock_list = self._get_stock_list()
        if stock_list.empty:
            st.error("无法获取股票列表，请检查数据源配置")
            return
        
        # 股票搜索框
        stock_search = st.text_input("输入股票代码或名称")
        
        if stock_search:
            # 搜索股票
            matched_stocks = stock_list[
                stock_list["ts_code"].str.contains(stock_search) | 
                stock_list["name"].str.contains(stock_search)
            ]
            
            if not matched_stocks.empty:
                selected_stock = st.selectbox(
                    "选择股票",
                    matched_stocks.apply(lambda x: f"{x['name']}({x['ts_code']})", axis=1).tolist()
                )
                
                # 提取股票代码
                stock_code = selected_stock.split("(")[1].split(")")[0]
                
                # 获取股票数据
                stock_data = self.data_fetcher.get_daily_data(stock_code)
                
                if not stock_data.empty:
                    # 显示股票信息
                    st.subheader(f"{selected_stock} 基本信息")
                    
                    # 显示K线图
                    st.subheader("K线图")
                    fig = px.line(
                        stock_data, x="trade_date", y="close",
                        title=f"{selected_stock} 收盘价走势"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示交易量
                    st.subheader("交易量")
                    fig = px.bar(
                        stock_data, x="trade_date", y="vol",
                        title=f"{selected_stock} 交易量"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示技术指标
                    st.subheader("技术指标")
                    # 这里可以添加MA、MACD等技术指标的计算和展示
                    
                    # 显示原始数据
                    st.subheader("原始数据")
                    st.dataframe(stock_data)
                else:
                    st.error("无法获取该股票的数据")
            else:
                st.warning("未找到匹配的股票")
    
    def _market_overview_tab(self):
        """市场概览标签页"""
        st.header("市场概览")
        
        # 获取指数数据
        indices = {
            "上证指数": "000001.SH",
            "深证成指": "399001.SZ",
            "创业板指": "399006.SZ",
            "沪深300": "000300.SH"
        }
        
        # 选择指数
        selected_index = st.selectbox("选择指数", list(indices.keys()))
        index_code = indices[selected_index]
        
        # 获取指数数据
        index_data = self.data_fetcher.get_daily_data(index_code)
        
        if not index_data.empty:
            # 显示指数走势
            st.subheader(f"{selected_index}走势")
            fig = px.line(
                index_data, x="trade_date", y="close",
                title=f"{selected_index}收盘价走势"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示行业热力图
            st.subheader("行业热力图")
            st.info("此功能正在开发中...")
            
            # 显示市场宽度指标
            st.subheader("市场宽度指标")
            st.info("此功能正在开发中...")
        else:
            st.error("无法获取指数数据")
    
    def _ai_prediction_tab(self):
        """AI预测标签页"""
        st.header("AI预测")
        
        # 股票选择
        stock_list = self._get_stock_list()
        if stock_list.empty:
            st.error("无法获取股票列表，请检查数据源配置")
            return
        
        # 股票搜索框
        stock_search = st.text_input("输入股票代码或名称", key="ai_stock_search")
        
        if stock_search:
            # 搜索股票
            matched_stocks = stock_list[
                stock_list["ts_code"].str.contains(stock_search) | 
                stock_list["name"].str.contains(stock_search)
            ]
            
            if not matched_stocks.empty:
                selected_stock = st.selectbox(
                    "选择股票",
                    matched_stocks.apply(lambda x: f"{x['name']}({x['ts_code']})", axis=1).tolist(),
                    key="ai_stock_select"
                )
                
                # 提取股票代码
                stock_code = selected_stock.split("(")[1].split(")")[0]
                
                # 预测天数
                days = st.slider("预测天数", min_value=5, max_value=30, value=10)
                
                # 执行预测
                if st.button("开始预测"):
                    with st.spinner("正在进行AI预测..."):
                        # 获取历史数据
                        stock_data = self.data_fetcher.get_daily_data(stock_code)
                        
                        if not stock_data.empty:
                            # 执行预测
                            prediction = self.predictor.predict(stock_data, days)
                            
                            # 显示预测结果
                            st.subheader("预测结果")
                            
                            # 合并历史数据和预测数据
                            last_date = stock_data["trade_date"].iloc[-1]
                            date_range = pd.date_range(start=last_date, periods=days+1)[1:]
                            pred_df = pd.DataFrame({
                                "trade_date": date_range,
                                "close": prediction
                            })
                            
                            # 绘制图表
                            fig = px.line(
                                pd.concat([
                                    stock_data[-30:][['trade_date', 'close']],
                                    pred_df
                                ]),
                                x="trade_date", y="close",
                                title=f"{selected_stock} 价格预测"
                            )
                            # 添加分隔线
                            fig.add_vline(x=last_date, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 显示预测数据表格
                            st.subheader("预测数据")
                            st.dataframe(pred_df)
                            
                            # 显示预测分析
                            st.subheader("预测分析")
                            price_change = (prediction[-1] - stock_data["close"].iloc[-1]) / stock_data["close"].iloc[-1] * 100
                            change_color = "green" if price_change > 0 else "red"
                            st.markdown(f"<h3 style='color:{change_color}'>预测{days}天后价格变化: {price_change:.2f}%</h3>", unsafe_allow_html=True)
                            
                            # 预测结论
                            if price_change > 5:
                                st.success("🔥 强烈看涨信号")
                            elif price_change > 0:
                                st.info("📈 看涨信号")
                            elif price_change > -5:
                                st.warning("📉 看跌信号")
                            else:
                                st.error("❄️ 强烈看跌信号")
                        else:
                            st.error("无法获取该股票的数据")
            else:
                st.warning("未找到匹配的股票")
    
    def _get_stock_list(self):
        """获取股票列表"""
        if "stock_list" in self.cache:
            return self.cache["stock_list"]
        
        # 获取股票列表
        stock_list = self.data_fetcher.get_stock_list()
        
        # 缓存结果
        if not stock_list.empty:
            self.cache["stock_list"] = stock_list
        
        return stock_list