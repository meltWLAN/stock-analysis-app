#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统 - 简化版Streamlit Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import os
import sys

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(
    page_title="股票分析系统",
    page_icon="📈",
    layout="wide"
)

# 应用CSS样式
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 0.5rem;
    }
    .stApp {
        max-width: 100%;
    }
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 模拟股票数据
def generate_sample_data(days=60):
    """生成样本股票数据"""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 生成起始价格
    start_price = 100
    
    # 生成价格数据
    price_changes = np.random.normal(0, 1, len(date_range)) * 2  # 每日变化率
    prices = start_price * (1 + price_changes).cumprod()
    
    # 生成OHLCV数据
    data = {
        'date': date_range,
        'open': prices * (1 + np.random.normal(0, 0.005, len(date_range))),
        'high': prices * (1 + np.random.normal(0.01, 0.01, len(date_range))),
        'low': prices * (1 - np.random.normal(0.01, 0.01, len(date_range))),
        'close': prices,
        'volume': np.random.normal(1000000, 300000, len(date_range))
    }
    
    # 确保high >= open, close, low 且 low <= open, close
    for i in range(len(date_range)):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df

# 计算技术指标
def calculate_indicators(df):
    """计算常用技术指标"""
    indicators = {}
    
    # 计算移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    
    indicators['MA5'] = df['ma5']
    indicators['MA20'] = df['ma20']
    indicators['MA60'] = df['ma60']
    
    # 计算RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 计算布林带
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    indicators['BB中轨'] = df['bb_middle']
    indicators['BB上轨'] = df['bb_upper']
    indicators['BB下轨'] = df['bb_lower']
    
    return df, indicators

# 绘制K线图
def plot_stock_chart(df, indicators=None):
    """绘制股票K线图及指标"""
    if df is None or df.empty:
        st.warning("没有数据可供绘制")
        return
        
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3]
    )
    
    # 添加K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线'
        ),
        row=1, col=1
    )
    
    # 添加成交量
    if 'volume' in df.columns:
        colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='成交量',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    # 添加指标
    if indicators:
        for name, data in indicators.items():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=data,
                    name=name,
                    line=dict(width=1)
                ),
                row=1, col=1
            )
    
    # 更新布局
    fig.update_layout(
        title='股票价格走势',
        xaxis_title='日期',
        yaxis_title='价格',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # 优化移动端显示
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# 趋势分析页面
def trend_analysis_page():
    st.header("趋势分析")
    
    # 股票选择
    stock_options = {
        "000001.SZ": "平安银行",
        "600000.SH": "浦发银行",
        "601318.SH": "中国平安",
        "000858.SZ": "五粮液",
        "000333.SZ": "美的集团"
    }
    
    selected_stock = st.selectbox(
        "选择股票",
        options=list(stock_options.keys()),
        format_func=lambda x: f"{stock_options[x]} ({x})"
    )
    
    # 日期范围选择
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime.now() - timedelta(days=60))
    with col2:
        end_date = st.date_input("结束日期", datetime.now())
    
    # 生成样本数据
    df = generate_sample_data(days=(end_date - start_date).days)
    
    # 计算指标
    df, indicators = calculate_indicators(df)
    
    # 显示分析结果
    score = 7.5  # 模拟得分
    trend = "上升" if df['close'].iloc[-1] > df['close'].iloc[-10] else "下降"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("趋势评分", f"{score}/10", delta=f"{score-5:.1f}")
    with col2:
        st.metric("趋势判断", trend, delta="上涨" if trend == "上升" else "下跌")
    
    # 显示分析详情
    with st.expander("详细分析", expanded=True):
        st.write(f"- 股票代码: {selected_stock}")
        st.write(f"- 股票名称: {stock_options[selected_stock]}")
        st.write(f"- 当前价格: {df['close'].iloc[-1]:.2f}")
        st.write(f"- 5日涨跌幅: {(df['close'].iloc[-1]/df['close'].iloc[-5]-1)*100:.2f}%")
        st.write(f"- 20日涨跌幅: {(df['close'].iloc[-1]/df['close'].iloc[-20]-1)*100:.2f}%")
        
        ma_status = "多头排列" if df['ma5'].iloc[-1] > df['ma20'].iloc[-1] > df['ma60'].iloc[-1] else "空头排列" if df['ma5'].iloc[-1] < df['ma20'].iloc[-1] < df['ma60'].iloc[-1] else "震荡"
        st.write(f"- 均线状态: {ma_status}")
        
        rsi_value = df['rsi'].iloc[-1]
        rsi_status = "超买" if rsi_value > 70 else "超卖" if rsi_value < 30 else "正常"
        st.write(f"- RSI({rsi_value:.1f}): {rsi_status}")
    
    # 绘制K线图
    fig = plot_stock_chart(df, indicators)
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示MACD和RSI
    col1, col2 = st.columns(2)
    with col1:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['macd'],
            name='MACD'
        ))
        macd_fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['macd_signal'],
            name='Signal'
        ))
        macd_fig.update_layout(title='MACD指标', height=300)
        st.plotly_chart(macd_fig, use_container_width=True)
        
    with col2:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['rsi'],
            name='RSI'
        ))
        # 添加超买超卖线
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(title='RSI指标', height=300)
        st.plotly_chart(rsi_fig, use_container_width=True)

# 股票筛选页面
def stock_screener_page():
    st.header("股票筛选")
    
    # 筛选条件
    st.subheader("筛选条件")
    
    col1, col2 = st.columns(2)
    with col1:
        industry = st.selectbox(
            "行业选择",
            ["全部", "银行", "保险", "证券", "房地产", "医药", "科技", "消费", "制造", "能源", "通信"]
        )
    with col2:
        strategy_type = st.selectbox(
            "策略类型",
            ["趋势策略", "量价策略", "综合策略"]
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_price = st.number_input("最低价格", 0.0, 1000.0, 5.0)
    with col2:
        max_price = st.number_input("最高价格", 0.0, 10000.0, 100.0)
    with col3:
        min_score = st.slider("最低评分", 0, 10, 7)
    
    # 执行筛选
    if st.button("开始筛选"):
        # 生成示例结果数据
        np.random.seed(42)
        
        # 生成模拟股票列表
        stock_codes = [
            "000001.SZ", "600000.SH", "601318.SH", "000858.SZ", "000333.SZ",
            "600036.SH", "601166.SH", "600519.SH", "601668.SH", "002415.SZ",
            "600887.SH", "601288.SH", "600030.SH", "000651.SZ", "600276.SH"
        ]
        stock_names = [
            "平安银行", "浦发银行", "中国平安", "五粮液", "美的集团",
            "招商银行", "兴业银行", "贵州茅台", "中国建筑", "海康威视",
            "伊利股份", "农业银行", "中信证券", "格力电器", "恒瑞医药"
        ]
        
        prices = np.random.uniform(min_price, max_price, len(stock_codes))
        scores = np.random.uniform(min_score, 10, len(stock_codes))
        trends = np.random.choice(["上升", "下降", "震荡"], len(stock_codes))
        signals = np.random.choice(["买入", "卖出", "观望"], len(stock_codes))
        
        # 创建结果数据
        results = []
        for i in range(len(stock_codes)):
            if scores[i] >= min_score and min_price <= prices[i] <= max_price:
                results.append({
                    'code': stock_codes[i],
                    'name': stock_names[i],
                    'price': prices[i],
                    'score': scores[i],
                    'trend': trends[i],
                    'signal': signals[i]
                })
        
        # 排序结果
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 展示结果
        st.subheader(f"筛选结果 ({len(results)} 只股票)")
        
        if not results:
            st.warning("没有找到符合条件的股票")
            return
        
        # 使用表格展示结果
        result_df = pd.DataFrame(results)
        result_df = result_df[['code', 'name', 'price', 'score', 'trend', 'signal']]
        result_df.columns = ['代码', '名称', '价格', '评分', '趋势', '信号']
        st.dataframe(result_df, use_container_width=True)
        
        # 显示每只股票的详细信息
        st.subheader("股票详情")
        for result in results[:5]:  # 只显示前5只
            with st.expander(f"{result['name']} ({result['code']})"):
                st.write(f"- 价格: {result['price']:.2f}")
                st.write(f"- 评分: {result['score']:.1f}")
                st.write(f"- 趋势: {result['trend']}")
                st.write(f"- 信号: {result['signal']}")
                
                # 生成随机详情
                details = {
                    "均线多头排列": np.random.choice([True, False]),
                    "MACD金叉": np.random.choice([True, False]),
                    "RSI值": np.random.uniform(30, 70),
                    "布林带位置": np.random.choice(["上轨", "中轨", "下轨"]),
                    "成交量变化": f"{np.random.uniform(-20, 50):.1f}%",
                    "主力资金流入": f"{np.random.uniform(-1000, 5000):.2f}万元"
                }
                
                st.write("详细分析:")
                for key, value in details.items():
                    if isinstance(value, bool):
                        st.write(f"  - {key}: {'是' if value else '否'}")
                    elif isinstance(value, float):
                        st.write(f"  - {key}: {value:.2f}")
                    else:
                        st.write(f"  - {key}: {value}")

# 关于页面
def about_page():
    st.header("关于系统")
    
    st.markdown("""
    ## 股票分析系统

    这是一个综合性的股票分析与筛选系统，结合了技术分析和机器学习的方法。

    ### 主要功能

    - **技术趋势分析**：对单只股票进行多维度技术分析
    - **股票智能筛选**：根据多种条件筛选出符合要求的股票
    - **量化评分系统**：对每只股票进行量化评分

    ### 技术特点

    - 移动端优化设计，随时随地可用
    - 简洁直观的用户界面
    - 快速响应的数据处理

    ### 使用提示

    - 趋势分析页面可以深入分析单只股票
    - 股票筛选页面可以快速发现潜力股票
    - 评分越高表示符合策略的程度越高
    """)
    
    st.info("本系统仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。")

# 主应用布局
def main():
    apply_custom_css()
    
    # 设置标题
    st.title("股票分析系统")
    
    # 创建标签页
    tabs = st.tabs(["趋势分析", "股票筛选", "关于"])
    
    # 各标签页内容
    with tabs[0]:
        trend_analysis_page()
    
    with tabs[1]:
        stock_screener_page()
    
    with tabs[2]:
        about_page()
    
    # 页脚
    st.markdown("---")
    st.markdown("股票分析系统 © 2025")

if __name__ == "__main__":
    main() 