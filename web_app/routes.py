from flask import Flask, render_template, jsonify, request
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

app = Flask(__name__)

@app.route('/')
def index():
    """首页 - 市场概览"""
    return render_template('index.html')

@app.route('/stock')
def stock_detail():
    """股票详情页"""
    code = request.args.get('code', '000001.SZ')  # 默认平安银行
    is_favorite = False  # 实际应从用户数据中获取
    
    # 模拟股票信息
    stock_info = {
        'code': code,
        'name': '模拟股票',
        'price': '28.65',
        'price_change': '+0.87',
        'price_change_percent': '3.13',
        'open': '27.95',
        'prev_close': '27.78',
        'high': '28.76',
        'low': '27.82',
        'volume': '1258.67',  # 万手
        'amount': '35.45',  # 亿元
        'pe': '12.45',
        'pb': '1.35',
        'market': '主板',
        'industry': '金融'
    }
    
    # 为演示，根据股票代码生成不同名称
    if code.startswith('000'):
        stock_info['name'] = '平安银行'
        stock_info['industry'] = '银行'
    elif code.startswith('600'):
        stock_info['name'] = '贵州茅台'
        stock_info['industry'] = '白酒'
    elif code.startswith('002'):
        stock_info['name'] = '东方雨虹'
        stock_info['industry'] = '建材'
    elif code.startswith('300'):
        stock_info['name'] = '迈瑞医疗'
        stock_info['industry'] = '医疗器械'
    elif code.startswith('688'):
        stock_info['name'] = '中芯国际'
        stock_info['industry'] = '半导体'
    
    return render_template('stock_detail.html', stock_info=stock_info, is_favorite=is_favorite)

@app.route('/stock_screener')
def stock_screener():
    """股票筛选器页面"""
    return render_template('stock_screener.html')

@app.route('/portfolio')
def portfolio():
    """持仓组合页面"""
    return render_template('portfolio.html')

@app.route('/market_monitor')
def market_monitor():
    """市场监控页面"""
    return render_template('market_monitor.html')

# API路由
@app.route('/api/stock/search')
def api_stock_search():
    """股票搜索API"""
    keyword = request.args.get('keyword', '')
    
    if not keyword or len(keyword) < 2:
        return jsonify({'success': False, 'message': '关键词太短', 'data': []})
    
    # 模拟搜索结果
    mock_stocks = [
        {'code': '000001.SZ', 'name': '平安银行'},
        {'code': '600000.SH', 'name': '浦发银行'},
        {'code': '600030.SH', 'name': '中信证券'},
        {'code': '601318.SH', 'name': '中国平安'},
        {'code': '000002.SZ', 'name': '万科A'},
        {'code': '600519.SH', 'name': '贵州茅台'},
        {'code': '601857.SH', 'name': '中国石油'},
        {'code': '600036.SH', 'name': '招商银行'},
        {'code': '300750.SZ', 'name': '宁德时代'},
        {'code': '002594.SZ', 'name': '比亚迪'}
    ]
    
    # 根据关键词过滤
    results = [s for s in mock_stocks if keyword.lower() in s['code'].lower() or keyword.lower() in s['name'].lower()]
    
    return jsonify({
        'success': True,
        'data': results[:5]  # 限制返回前5个
    })

@app.route('/api/market/overview')
def api_market_overview():
    """市场概览数据API"""
    # 模拟指数数据
    indices = {
        '000001.SH': {  # 上证指数
            'name': '上证指数',
            'current': '3,246.27',
            'change': '+24.97',
            'change_percent': '+0.78'
        },
        '399001.SZ': {  # 深证成指
            'name': '深证成指',
            'current': '10,638.75',
            'change': '+129.84',
            'change_percent': '+1.23'
        },
        '399006.SZ': {  # 创业板指
            'name': '创业板指',
            'current': '2,168.59',
            'change': '+33.63',
            'change_percent': '+1.57'
        },
        '000688.SH': {  # 科创50
            'name': '科创50',
            'current': '1,056.43',
            'change': '-3.02',
            'change_percent': '-0.28'
        }
    }
    
    # 模拟热力图数据（前端通过Plotly/ApexCharts等图表库渲染）
    chart_data = {"data": [], "layout": {}}
    
    return jsonify({
        'success': True,
        'data': {
            'indices': indices,
        },
        'chart': json.dumps(chart_data)
    })

@app.route('/api/stock/price')
def api_stock_price():
    """获取股票价格数据"""
    code = request.args.get('code', '000001.SZ')
    period = request.args.get('period', '1d')  # 1d, 1w, 1m, 3m, 1y
    
    # 生成模拟数据
    end_date = datetime.now()
    
    if period == '1d':
        days = 1
        interval = 'min'
    elif period == '1w':
        days = 7
        interval = 'day'
    elif period == '1m':
        days = 30
        interval = 'day'
    elif period == '3m':
        days = 90
        interval = 'day'
    else:  # 1y
        days = 365
        interval = 'day'
    
    start_date = end_date - timedelta(days=days)
    
    # 生成日期范围
    if interval == 'day':
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    else:  # min
        dates = pd.date_range(start=end_date.replace(hour=9, minute=30), 
                              end=end_date.replace(hour=15, minute=0), 
                              freq='1min')
    
    # 生成价格数据
    base_price = 28.0
    price_data = []
    
    for date in dates:
        timestamp = int(date.timestamp() * 1000)  # JavaScript时间戳
        
        # 添加一些随机波动
        random_change = np.random.normal(0, 0.5)
        price = base_price + random_change
        base_price = price  # 更新基准价格
        
        if price < 20:
            price = 20 + np.random.random()
        
        open_price = price * (1 + np.random.normal(0, 0.01))
        high_price = max(price, open_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(price, open_price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price
        volume = int(np.random.normal(10000, 3000))
        
        price_data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    return jsonify({
        'success': True,
        'data': price_data
    })

@app.route('/api/stock/forecast')
def api_stock_forecast():
    """获取股票预测数据"""
    code = request.args.get('code', '000001.SZ')
    
    # 获取当前价格作为基准
    current_price = 28.65
    
    # 生成未来30天的日期
    end_date = datetime.now() + timedelta(days=30)
    start_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    
    # 生成预测数据
    forecast_data = []
    upper_bound_data = []
    lower_bound_data = []
    
    base_price = current_price
    trend = np.random.choice([-1, 1], p=[0.3, 0.7])  # 70%概率上涨
    
    for date in dates:
        timestamp = int(date.timestamp() * 1000)  # JavaScript时间戳
        
        # 添加一些随机波动和小趋势
        day_factor = (date - start_date).days / 30.0  # 0到1之间的因子
        trend_factor = trend * day_factor * 0.15  # 最多15%的趋势变化
        
        random_change = np.random.normal(0, 0.3)
        forecast = base_price * (1 + trend_factor + random_change * 0.01)
        
        # 为每个日期生成上下区间
        volatility = 0.03 * (1 + day_factor)  # 波动率随时间增加
        upper_bound = forecast * (1 + volatility)
        lower_bound = forecast * (1 - volatility * 0.8)  # 下界波动稍小
        
        # 添加到结果中
        forecast_data.append({
            'timestamp': timestamp,
            'price': round(forecast, 2)
        })
        
        upper_bound_data.append({
            'timestamp': timestamp,
            'price': round(upper_bound, 2)
        })
        
        lower_bound_data.append({
            'timestamp': timestamp,
            'price': round(lower_bound, 2)
        })
    
    return jsonify({
        'success': True,
        'data': {
            'forecast': forecast_data,
            'upper_bound': upper_bound_data,
            'lower_bound': lower_bound_data,
            'confidence': 75,  # 置信度百分比
            'recommendation': '买入' if trend > 0 else '卖出'
        }
    })

@app.route('/api/stock/similar')
def api_stock_similar():
    """获取相似股票"""
    code = request.args.get('code', '000001.SZ')
    
    # 模拟相似股票数据
    similar_stocks = [
        {'code': '300033.SZ', 'name': '同花顺', 'price': '42.68', 'change_percent': '+2.54'},
        {'code': '300059.SZ', 'name': '东方财富', 'price': '18.43', 'change_percent': '+1.21'},
        {'code': '600570.SH', 'name': '恒生电子', 'price': '64.25', 'change_percent': '-0.87'},
        {'code': '300226.SZ', 'name': '上海钢联', 'price': '72.16', 'change_percent': '-1.32'}
    ]
    
    return jsonify({
        'success': True,
        'data': similar_stocks
    })


if __name__ == '__main__':
    app.run(debug=True) 