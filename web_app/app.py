#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票分析系统 Web应用
提供手机端访问界面
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import plotly
import plotly.express as px
import plotly.graph_objects as go

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入数据处理模块
from data.data_validator import FinancialDataValidator
from data.stock_data import StockData  # 假设有此模块
import utils.plot_utils as plot_utils  # 假设有此模块

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'web_app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = os.urandom(24)  # 用于session加密
app.config['JSON_AS_ASCII'] = False  # 确保中文正确显示

# 初始化数据处理组件
try:
    stock_data = StockData()
    validator = FinancialDataValidator()
    logger.info("数据组件初始化成功")
except Exception as e:
    logger.error(f"数据组件初始化失败: {e}")

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 股票基本信息API
@app.route('/api/stock/info', methods=['GET'])
def get_stock_info():
    stock_code = request.args.get('code', '')
    
    # 验证股票代码
    if not validator.validate_stock_code(stock_code):
        return jsonify({'success': False, 'message': '无效的股票代码'})
    
    try:
        # 获取股票基本信息
        info = stock_data.get_stock_info(stock_code)
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        logger.error(f"获取股票信息出错: {e}")
        return jsonify({'success': False, 'message': f'获取股票信息失败: {str(e)}'})

# 股票价格数据API
@app.route('/api/stock/price', methods=['GET'])
def get_stock_price():
    stock_code = request.args.get('code', '')
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # 验证股票代码和日期
    if not validator.validate_stock_code(stock_code):
        return jsonify({'success': False, 'message': '无效的股票代码'})
    
    if not validator.validate_date_range(start_date, end_date):
        return jsonify({'success': False, 'message': '无效的日期范围'})
    
    try:
        # 获取股票价格数据
        price_data = stock_data.get_price_data(stock_code, start_date, end_date)
        
        # 验证数据
        validation_result = validator.validate_price_data(price_data)
        
        # 转换为JSON可序列化格式
        price_json = price_data.reset_index().to_dict(orient='records')
        
        # 生成K线图
        candlestick = go.Figure(data=[go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='K线'
        )])
        candlestick.update_layout(
            title=f'{stock_code} 价格走势',
            xaxis_title='日期',
            yaxis_title='价格',
            template='plotly_white'
        )
        
        # 转换图表为JSON
        candlestick_json = json.dumps(candlestick, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True, 
            'data': price_json,
            'chart': candlestick_json,
            'validation': validation_result
        })
    except Exception as e:
        logger.error(f"获取股票价格数据出错: {e}")
        return jsonify({'success': False, 'message': f'获取股票价格数据失败: {str(e)}'})

# 财务数据API
@app.route('/api/stock/financial', methods=['GET'])
def get_financial_data():
    stock_code = request.args.get('code', '')
    
    # 验证股票代码
    if not validator.validate_stock_code(stock_code):
        return jsonify({'success': False, 'message': '无效的股票代码'})
    
    try:
        # 获取财务数据
        financial_data = stock_data.get_financial_data(stock_code)
        
        # 验证数据
        validation_result = validator.validate_financial_data(financial_data)
        
        # 转换为JSON可序列化格式
        financial_json = financial_data.to_dict(orient='records')
        
        return jsonify({
            'success': True, 
            'data': financial_json,
            'validation': validation_result
        })
    except Exception as e:
        logger.error(f"获取财务数据出错: {e}")
        return jsonify({'success': False, 'message': f'获取财务数据失败: {str(e)}'})

# 市场概览API
@app.route('/api/market/overview', methods=['GET'])
def get_market_overview():
    try:
        # 获取市场概览数据
        market_data = stock_data.get_market_overview()
        
        # 生成市场热力图
        market_map = plot_utils.create_market_heatmap(market_data)
        market_map_json = json.dumps(market_map, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True, 
            'data': market_data,
            'chart': market_map_json
        })
    except Exception as e:
        logger.error(f"获取市场概览数据出错: {e}")
        return jsonify({'success': False, 'message': f'获取市场概览数据失败: {str(e)}'})

# 使用API数据进行回测测试
@app.route('/api/backtest/simple', methods=['POST'])
def run_simple_backtest():
    try:
        data = request.json
        stock_code = data.get('stock_code', '')
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        strategy = data.get('strategy', 'ma_cross')
        params = data.get('params', {})
        
        # 验证参数
        if not validator.validate_stock_code(stock_code):
            return jsonify({'success': False, 'message': '无效的股票代码'})
        
        if not validator.validate_date_range(start_date, end_date):
            return jsonify({'success': False, 'message': '无效的日期范围'})
        
        # 运行回测
        # 假设有一个简化的回测模块
        from strategies.simple_backtest import run_backtest
        
        result = run_backtest(stock_code, start_date, end_date, strategy, params)
        
        # 生成回测结果图表
        performance_chart = plot_utils.create_backtest_chart(result)
        chart_json = json.dumps(performance_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'data': result,
            'chart': chart_json
        })
    except Exception as e:
        logger.error(f"回测出错: {e}")
        return jsonify({'success': False, 'message': f'回测失败: {str(e)}'})

# 股票搜索API
@app.route('/api/stock/search', methods=['GET'])
def search_stocks():
    keyword = request.args.get('keyword', '')
    
    if not keyword or len(keyword) < 2:
        return jsonify({'success': False, 'message': '请输入至少2个字符进行搜索'})
    
    try:
        # 搜索股票
        results = stock_data.search_stocks(keyword)
        return jsonify({'success': True, 'data': results})
    except Exception as e:
        logger.error(f"搜索股票出错: {e}")
        return jsonify({'success': False, 'message': f'搜索失败: {str(e)}'})

# 启动应用
if __name__ == '__main__':
    # 创建日志目录
    os.makedirs(os.path.join(parent_dir, 'logs'), exist_ok=True)
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True) 