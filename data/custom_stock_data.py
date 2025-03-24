#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义股票数据和交易建议模块 - 用于保存手动设置的交易建议数据
"""

import pandas as pd

# 自定义股票交易建议表
CUSTOM_TRADING_ADVICE = {
    "688173.SH": {
        "name": "希荻微",
        "trading_advice": "建议观望，等待季度财报发布",
        "entry_price": 43.50,
        "stop_loss": 40.80,
        "target_price": 48.20
    },
    "688037.SH": {
        "name": "芯源微",
        "trading_advice": "可考虑分批买入",
        "entry_price": 58.20,
        "stop_loss": 54.50,
        "target_price": 65.00
    },
    "300046.SZ": {
        "name": "台基股份",
        "trading_advice": "短期可介入，注意控制仓位",
        "entry_price": 12.80,
        "stop_loss": 11.90,
        "target_price": 14.50
    },
    "688361.SH": {
        "name": "中科飞测",
        "trading_advice": "技术面走强，可适量买入",
        "entry_price": 76.40,
        "stop_loss": 72.80,
        "target_price": 83.20
    }
}

def get_custom_advice(stock_code):
    """
    获取指定股票的自定义交易建议
    
    Args:
        stock_code: 股票代码
        
    Returns:
        dict: 包含交易建议的字典，如果没有找到则返回None
    """
    return CUSTOM_TRADING_ADVICE.get(stock_code)

def get_all_custom_advice():
    """
    获取所有自定义交易建议
    
    Returns:
        pandas.DataFrame: 包含所有交易建议的DataFrame
    """
    advice_list = []
    for stock_code, advice in CUSTOM_TRADING_ADVICE.items():
        row = {
            "ts_code": stock_code,
            "name": advice["name"],
            "trading_advice": advice["trading_advice"],
            "entry_price": advice["entry_price"],
            "stop_loss": advice["stop_loss"],
            "target_price": advice["target_price"]
        }
        advice_list.append(row)
    
    return pd.DataFrame(advice_list) 