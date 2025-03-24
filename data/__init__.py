#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据包初始化
"""

__version__ = "1.0.0"

# 数据获取与处理模块

try:
    from .custom_stock_data import get_custom_advice, get_all_custom_advice
except ImportError:
    pass

from .stock_data import StockData
from .custom_stock_data import CustomStockData
from .data_fetcher import DataFetcher
from .data_source_manager import DataSourceManager
from .data_pipeline import DataPipeline
from .data_processor import DataProcessor
from .data_validator import FinancialDataValidator