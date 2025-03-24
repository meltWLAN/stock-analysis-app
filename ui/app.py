#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
应用入口模块 - 负责启动Streamlit应用
"""

import logging
from ui.stock_screener import StockScreenerApp

def run_app():
    """运行Streamlit应用"""
    app = StockScreenerApp()
    app.run()