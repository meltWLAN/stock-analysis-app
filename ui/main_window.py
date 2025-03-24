#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QSplitter,
                               QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                               QPushButton, QSpinBox, QDoubleSpinBox, QListWidget,
                               QLineEdit, QTableWidget, QTableWidgetItem, QMessageBox,
                               QProgressDialog, QDialog, QHeaderView, QAction,
                               QFileDialog, QProgressBar, QApplication, QGroupBox, QTextEdit,
                               QTextBrowser)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QBrush, QColor, QFont

# 导入自定义模块
from data.data_fetcher import DataFetcher
from models.prediction import StockPredictor

# 导入策略模块
from strategies.fund_strategy import FundStrategy
from strategies.market_strategy import MarketStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.volume_price_strategy import VolumePriceStrategy

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self, system_components=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化GUI主窗口")
        
        # 保存系统组件
        self.system_components = system_components
        
        # 初始化数据和组件
        self._init_data()
        self._init_ui()
    
    def _init_data(self):
        """初始化数据"""
        # 初始化数据获取器
        self.data_fetcher = DataFetcher.get_instance()
        
        # 如果有系统组件，使用系统组件
        if self.system_components:
            self.cache_manager = self.system_components.get('cache_manager')
            self.data_source_manager = self.system_components.get('data_source_manager')
            self.data_pipeline = self.system_components.get('default_pipeline')
            self.data_processor = self.system_components.get('enhanced_processor')
            self.data_validator = self.system_components.get('data_validator')
        
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
    
    def _init_ui(self):
        """初始化界面"""
        # 设置窗口基本属性
        self.setWindowTitle("至简交易选股系统")
        self.setMinimumSize(1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        layout = QVBoxLayout(central_widget)
        
        # 创建标题标签
        title_label = QLabel("至简交易选股系统")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title_label)
        
        subtitle_label = QLabel("基于简放交易理念 + AI + 量化因子的股票筛选系统")
        subtitle_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(subtitle_label)
        
        # 创建标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 添加各个标签页
        tab_widget.addTab(self._create_screening_tab(), "选股")
        tab_widget.addTab(self._create_analysis_tab(), "个股分析")
        tab_widget.addTab(self._create_market_tab(), "市场概览")
        tab_widget.addTab(self._create_ai_tab(), "AI预测")
        
        # 创建状态栏
        self.statusBar().showMessage(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载行业列表
        self._load_industry_list()
        
        # 添加UI组件到工具栏
        toolbar = self.addToolBar("工具栏")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # 添加选股按钮
        self.select_stock_action = QAction("选股", self)
        self.select_stock_action.triggered.connect(self._on_select_stock_clicked)
        toolbar.addAction(self.select_stock_action)
        
        # 添加分析按钮
        self.analysis_action = QAction("技术分析", self)
        self.analysis_action.triggered.connect(self._on_analysis_clicked)
        toolbar.addAction(self.analysis_action)
        
        # 添加AI预测按钮
        self.prediction_action = QAction("AI预测", self)
        self.prediction_action.triggered.connect(self._on_prediction_clicked)
        toolbar.addAction(self.prediction_action)
        
        # 添加热门行业按钮
        self.hot_industries_action = QAction("热门行业", self)
        self.hot_industries_action.triggered.connect(self._on_hot_industries_clicked)
        toolbar.addAction(self.hot_industries_action)
        
        # 添加自定义交易建议按钮
        self.trading_advice_action = QAction("交易建议", self)
        self.trading_advice_action.triggered.connect(self._on_custom_advice_clicked)
        toolbar.addAction(self.trading_advice_action)
        
        # 添加设置按钮
        self.settings_action = QAction("设置", self)
        self.settings_action.triggered.connect(self._on_settings_clicked)
        toolbar.addAction(self.settings_action)
    
    def _load_industry_list(self):
        """加载行业列表"""
        try:
            # 获取东方财富行业分类
            industry_list = self.data_fetcher.get_industry_list()
            
            # 清空现有项
            self.industry_combo.clear()
            
            # 添加行业选项
            for industry in industry_list:
                self.industry_combo.addItem(industry)
                
            self.logger.info(f"已加载 {len(industry_list)} 个行业分类")
            
        except Exception as e:
            self.logger.error(f"加载行业列表失败: {str(e)}")
            # 添加默认的"全部"选项
            self.industry_combo.addItem("全部")

    def _create_screening_tab(self):
        """创建选股标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 创建筛选条件区域
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        
        # 左侧筛选条件
        left_group = QWidget()
        left_layout = QVBoxLayout(left_group)
        
        # 行业选择
        industry_label = QLabel("选择行业:")
        self.industry_combo = QComboBox()
        # 初始只添加"全部"选项，后续在_load_industry_list方法中加载完整列表
        self.industry_combo.addItem("全部")
        left_layout.addWidget(industry_label)
        left_layout.addWidget(self.industry_combo)
        
        # 策略选择
        strategy_label = QLabel("选择策略:")
        self.strategy_list = QListWidget()
        self.strategy_list.addItems(self.strategies.keys())
        self.strategy_list.setSelectionMode(QListWidget.MultiSelection)
        left_layout.addWidget(strategy_label)
        left_layout.addWidget(self.strategy_list)
        
        filter_layout.addWidget(left_group)
        
        # 右侧筛选条件
        right_group = QWidget()
        right_layout = QVBoxLayout(right_group)
        
        # 市值范围
        cap_label = QLabel("市值范围（亿元）:")
        cap_widget = QWidget()
        cap_layout = QHBoxLayout(cap_widget)
        self.min_cap = QSpinBox()
        self.min_cap.setRange(0, 5000)
        self.min_cap.setValue(50)
        self.max_cap = QSpinBox()
        self.max_cap.setRange(0, 5000)
        self.max_cap.setValue(1000)
        cap_layout.addWidget(self.min_cap)
        cap_layout.addWidget(QLabel("~"))
        cap_layout.addWidget(self.max_cap)
        right_layout.addWidget(cap_label)
        right_layout.addWidget(cap_widget)
        
        # 最小交易量
        volume_label = QLabel("最小日均交易量（万手）:")
        self.min_volume = QSpinBox()
        self.min_volume.setRange(0, 10000)
        self.min_volume.setValue(100)
        right_layout.addWidget(volume_label)
        right_layout.addWidget(self.min_volume)
        
        filter_layout.addWidget(right_group)
        layout.addWidget(filter_widget)
        
        # 创建筛选按钮
        screen_btn = QPushButton("开始筛选")
        screen_btn.clicked.connect(self._on_screen_clicked)
        
        # 创建自定义交易建议按钮
        custom_advice_btn = QPushButton("显示自定义交易建议")
        custom_advice_btn.clicked.connect(self._on_custom_advice_clicked)
        
        # 布局添加
        button_layout = QHBoxLayout()
        button_layout.addWidget(screen_btn)
        button_layout.addWidget(custom_advice_btn)
        
        # 修复：新增screening_layout属性
        self.screening_layout = layout
        self.screening_layout.addLayout(button_layout)
        
        # 创建结果表格
        self.result_table = QTableWidget()
        layout.addWidget(self.result_table)
        
        return tab
    
    def _create_analysis_tab(self):
        """创建个股分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 搜索框
        search_widget = QWidget()
        search_layout = QHBoxLayout(search_widget)
        search_label = QLabel("输入股票代码或名称:")
        self.stock_search = QLineEdit()
        search_btn = QPushButton("查找")
        search_btn.clicked.connect(self._on_search_clicked)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.stock_search)
        search_layout.addWidget(search_btn)
        layout.addWidget(search_widget)
        
        # 分析按钮区域
        analysis_widget = QWidget()
        analysis_layout = QHBoxLayout(analysis_widget)
        
        # 添加各种分析按钮
        trend_btn = QPushButton("趋势分析")
        trend_btn.clicked.connect(lambda: self._analyze_stock("trend"))
        
        volume_price_btn = QPushButton("量价分析")
        volume_price_btn.clicked.connect(lambda: self._analyze_stock("volume_price"))
        
        fund_btn = QPushButton("资金分析")
        fund_btn.clicked.connect(lambda: self._analyze_stock("fund"))
        
        predict_btn = QPushButton("AI预测")
        predict_btn.clicked.connect(lambda: self._analyze_stock("predict"))
        
        analysis_layout.addWidget(trend_btn)
        analysis_layout.addWidget(volume_price_btn)
        analysis_layout.addWidget(fund_btn)
        analysis_layout.addWidget(predict_btn)
        
        layout.addWidget(analysis_widget)
        
        # 分析结果区域
        result_label = QLabel("分析结果")
        result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(result_label)
        
        self.analysis_result = QTableWidget()
        layout.addWidget(self.analysis_result)
        
        return tab
    
    def _create_market_tab(self):
        """创建市场概览标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 指数选择
        index_widget = QWidget()
        index_layout = QHBoxLayout(index_widget)
        index_label = QLabel("选择指数:")
        self.index_combo = QComboBox()
        self.index_combo.addItems(["上证指数", "深证成指", "创业板指", "沪深300"])
        index_layout.addWidget(index_label)
        index_layout.addWidget(self.index_combo)
        layout.addWidget(index_widget)
        
        # 分析按钮区域
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        
        # 指数分析按钮
        index_analysis_btn = QPushButton("指数分析")
        index_analysis_btn.clicked.connect(self._on_index_analysis_clicked)
        
        # 市场情绪按钮
        market_sentiment_btn = QPushButton("市场情绪")
        market_sentiment_btn.clicked.connect(self._on_market_sentiment_clicked)
        
        # 北向资金按钮
        north_fund_btn = QPushButton("北向资金")
        north_fund_btn.clicked.connect(self._on_north_fund_clicked)
        
        # 市场宽度按钮
        market_breadth_btn = QPushButton("市场宽度")
        market_breadth_btn.clicked.connect(self._on_market_breadth_clicked)
        
        # 热门行业分析按钮
        hot_industries_btn = QPushButton("热门行业分析")
        hot_industries_btn.clicked.connect(self._on_hot_industries_clicked)
        
        # 添加按钮到布局
        buttons_layout.addWidget(index_analysis_btn)
        buttons_layout.addWidget(market_sentiment_btn)
        buttons_layout.addWidget(north_fund_btn)
        buttons_layout.addWidget(market_breadth_btn)
        buttons_layout.addWidget(hot_industries_btn)
        layout.addWidget(buttons_widget)
        
        # 分析结果显示区域
        result_label = QLabel("市场分析结果")
        result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(result_label)
        
        self.market_result = QTableWidget()
        layout.addWidget(self.market_result)
        
        return tab
    
    def _create_ai_tab(self):
        """创建AI预测标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 股票选择
        stock_widget = QWidget()
        stock_layout = QHBoxLayout(stock_widget)
        stock_label = QLabel("输入股票代码或名称:")
        self.pred_stock_search = QLineEdit()
        search_btn = QPushButton("查找")
        search_btn.clicked.connect(self._on_ai_search_clicked)
        stock_layout.addWidget(stock_label)
        stock_layout.addWidget(self.pred_stock_search)
        stock_layout.addWidget(search_btn)
        layout.addWidget(stock_widget)
        
        # 预测参数区域
        param_widget = QWidget()
        param_layout = QHBoxLayout(param_widget)
        
        # 预测天数
        days_label = QLabel("预测天数:")
        self.pred_days = QSpinBox()
        self.pred_days.setRange(1, 30)
        self.pred_days.setValue(5)
        
        # 模型选择
        model_label = QLabel("模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "Prophet", "综合模型"])
        
        # 参数添加到布局
        param_layout.addWidget(days_label)
        param_layout.addWidget(self.pred_days)
        param_layout.addWidget(model_label)
        param_layout.addWidget(self.model_combo)
        layout.addWidget(param_widget)
        
        # 分析按钮
        analyze_btn = QPushButton("AI分析与交易建议")
        analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        analyze_btn.clicked.connect(self._on_ai_analyze_clicked)
        layout.addWidget(analyze_btn)
        
        # 预测结果区域
        result_label = QLabel("AI预测结果")
        result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(result_label)
        
        self.ai_result = QTableWidget()
        layout.addWidget(self.ai_result)
        
        # 交易建议区域
        advice_label = QLabel("交易指导建议")
        advice_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(advice_label)
        
        self.advice_result = QTableWidget()
        layout.addWidget(self.advice_result)
        
        return tab
    
    def _on_screen_clicked(self):
        """处理筛选按钮点击事件"""
        try:
            # 获取筛选条件
            selected_industry = self.industry_combo.currentText()
            selected_strategies = [item.text() for item in self.strategy_list.selectedItems()]
            
            if not selected_strategies:
                QMessageBox.warning(self, "警告", "请至少选择一种策略！")
                return

            # 显示选择的策略要求
            strategy_count = len(selected_strategies)
            if strategy_count < 3:
                result = QMessageBox.question(
                    self, "策略确认", 
                    f"您选择了{strategy_count}个策略，建议选择3-4个策略进行多因子联合筛选，是否继续？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if result == QMessageBox.No:
                    return
                
            # 根据行业获取股票列表
            if selected_industry == "全部":
                filtered_stocks = self._get_stock_list()
            else:
                filtered_stocks = self.data_fetcher.get_industry_stocks(selected_industry)
            
            if filtered_stocks is None or filtered_stocks.empty:
                QMessageBox.critical(self, "错误", f"无法获取行业 '{selected_industry}' 的股票列表，请检查数据源配置！")
                return
            
            # 显示正在筛选的提示
            self.statusBar().showMessage(f"正在筛选 {selected_industry} 行业的股票，共 {len(filtered_stocks)} 只...")
            
            # 存储每个策略筛选的结果和每支股票符合的策略数量
            strategy_results = {}
            stock_strategy_count = {}
            
            # 应用选择的策略并记录每支股票符合的策略数量
            for strategy_name in selected_strategies:
                strategy = self.strategies[strategy_name]
                strategy_result = strategy.screen(filtered_stocks)
                if strategy_result is None or strategy_result.empty:
                    self.logger.warning(f"策略 '{strategy_name}' 未返回有效结果")
                    continue
                    
                strategy_results[strategy_name] = strategy_result
                
                # 确保ts_code列存在
                if 'ts_code' not in strategy_result.columns:
                    self.logger.warning(f"策略 '{strategy_name}' 的结果中缺少'ts_code'列")
                    continue
                
                # 记录每支股票符合的策略
                for _, stock in strategy_result.iterrows():
                    stock_code = stock['ts_code']
                    if stock_code not in stock_strategy_count:
                        stock_strategy_count[stock_code] = {'count': 0, 'strategies': []}
                    stock_strategy_count[stock_code]['count'] += 1
                    stock_strategy_count[stock_code]['strategies'].append(strategy_name)
            
            # 如果没有任何有效的策略结果，则返回
            if not strategy_results:
                QMessageBox.information(self, "提示", "所选策略没有返回有效的筛选结果，请尝试其他策略")
                self.statusBar().showMessage("筛选完成，未找到符合条件的股票")
                return
            
            # 筛选满足所有策略条件的股票
            passing_stock_codes = [code for code, info in stock_strategy_count.items() 
                                if info['count'] == len(selected_strategies)]
            
            # 如果要求太严格，找不到股票，则退而求其次，找到满足大部分策略的股票
            min_strategy_count = len(selected_strategies)  # 初始要求满足所有策略
            while len(passing_stock_codes) == 0 and min_strategy_count > 0:
                min_strategy_count -= 1
                passing_stock_codes = [code for code, info in stock_strategy_count.items() 
                                    if info['count'] >= min_strategy_count]
            
            # 从原始股票列表中筛选出符合条件的股票
            filtered_results = []
            for strategy_result in strategy_results.values():
                if 'ts_code' in strategy_result.columns:
                    result_subset = strategy_result[strategy_result['ts_code'].isin(passing_stock_codes)]
                    filtered_results.append(result_subset)
            
            # 合并结果
            if filtered_results:
                # 确保每个结果集有相同的列
                common_columns = set.intersection(*[set(df.columns) for df in filtered_results])
                if not common_columns:
                    common_columns = {'ts_code', 'name'}  # 至少需要这两列
                
                # 筛选公共列
                filtered_results = [df[list(common_columns)].copy() for df in filtered_results if set(common_columns).issubset(df.columns)]
                
                if not filtered_results:
                    QMessageBox.information(self, "提示", "策略结果格式不一致，无法合并，请检查策略实现")
                    self.statusBar().showMessage("筛选完成，无法合并结果")
                    return
                
                final_result = pd.concat(filtered_results).drop_duplicates(subset=['ts_code'])
                
                # 添加符合的策略信息
                final_result['strategy_info'] = final_result['ts_code'].apply(
                    lambda x: ', '.join(stock_strategy_count[x]['strategies']) if x in stock_strategy_count else ''
                )
                final_result['strategy_count'] = final_result['ts_code'].apply(
                    lambda x: stock_strategy_count[x]['count'] if x in stock_strategy_count else 0
                )
                
                # 按符合策略数量降序排序
                final_result = final_result.sort_values(by='strategy_count', ascending=False)
                
                # 生成交易建议
                self._add_trading_advice(final_result, stock_strategy_count, selected_strategies)
                
                # 显示结果
                self._display_results(final_result)
                
                message = f"筛选完成，找到 {len(final_result)} 只符合条件的股票"
                if min_strategy_count < len(selected_strategies):
                    message += f"（满足至少 {min_strategy_count} 个策略）"
                else:
                    message += "（满足全部策略）"
                
                QMessageBox.information(self, "完成", message)
                self.statusBar().showMessage(message + "。点击股票查看详细分析")
            else:
                QMessageBox.information(self, "提示", "未找到符合条件的股票，请尝试减少策略条件")
                self.statusBar().showMessage("未找到符合条件的股票")
            
        except Exception as e:
            self.logger.error(f"筛选过程出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"筛选过程出错: {str(e)}")
            self.statusBar().showMessage("筛选过程出错，请查看日志")
            
    def _add_trading_advice(self, result_df, stock_strategy_count, selected_strategies):
        """向结果添加交易建议"""
        try:
            # 导入自定义交易建议数据
            try:
                from data.custom_stock_data import get_custom_advice
                use_custom_data = True
            except ImportError:
                use_custom_data = False
                self.logger.warning("未找到自定义交易建议数据模块，将使用算法生成交易建议")
            
            # 添加交易建议列
            result_df['trading_advice'] = ""
            result_df['entry_price'] = 0.0
            result_df['stop_loss'] = 0.0
            result_df['target_price'] = 0.0
            
            for i, row in result_df.iterrows():
                try:
                    # 确保ts_code列存在
                    if 'ts_code' not in row:
                        self.logger.warning(f"行中缺少'ts_code'列，跳过此行")
                        continue
                        
                    stock_code = row['ts_code']
                    
                    # 检查是否有自定义交易建议
                    if use_custom_data:
                        custom_advice = get_custom_advice(stock_code)
                        if custom_advice:
                            result_df.loc[i, 'trading_advice'] = custom_advice['trading_advice']
                            result_df.loc[i, 'entry_price'] = custom_advice['entry_price']
                            result_df.loc[i, 'stop_loss'] = custom_advice['stop_loss']
                            result_df.loc[i, 'target_price'] = custom_advice['target_price']
                            # 跳过算法生成的交易建议
                            continue
                    
                    # 检查stock_code是否在stock_strategy_count中
                    if stock_code not in stock_strategy_count:
                        self.logger.warning(f"股票代码 {stock_code} 不在策略计数中，跳过此行")
                        continue
                        
                    strategy_count = stock_strategy_count[stock_code]['count']
                    
                    # 获取股票数据
                    price_df = self.data_fetcher.get_stock_kline(stock_code)
                    if price_df is None or price_df.empty:
                        self.logger.warning(f"无法获取股票 {stock_code} 的K线数据，跳过交易建议")
                        continue
                        
                    # 提取最近的价格
                    if 'close' not in price_df.columns:
                        self.logger.warning(f"股票 {stock_code} 的K线数据中缺少'close'列，跳过交易建议")
                        continue
                        
                    current_price = price_df['close'].iloc[-1]
                    
                    # 分析技术面指标
                    trend_strategy = self.strategies["趋势策略"]
                    ma_df = trend_strategy.calculate_ma(price_df)
                    ma_aligned = False
                    pullback = False
                    
                    if ma_df is not None:
                        ma_aligned = trend_strategy.check_ma_alignment(ma_df)
                        pullback = trend_strategy.check_pullback(price_df, ma_df)
                    
                    price_breakthrough = trend_strategy.check_price_breakthrough(price_df)
                    
                    # 生成交易建议
                    if strategy_count == len(selected_strategies):
                        if ma_aligned and price_breakthrough:
                            advice = "强烈买入"
                            entry_price = round(current_price, 2)  # 市价买入
                            stop_loss = round(current_price * 0.95, 2)  # 止损位：当前价格的95%
                            target_price = round(current_price * 1.15, 2)  # 目标价：当前价格的115%
                        elif ma_aligned or price_breakthrough:
                            advice = "适量买入"
                            entry_price = round(current_price * 0.98, 2)  # 稍低于市价
                            stop_loss = round(current_price * 0.93, 2)  # 止损位：当前价格的93%
                            target_price = round(current_price * 1.10, 2)  # 目标价：当前价格的110%
                        else:
                            advice = "观察买入"
                            entry_price = round(current_price * 0.95, 2)  # 回调位
                            stop_loss = round(current_price * 0.90, 2)  # 止损位：当前价格的90%
                            target_price = round(current_price * 1.08, 2)  # 目标价：当前价格的108%
                    elif strategy_count >= len(selected_strategies) - 1:
                        if ma_aligned:
                            advice = "分批买入"
                            entry_price = round(current_price * 0.97, 2)  # 稍低于市价
                            stop_loss = round(current_price * 0.92, 2)  # 止损位：当前价格的92%
                            target_price = round(current_price * 1.08, 2)  # 目标价：当前价格的108%
                        else:
                            advice = "谨慎买入"
                            entry_price = round(current_price * 0.95, 2)  # 回调位
                            stop_loss = round(current_price * 0.90, 2)  # 止损位：当前价格的90%
                            target_price = round(current_price * 1.06, 2)  # 目标价：当前价格的106%
                    else:
                        advice = "观望"
                        entry_price = round(current_price * 0.93, 2)  # 大幅回调
                        stop_loss = round(current_price * 0.88, 2)  # 止损位：当前价格的88%
                        target_price = round(current_price * 1.05, 2)  # 目标价：当前价格的105%
                    
                    # 更新结果DataFrame
                    result_df.loc[i, 'trading_advice'] = advice
                    result_df.loc[i, 'entry_price'] = entry_price
                    result_df.loc[i, 'stop_loss'] = stop_loss
                    result_df.loc[i, 'target_price'] = target_price
                    
                except Exception as e:
                    self.logger.error(f"生成单个股票的交易建议出错: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"生成交易建议整体过程出错: {str(e)}")
            QMessageBox.warning(self, "警告", f"生成交易建议过程中出现问题: {str(e)}")
    
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
    
    def _display_results(self, results):
        """显示筛选结果"""
        # 添加策略信息列
        if 'strategy_info' not in results.columns:
            results['strategy_info'] = ""
            
        # 重新排列列的顺序，将交易建议相关列放在前面
        if 'trading_advice' in results.columns:
            cols = ['ts_code', 'name', 'trading_advice', 'entry_price', 'stop_loss', 'target_price', 'strategy_count', 'strategy_info']
            other_cols = [col for col in results.columns if col not in cols]
            cols.extend(other_cols)
            results = results[cols]
            
        # 设置表格列
        self.result_table.setColumnCount(len(results.columns))
        
        # 将列标题翻译为中文
        column_labels = []
        for col in results.columns:
            if col == 'ts_code':
                column_labels.append('股票代码')
            elif col == 'name':
                column_labels.append('股票名称')
            elif col == 'trading_advice':
                column_labels.append('交易建议')
            elif col == 'entry_price':
                column_labels.append('入场价')
            elif col == 'stop_loss':
                column_labels.append('止损价')
            elif col == 'target_price':
                column_labels.append('目标价')
            elif col == 'strategy_count':
                column_labels.append('满足策略数')
            elif col == 'strategy_info':
                column_labels.append('策略信息')
            else:
                column_labels.append(col)
        
        self.result_table.setHorizontalHeaderLabels(column_labels)
        
        # 设置表格行数据
        self.result_table.setRowCount(len(results))
        for i, (idx, row) in enumerate(results.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                
                # 设置颜色
                column_name = results.columns[j]
                if column_name == 'trading_advice':
                    if '强烈买入' in str(value) or '适量买入' in str(value):
                        item.setForeground(Qt.red)
                    elif '谨慎买入' in str(value) or '分批买入' in str(value):
                        item.setForeground(Qt.darkRed)
                    elif '观望' in str(value):
                        item.setForeground(Qt.blue)
                    elif '减仓' in str(value) or '卖出' in str(value):
                        item.setForeground(Qt.green)
                        
                self.result_table.setItem(i, j, item)
        
        # 调整列宽
        self.result_table.resizeColumnsToContents()
        
        # 显示策略解释
        self.statusBar().showMessage(f"筛选完成，共找到 {len(results)} 只符合条件的股票。点击行查看详细分析")
        
        # 连接行点击事件
        self.result_table.cellClicked.connect(self._on_result_row_clicked)
    
    def _on_search_clicked(self):
        """处理股票搜索按钮点击事件"""
        search_text = self.stock_search.text().strip()
        if not search_text:
            QMessageBox.warning(self, "警告", "请输入股票代码或名称！")
            return
            
        try:
            # 获取股票列表
            stock_list = self._get_stock_list()
            if stock_list.empty:
                QMessageBox.critical(self, "错误", "无法获取股票列表，请检查数据源配置！")
                return
                
            # 搜索股票
            matched_stocks = stock_list[
                stock_list["ts_code"].str.contains(search_text) | 
                stock_list["name"].str.contains(search_text)
            ]
            
            if matched_stocks.empty:
                QMessageBox.information(self, "提示", "未找到匹配的股票，请检查输入！")
                return
                
            # 显示匹配的股票
            self.analysis_result.setColumnCount(len(matched_stocks.columns))
            
            # 将列标题翻译为中文
            column_labels = []
            for col in matched_stocks.columns:
                if col == 'ts_code':
                    column_labels.append('股票代码')
                elif col == 'name':
                    column_labels.append('股票名称')
                elif col == 'area':
                    column_labels.append('地区')
                elif col == 'industry':
                    column_labels.append('行业')
                elif col == 'market':
                    column_labels.append('市场')
                elif col == 'list_date':
                    column_labels.append('上市日期')
                elif col == 'list_status':
                    column_labels.append('上市状态')
                else:
                    column_labels.append(col)
            
            self.analysis_result.setHorizontalHeaderLabels(column_labels)
            
            self.analysis_result.setRowCount(len(matched_stocks))
            for i, (_, row) in enumerate(matched_stocks.iterrows()):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.analysis_result.setItem(i, j, item)
            
            # 调整列宽
            self.analysis_result.resizeColumnsToContents()
            
            # 存储当前搜索的股票代码
            if len(matched_stocks) == 1:
                stock_code = matched_stocks["ts_code"].iloc[0]
                self.current_analysis_stock = stock_code
                self.statusBar().showMessage(f"当前选中股票: {stock_code}")
            
        except Exception as e:
            self.logger.error(f"搜索股票过程出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"搜索股票过程出错: {str(e)}")
    
    def _analyze_stock(self, analysis_type):
        """处理股票分析事件"""
        if not hasattr(self, 'current_analysis_stock'):
            QMessageBox.warning(self, "警告", "请先搜索并选择一只股票！")
            return
            
        stock_code = self.current_analysis_stock
        
        try:
            # 获取股票数据
            price_df = self.data_fetcher.get_stock_kline(stock_code)
            
            if price_df is None or price_df.empty:
                QMessageBox.warning(self, "警告", f"无法获取股票 {stock_code} 的行情数据！")
                return
                
            # 根据分析类型执行不同的分析
            result_df = None
            if analysis_type == "trend":
                # 趋势分析
                strategy = self.strategies["趋势策略"]
                ma_df = strategy.calculate_ma(price_df)
                ma_aligned = strategy.check_ma_alignment(ma_df)
                price_breakthrough = strategy.check_price_breakthrough(price_df)
                pullback = strategy.check_pullback(price_df, ma_df)
                
                result_df = pd.DataFrame({
                    "分析项目": ["均线多头排列", "价格突破新高", "回调企稳"],
                    "分析结果": [
                        "是" if ma_aligned else "否", 
                        "是" if price_breakthrough else "否",
                        "是" if pullback else "否"
                    ],
                    "分析说明": [
                        "20日、60日、120日均线多头排列，趋势良好" if ma_aligned else "均线未形成多头排列，趋势较弱",
                        "价格突破近期高点，上涨趋势明显" if price_breakthrough else "价格未突破近期高点，缺乏突破动力",
                        "回调已接近支撑位，存在反弹机会" if pullback else "回调幅度不符合要求，或未触及支撑位"
                    ]
                })
                
            elif analysis_type == "volume_price":
                # 量价分析
                strategy = self.strategies["量价策略"]
                breakthrough_vol = strategy.check_breakthrough_volume(price_df, price_df)
                pullback_vol = strategy.check_pullback_volume(price_df, price_df)
                second_vol = strategy.check_second_volume_increase(price_df, price_df)
                
                result_df = pd.DataFrame({
                    "分析项目": ["突破放量", "回调缩量", "二次放量"],
                    "分析结果": [
                        "是" if breakthrough_vol else "否", 
                        "是" if pullback_vol else "否",
                        "是" if second_vol else "否"
                    ],
                    "分析说明": [
                        "突破时成交量明显放大，买盘积极" if breakthrough_vol else "突破时成交量不足，缺乏持续性",
                        "回调时成交量明显萎缩，抛压减轻" if pullback_vol else "回调时成交量未明显萎缩，抛压仍大",
                        "二次上攻时成交量再次放大，买盘再度积极" if second_vol else "未出现二次放量上攻，缺乏持续买盘"
                    ]
                })
                
            elif analysis_type == "fund":
                # 资金分析
                strategy = self.strategies["基本面策略"]
                money_flow_df = self.data_fetcher.get_money_flow(stock_code)
                
                if money_flow_df is None or money_flow_df.empty:
                    QMessageBox.warning(self, "警告", f"无法获取股票 {stock_code} 的资金流向数据！")
                    return
                    
                main_inflow = strategy.check_main_fund_inflow(money_flow_df)
                
                result_df = pd.DataFrame({
                    "分析项目": ["主力资金流入", "机构买入", "北向资金持股"],
                    "分析结果": [
                        "是" if main_inflow else "否", 
                        "未知",  # 实际应用中需要获取龙虎榜数据
                        "未知"   # 实际应用中需要获取北向资金数据
                    ],
                    "分析说明": [
                        "近期主力资金持续净流入，做多意愿强" if main_inflow else "主力资金净流出或流入不足，做多意愿弱",
                        "需要龙虎榜数据支持分析",
                        "需要北向资金数据支持分析"
                    ]
                })
                
            elif analysis_type == "predict":
                # AI预测
                pred_days = 5
                pred_result = self.predictor.predict_price(price_df, days=pred_days)
                
                if pred_result is None:
                    QMessageBox.warning(self, "警告", "AI预测失败，请检查模型配置！")
                    return
                    
                last_close = price_df["close"].iloc[-1]
                pred_change = (pred_result[-1] / last_close - 1) * 100
                
                result_df = pd.DataFrame({
                    "预测日期": [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)],
                    "预测价格": [round(p, 2) for p in pred_result],
                    "涨跌幅(%)": [round((p / last_close - 1) * 100, 2) for p in pred_result]
                })
            
            # 显示分析结果
            if result_df is not None:
                self.analysis_result.setColumnCount(len(result_df.columns))
                self.analysis_result.setHorizontalHeaderLabels(result_df.columns)
                
                self.analysis_result.setRowCount(len(result_df))
                for i, (_, row) in enumerate(result_df.iterrows()):
                    for j, value in enumerate(row):
                        item = QTableWidgetItem(str(value))
                        self.analysis_result.setItem(i, j, item)
                
                # 调整列宽
                self.analysis_result.resizeColumnsToContents()
                
        except Exception as e:
            self.logger.error(f"分析股票过程出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"分析股票过程出错: {str(e)}")
    
    def _on_result_row_clicked(self, row, column):
        """处理结果表格行点击事件"""
        # 获取所选行的股票代码和名称
        ts_code_idx = self.result_table.horizontalHeaderItem(0).text() == 'ts_code' and 0 or \
                      self.result_table.horizontalHeaderItem(1).text() == 'ts_code' and 1
        name_idx = self.result_table.horizontalHeaderItem(0).text() == 'name' and 0 or \
                   self.result_table.horizontalHeaderItem(1).text() == 'name' and 1 or \
                   self.result_table.horizontalHeaderItem(2).text() == 'name' and 2
        
        if ts_code_idx >= 0:
            stock_code = self.result_table.item(row, ts_code_idx).text()
            stock_name = name_idx >= 0 and self.result_table.item(row, name_idx).text() or "未知"
            
            # 获取股票数据并执行分析
            try:
                # 获取K线数据
                price_df = self.data_fetcher.get_stock_kline(stock_code)
                if price_df is None or price_df.empty:
                    QMessageBox.warning(self, "警告", f"无法获取{stock_code}的K线数据")
                    return
                
                # 执行各策略分析
                analysis_results = {}
                
                # 趋势分析
                trend_strategy = self.strategies["趋势策略"]
                ma_df = trend_strategy.calculate_ma(price_df)
                ma_aligned = trend_strategy.check_ma_alignment(ma_df)
                price_breakthrough = trend_strategy.check_price_breakthrough(price_df)
                pullback = trend_strategy.check_pullback(price_df, ma_df)
                
                analysis_results["趋势分析"] = {
                    "均线多头排列": ma_aligned,
                    "价格突破新高": price_breakthrough,
                    "回调企稳": pullback
                }
                
                # 量价分析
                vp_strategy = self.strategies["量价策略"]
                breakthrough_vol = vp_strategy.check_breakthrough_volume(price_df, price_df)
                pullback_vol = vp_strategy.check_pullback_volume(price_df, price_df)
                second_vol = vp_strategy.check_second_volume_increase(price_df, price_df)
                
                analysis_results["量价分析"] = {
                    "突破放量": breakthrough_vol,
                    "回调缩量": pullback_vol,
                    "二次放量": second_vol
                }
                
                # 资金分析
                try:
                    fund_strategy = self.strategies["基本面策略"]
                    money_flow_df = self.data_fetcher.get_money_flow(stock_code)
                    if money_flow_df is not None and not money_flow_df.empty:
                        main_inflow = fund_strategy.check_main_fund_inflow(money_flow_df)
                        analysis_results["资金分析"] = {
                            "主力资金流入": main_inflow
                        }
                except Exception as e:
                    self.logger.warning(f"资金分析出错: {e}")
                
                # 生成详细分析结果
                detail_text = f"股票: {stock_name}({stock_code}) 详细分析:\n\n"
                
                for category, items in analysis_results.items():
                    detail_text += f"【{category}】\n"
                    for item, result in items.items():
                        detail_text += f"- {item}: {'✓' if result else '✗'}"
                        if item == "均线多头排列" and result:
                            detail_text += " (20日、60日、120日均线多头排列，趋势良好)"
                        elif item == "均线多头排列" and not result:
                            detail_text += " (均线未形成多头排列，趋势较弱)"
                        elif item == "价格突破新高" and result:
                            detail_text += " (价格突破近期高点，上涨趋势明显)"
                        elif item == "价格突破新高" and not result:
                            detail_text += " (价格未突破近期高点，缺乏突破动力)"
                        elif item == "回调企稳" and result:
                            detail_text += " (回调已接近支撑位，存在反弹机会)"
                        elif item == "回调企稳" and not result:
                            detail_text += " (回调幅度不符合要求，或未触及支撑位)"
                        elif item == "突破放量" and result:
                            detail_text += " (突破时成交量明显放大，买盘积极)"
                        elif item == "突破放量" and not result:
                            detail_text += " (突破时成交量不足，缺乏持续性)"
                        elif item == "回调缩量" and result:
                            detail_text += " (回调时成交量明显萎缩，抛压减轻)"
                        elif item == "回调缩量" and not result:
                            detail_text += " (回调时成交量未明显萎缩，抛压仍大)"
                        elif item == "二次放量" and result:
                            detail_text += " (二次上攻时成交量再次放大，买盘再度积极)"
                        elif item == "二次放量" and not result:
                            detail_text += " (未出现二次放量上攻，缺乏持续买盘)"
                        elif item == "主力资金流入" and result:
                            detail_text += " (近期主力资金持续净流入，做多意愿强)"
                        elif item == "主力资金流入" and not result:
                            detail_text += " (主力资金净流出或流入不足，做多意愿弱)"
                        detail_text += "\n"
                    detail_text += "\n"
                
                # 添加总结
                score = 0
                max_score = 0
                for category, items in analysis_results.items():
                    for _, result in items.items():
                        max_score += 1
                        if result:
                            score += 1
                
                rating = score / max_score if max_score > 0 else 0
                if rating >= 0.7:
                    strength = "强势"
                elif rating >= 0.5:
                    strength = "偏强"
                elif rating >= 0.3:
                    strength = "偏弱"
                else:
                    strength = "弱势"
                
                detail_text += f"综合评分: {score}/{max_score}，股票表现{strength}"
                
                # 显示详细分析结果
                QMessageBox.information(self, f"{stock_name} 详细分析", detail_text)
                
            except Exception as e:
                self.logger.error(f"分析股票详情出错: {str(e)}")
                QMessageBox.critical(self, "错误", f"分析股票详情出错: {str(e)}")
    
    def _on_ai_analyze_clicked(self):
        """处理AI分析按钮点击事件"""
        if not hasattr(self, 'current_ai_stock'):
            QMessageBox.warning(self, "警告", "请先搜索并选择一只股票！")
            return
            
        stock_code = self.current_ai_stock
        stock_name = getattr(self, 'current_ai_stock_name', "未知")
        
        try:
            # 获取参数
            pred_days = self.pred_days.value()
            model_type = self.model_combo.currentText()
            if model_type == "LSTM":
                ai_model = "lstm"
            elif model_type == "Prophet":
                ai_model = "prophet"
            else:
                ai_model = "ensemble"  # 综合模型
                
            # 获取股票数据
            price_df = self.data_fetcher.get_stock_kline(stock_code)
            
            if price_df is None or price_df.empty:
                QMessageBox.warning(self, "警告", f"无法获取股票 {stock_code} 的行情数据！")
                return
                
            # 执行AI预测
            pred_result = self.predictor.predict_price(price_df, days=pred_days, model_type=ai_model)
            
            if pred_result is None:
                QMessageBox.warning(self, "警告", "AI预测失败，请检查模型配置！")
                return
                
            # 计算预测结果
            last_close = price_df["close"].iloc[-1]
            pred_changes = [(p / last_close - 1) * 100 for p in pred_result]
            
            # 创建预测结果DataFrame
            result_df = pd.DataFrame({
                "预测日期": [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(pred_days)],
                "预测价格": [round(p, 2) for p in pred_result],
                "涨跌幅(%)": [round(c, 2) for c in pred_changes]
            })
            
            # 显示预测结果
            self.ai_result.setColumnCount(len(result_df.columns))
            self.ai_result.setHorizontalHeaderLabels(result_df.columns)
            
            self.ai_result.setRowCount(len(result_df))
            for i, (_, row) in enumerate(result_df.iterrows()):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    # 根据涨跌幅设置颜色
                    if j == 2:  # 涨跌幅列
                        value_float = float(value)
                        if value_float > 0:
                            item.setForeground(Qt.red)
                        elif value_float < 0:
                            item.setForeground(Qt.green)
                    self.ai_result.setItem(i, j, item)
            
            # 调整列宽
            self.ai_result.resizeColumnsToContents()
            
            # 生成交易建议
            self._generate_trading_advice(stock_code, stock_name, price_df, pred_result, pred_changes)
            
        except Exception as e:
            self.logger.error(f"AI分析出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"AI分析出错: {str(e)}")

    def _generate_trading_advice(self, stock_code, stock_name, price_df, pred_prices, pred_changes):
        """生成详细的交易建议"""
        try:
            # 获取当前价格
            current_price = price_df["close"].iloc[-1]
            
            # 分析预测趋势
            avg_change = sum(pred_changes) / len(pred_changes)
            max_price = max(pred_prices)
            min_price = min(pred_prices)
            last_price = pred_prices[-1]
            
            # 分析K线形态
            trend_strategy = self.strategies["趋势策略"]
            ma_df = trend_strategy.calculate_ma(price_df)
            ma_aligned = False
            pullback = False
            
            if ma_df is not None:
                ma_aligned = trend_strategy.check_ma_alignment(ma_df)
                pullback = trend_strategy.check_pullback(price_df, ma_df)
            
            price_breakthrough = trend_strategy.check_price_breakthrough(price_df)
            
            # 分析量价关系
            vp_strategy = self.strategies["量价策略"]
            breakthrough_vol = vp_strategy.check_breakthrough_volume(price_df, price_df)
            
            # 制定交易策略
            if avg_change > 3:  # 平均涨幅大于3%，强烈看多
                suggestion = "强烈买入"
                action = "分批买入"
                reason = "AI预测未来价格持续上涨，涨幅明显"
            elif avg_change > 1:  # 平均涨幅在1%-3%之间，看多
                suggestion = "买入"
                action = "适量买入"
                reason = "AI预测未来价格温和上涨"
            elif avg_change > 0:  # 平均涨幅在0-1%之间，看多但谨慎
                suggestion = "谨慎买入"
                action = "小仓位试探性买入"
                reason = "AI预测未来价格略有上涨，涨幅有限"
            elif avg_change > -1:  # 平均跌幅在0-1%之间，观望
                suggestion = "观望"
                action = "暂不操作"
                reason = "AI预测未来价格略有下跌，但幅度较小，可等待更好的买点"
            elif avg_change > -3:  # 平均跌幅在1%-3%之间，谨慎看空
                suggestion = "减仓"
                action = "部分仓位止盈或止损"
                reason = "AI预测未来价格将有一定程度的下跌"
            else:  # 平均跌幅大于3%，强烈看空
                suggestion = "卖出"
                action = "清仓或大幅减仓"
                reason = "AI预测未来价格将明显下跌"
            
            # 计算建议的入场价、止损位和目标价
            if avg_change > 0:  # 看多策略
                # 入场价：当前价格的98%或最近支撑位
                entry_price = round(current_price * 0.98, 2)
                
                # 止损位：入场价的95%或预测最低价的95%，取较大值
                stop_loss = round(max(entry_price * 0.95, min_price * 0.95), 2)
                
                # 目标价：当前价格 + (预测最高价 - 当前价格) * 0.8，即预测最高价的80%收益
                target_price = round(current_price + (max_price - current_price) * 0.8, 2)
                
            else:  # 看空策略
                # 入场价（卖出价）：当前价格的102%或最近阻力位
                entry_price = round(current_price * 1.02, 2)
                
                # 止损位（此时为止盈位）：入场价的105%
                stop_loss = round(entry_price * 1.05, 2)
                
                # 目标价（此时为回补价）：当前价格 - (当前价格 - 预测最低价) * 0.8
                target_price = round(current_price - (current_price - min_price) * 0.8, 2)
            
            # 综合技术面因素
            if ma_aligned and price_breakthrough and breakthrough_vol and avg_change > 0:
                confidence = "高"
                confidence_reason = "AI预测看多，同时技术指标显示趋势强，均线多头排列，突破新高且成交量配合"
            elif ma_aligned and avg_change > 0:
                confidence = "中等偏上"
                confidence_reason = "AI预测看多，均线多头排列，趋势向好"
            elif not ma_aligned and not price_breakthrough and avg_change < 0:
                confidence = "高"
                confidence_reason = "AI预测看空，同时技术指标显示趋势弱，均线无多头排列，未突破新高"
            else:
                confidence = "中等"
                confidence_reason = "AI预测与部分技术指标存在分歧，建议谨慎操作"
            
            # 风险提示
            if avg_change > 0:
                risk = "下方支撑不足，可能跳空低开；大盘突然转弱，带动个股下跌"
            else:
                risk = "上方套牢盘较多，可能触发解套卖出；大盘突然转强，带动个股反弹"
            
            # 创建交易建议结果
            advice_df = pd.DataFrame({
                "交易建议项目": [
                    "操作建议", "操作方式", "建议理由", 
                    "建议入场价", "止损位", "目标价", 
                    "预期收益", "风险等级", "置信度", "风险提示"
                ],
                "建议内容": [
                    suggestion, action, reason,
                    f"{entry_price} 元", f"{stop_loss} 元", f"{target_price} 元",
                    f"{round(abs((target_price/entry_price - 1) * 100), 2)}%", 
                    "中等" if abs(avg_change) < 5 else "高",
                    confidence,
                    risk
                ],
                "补充说明": [
                    f"基于AI预测的{pred_changes[-1]:.2f}%涨跌幅",
                    f"根据预测趋势和技术指标综合制定",
                    f"AI预测未来{len(pred_prices)}天平均涨跌幅为{avg_change:.2f}%",
                    f"略低于当前价格{current_price}元，提高成功率",
                    f"最大承受亏损比例约{round(abs((stop_loss/entry_price - 1) * 100), 2)}%",
                    f"对应AI预测最高价{max_price}元的80%目标",
                    f"风险收益比约为1:{round(abs((target_price-entry_price)/(entry_price-stop_loss)), 1)}",
                    f"波动率{round(np.std(pred_changes), 2)}%",
                    confidence_reason,
                    "市场可能存在不可预测的系统性风险"
                ]
            })
            
            # 显示交易建议
            self.advice_result.setColumnCount(len(advice_df.columns))
            self.advice_result.setHorizontalHeaderLabels(advice_df.columns)
            
            self.advice_result.setRowCount(len(advice_df))
            for i, (_, row) in enumerate(advice_df.iterrows()):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    # 设置建议的颜色
                    if i == 0 and j == 1:  # 操作建议
                        if "买入" in value:
                            item.setForeground(Qt.red)
                        elif "卖出" in value or "减仓" in value:
                            item.setForeground(Qt.green)
                    self.advice_result.setItem(i, j, item)
            
            # 调整列宽
            self.advice_result.resizeColumnsToContents()
            # 调整行高
            for i in range(len(advice_df)):
                self.advice_result.setRowHeight(i, 40)
            
            # 更新状态栏
            self.statusBar().showMessage(f"{stock_name}({stock_code}) AI分析完成，建议: {suggestion}，预期收益: {round(abs((target_price/entry_price - 1) * 100), 2)}%")
        
        except Exception as e:
            self.logger.error(f"生成交易建议出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成交易建议出错: {str(e)}")

    def _on_custom_advice_clicked(self):
        """显示自定义交易建议"""
        try:
            from data.custom_stock_data import get_all_custom_advice
            
            # 获取所有自定义交易建议
            advice_df = get_all_custom_advice()
            
            if not advice_df.empty:
                # 显示交易建议
                self._display_results(advice_df)
                self.statusBar().showMessage("已显示自定义交易建议")
            else:
                QMessageBox.information(self, "提示", "没有找到自定义交易建议数据")
                self.statusBar().showMessage("没有找到自定义交易建议数据")
        except ImportError:
            QMessageBox.warning(self, "提示", "未找到自定义交易建议模块")
            self.statusBar().showMessage("未找到自定义交易建议模块")
        except Exception as e:
            self.logger.error(f"显示自定义交易建议时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示自定义交易建议时出错: {str(e)}")
            self.statusBar().showMessage("显示自定义交易建议时出错，请查看日志")

    def _generate_ai_investment_advice(self, pred_result, price_df):
        """根据AI预测生成投资建议"""
        # 获取当前价格
        current_price = price_df["close"].iloc[-1]
        
        # 分析预测趋势
        last_pred = pred_result[-1]
        pred_change = (last_pred / current_price - 1) * 100
        
        # 根据预测涨跌幅给出建议
        if pred_change > 10:
            return "强烈推荐买入，预期有显著上涨空间"
        elif pred_change > 5:
            return "建议买入，预期有较好上涨空间"
        elif pred_change > 2:
            return "可以考虑小仓位买入，预期有小幅上涨"
        elif pred_change > -2:
            return "建议观望，预期价格变化不大"
        elif pred_change > -5:
            return "可以考虑减仓，预期有小幅下跌风险"
        else:
            return "建议规避，预期有较大下跌风险"
            
    def _on_hot_industries_clicked(self):
        """处理热门行业分析按钮点击事件"""
        try:
            # 显示进度对话框
            progress = QProgressDialog("正在分析热门行业数据...", "取消", 0, 100, self)
            progress.setWindowTitle("热门行业分析")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            progress.show()
            
            QApplication.processEvents()
            
            self.logger.info("热门行业分析：开始获取数据")
            
            # 更新进度
            progress.setValue(20)
            
            # 获取热门行业数据
            industry_data = self.data_fetcher.get_hot_industries_data()
            
            # 更新进度
            progress.setValue(80)
            
            if progress.wasCanceled():
                self.logger.info("用户取消了热门行业分析")
                return
                
            # 检查是否成功获取数据
            if not industry_data:
                QMessageBox.critical(self, "错误", "无法获取热门行业数据，请稍后再试")
                return
                
            # 更新进度
            progress.setValue(90)
            
            # 显示热门行业分析结果
            self._display_hot_industries(industry_data)
            
            # 完成进度
            progress.setValue(100)
            
        except Exception as e:
            self.logger.error(f"热门行业分析时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"热门行业分析出错: {str(e)}")
    
    def _display_hot_industries(self, industry_data):
        """显示热门行业分析结果"""
        try:
            # 创建对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("热门行业分析")
            dialog.resize(900, 700)
            
            # 主布局
            main_layout = QVBoxLayout()
            
            # 添加标题
            title_label = QLabel("<h2>热门行业分析报告</h2>")
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)
            
            # 添加时间信息
            time_label = QLabel(f"<p>分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            time_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(time_label)
            
            # 添加市场概况部分
            market_overview = self._generate_market_overview()
            main_layout.addWidget(market_overview)
            
            # 创建Tab控件
            tab_widget = QTabWidget()
            
            # Tab1: 行业排行
            ranking_tab = QWidget()
            ranking_layout = QVBoxLayout(ranking_tab)
            
            # 添加排行说明
            desc_label = QLabel("<h3>热门行业综合评分排行</h3>")
            desc_label.setAlignment(Qt.AlignCenter)
            ranking_layout.addWidget(desc_label)
            
            # 排行榜说明
            explanation = QLabel("综合评分基于动量、成交量、上涨比例、相对强度等因素，得分越高表明行业表现越强势")
            explanation.setWordWrap(True)
            explanation.setAlignment(Qt.AlignCenter)
            explanation.setStyleSheet("color: #666;")
            ranking_layout.addWidget(explanation)
            
            # 创建行业排行表格
            current_table = QTableWidget()
            current_table.setColumnCount(8)
            current_table.setHorizontalHeaderLabels(["行业名称", "涨跌幅", "动量得分", "成交量得分", "上涨比例", "相对强度", "综合评分", "龙头股"])
            
            # 添加数据行
            current_table.setRowCount(len(industry_data))
            
            for row, industry in enumerate(industry_data):
                # 获取数据
                name = industry['name']
                change = industry.get('change', 0)
                momentum = industry.get('momentum_score', 0)
                volume = industry.get('volume_score', 0)
                up_ratio = industry.get('up_ratio', 0)
                relative_strength = industry.get('relative_strength', 0)
                score = industry.get('composite_score', 0)
                leading = industry.get('leading_stock', '')
                
                # 行业名称
                name_item = QTableWidgetItem(name)
                name_item.setToolTip("双击查看行业详情")
                name_item.setFont(QFont("", -1, QFont.Bold))
                current_table.setItem(row, 0, name_item)
                
                # 涨跌幅
                change_item = QTableWidgetItem(f"{change:.2f}%" if change else "0.00%")
                if change > 0:
                    change_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif change < 0:
                    change_item.setForeground(QBrush(QColor(0, 255, 0)))
                current_table.setItem(row, 1, change_item)
                
                # 动量得分
                momentum_item = QTableWidgetItem(f"{momentum:.2f}" if momentum else "0.00")
                if momentum > 5:
                    momentum_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif momentum < -3:
                    momentum_item.setForeground(QBrush(QColor(0, 255, 0)))
                current_table.setItem(row, 2, momentum_item)
                
                # 成交量得分
                volume_item = QTableWidgetItem(f"{volume:.2f}" if volume else "0.00")
                if volume > 5:
                    volume_item.setForeground(QBrush(QColor(255, 0, 0)))
                current_table.setItem(row, 3, volume_item)
                
                # 上涨比例
                up_ratio_item = QTableWidgetItem(f"{up_ratio:.2f}%" if up_ratio else "0.00%")
                if up_ratio > 60:
                    up_ratio_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif up_ratio < 40:
                    up_ratio_item.setForeground(QBrush(QColor(0, 255, 0)))
                current_table.setItem(row, 4, up_ratio_item)
                
                # 相对强度
                rel_strength_item = QTableWidgetItem(f"{relative_strength:.2f}" if relative_strength else "0.00")
                if relative_strength > 1:
                    rel_strength_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif relative_strength < -1:
                    rel_strength_item.setForeground(QBrush(QColor(0, 255, 0)))
                current_table.setItem(row, 5, rel_strength_item)
                
                # 综合评分
                score_item = QTableWidgetItem(f"{score:.2f}" if score else "0.00")
                if score > 7:
                    score_item.setForeground(QBrush(QColor(255, 0, 0)))
                    score_item.setFont(QFont("", -1, QFont.Bold))
                elif score > 5:
                    score_item.setForeground(QBrush(QColor(255, 165, 0)))
                current_table.setItem(row, 6, score_item)
                
                # 龙头股
                leading_item = QTableWidgetItem(leading)
                leading_item.setToolTip(f"{leading} - {name}行业龙头股")
                current_table.setItem(row, 7, leading_item)
            
            # 设置表格属性
            current_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 不可编辑
            current_table.setSelectionBehavior(QTableWidget.SelectRows)  # 整行选择
            current_table.setAlternatingRowColors(True)  # 交替行颜色
            current_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应
            
            # 添加双击事件，选择行业显示行业股票
            current_table.cellDoubleClicked.connect(lambda row, col: self._show_industry_stocks(industry_data[row]['name']))
            
            # 添加说明文本
            note_label = QLabel("双击行业可查看该行业的股票详情，包含量比和买入/卖出/止损价等交易参考")
            note_label.setStyleSheet("color: blue;")
            
            # 添加表格和说明到布局
            ranking_layout.addWidget(current_table)
            ranking_layout.addWidget(note_label)
            
            # 添加投资建议区域
            recommendation_group = QGroupBox("热门行业投资建议")
            recommendation_layout = QVBoxLayout(recommendation_group)
            
            # 获取前3个行业作为重点推荐
            top_industries = industry_data[:min(3, len(industry_data))]
            
            for i, industry in enumerate(top_industries):
                rec_text = self._generate_industry_recommendation(industry, i+1)
                rec_label = QLabel(rec_text)
                rec_label.setWordWrap(True)
                rec_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 5px; margin: 5px;")
                recommendation_layout.addWidget(rec_label)
                
                # 添加一个小间隔
                if i < len(top_industries) - 1:
                    spacer = QLabel("")
                    spacer.setFixedHeight(10)
                    recommendation_layout.addWidget(spacer)
            
            ranking_layout.addWidget(recommendation_group)
            
            # 添加行业比较图表
            chart_tab = QWidget()
            chart_layout = QVBoxLayout(chart_tab)
            
            # 创建行业比较图表标题
            chart_title = QLabel("<h3>行业综合评分对比</h3>")
            chart_title.setAlignment(Qt.AlignCenter)
            chart_layout.addWidget(chart_title)
            
            # 图表区域
            chart_widget = QWidget()
            chart_widget.setFixedHeight(350)
            chart_widget.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
            
            # 创建图表布局
            chart_inner_layout = QHBoxLayout(chart_widget)
            chart_inner_layout.setContentsMargins(10, 20, 10, 10)
            
            # 获取前10个行业数据用于显示
            top_industries = sorted(industry_data, key=lambda x: x['composite_score'], reverse=True)[:10]
            
            # 创建图形布局
            bar_widget = QWidget()
            bar_layout = QVBoxLayout(bar_widget)
            bar_layout.setSpacing(5)
            
            # 计算最高分值用于归一化
            max_score = max([ind['composite_score'] for ind in top_industries]) if top_industries else 1
            
            # 为每个行业创建条形图
            for industry in top_industries:
                # 行业横条布局
                industry_row = QWidget()
                row_layout = QHBoxLayout(industry_row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                
                # 行业名称
                name_label = QLabel(industry['name'])
                name_label.setFixedWidth(100)
                name_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                # 行业评分条
                score_bar = QProgressBar()
                score_bar.setFixedHeight(25)
                score_bar.setTextVisible(True)
                
                # 设置条形样式
                if industry['composite_score'] > 7:
                    score_bar.setStyleSheet("QProgressBar {border: 1px solid gray; border-radius: 3px; text-align: center;} "
                                         "QProgressBar::chunk {background-color: #ff4d4d;}")
                elif industry['composite_score'] > 5:
                    score_bar.setStyleSheet("QProgressBar {border: 1px solid gray; border-radius: 3px; text-align: center;} "
                                         "QProgressBar::chunk {background-color: #ff9933;}")
                else:
                    score_bar.setStyleSheet("QProgressBar {border: 1px solid gray; border-radius: 3px; text-align: center;} "
                                         "QProgressBar::chunk {background-color: #4da6ff;}")
                
                # 设置进度值（归一化为0-100）
                normalized_value = int((industry['composite_score'] / max_score) * 100)
                score_bar.setValue(normalized_value)
                score_bar.setFormat(f"{industry['composite_score']:.2f} 分")
                
                # 添加到行
                row_layout.addWidget(name_label)
                row_layout.addWidget(score_bar)
                
                # 添加到布局
                bar_layout.addWidget(industry_row)
            
            # 添加条形图到图表区域
            chart_inner_layout.addWidget(bar_widget)
            
            # 添加图表到布局
            chart_layout.addWidget(chart_widget)
            
            # 添加图表说明
            chart_desc = QLabel("行业评分是基于多种因素综合计算的结果，包括行业涨跌幅、成分股表现、资金流向等。分数越高表示行业表现越强势。")
            chart_desc.setWordWrap(True)
            chart_desc.setStyleSheet("color: #666; margin-top: 10px;")
            chart_layout.addWidget(chart_desc)
            
            # 未来趋势预测Tab
            trend_tab = QWidget()
            trend_layout = QVBoxLayout(trend_tab)
            
            # 添加未来趋势说明
            trend_title = QLabel("<h3>行业未来趋势预测</h3>")
            trend_title.setAlignment(Qt.AlignCenter)
            trend_layout.addWidget(trend_title)
            
            # 获取未来热门行业预测数据
            try:
                future_industries = self.data_fetcher.predict_future_hot_industries()
                
                if future_industries:
                    # 创建未来趋势表格
                    trend_table = QTableWidget()
                    trend_table.setColumnCount(6)
                    trend_table.setHorizontalHeaderLabels(["行业名称", "当前评分", "预测评分", "预测涨跌", "关注度", "投资建议"])
                    
                    # 添加数据行
                    trend_table.setRowCount(len(future_industries))
                    
                    for row, industry in enumerate(future_industries):
                        # 行业名称
                        name_item = QTableWidgetItem(industry['name'])
                        name_item.setFont(QFont("", -1, QFont.Bold))
                        trend_table.setItem(row, 0, name_item)
                        
                        # 当前评分
                        current_score = industry.get('composite_score', 0)
                        current_item = QTableWidgetItem(f"{current_score:.2f}")
                        trend_table.setItem(row, 1, current_item)
                        
                        # 预测评分
                        future_score = industry.get('future_score', 0)
                        future_item = QTableWidgetItem(f"{future_score:.2f}")
                        if future_score > 7:
                            future_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif future_score > 5:
                            future_item.setForeground(QBrush(QColor(255, 165, 0)))
                        trend_table.setItem(row, 2, future_item)
                        
                        # 预测涨跌
                        pred_change = industry.get('pred_change', 0)
                        change_item = QTableWidgetItem(f"{pred_change:+.2f}%")
                        if pred_change > 0:
                            change_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif pred_change < 0:
                            change_item.setForeground(QBrush(QColor(0, 255, 0)))
                        trend_table.setItem(row, 3, change_item)
                        
                        # 关注度变化
                        focus = industry.get('focus_change', '')
                        focus_item = QTableWidgetItem(focus)
                        if "上升" in focus:
                            focus_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif "下降" in focus:
                            focus_item.setForeground(QBrush(QColor(0, 255, 0)))
                        trend_table.setItem(row, 4, focus_item)
                        
                        # 投资建议
                        advice = industry.get('investment_advice', '')
                        advice_item = QTableWidgetItem(advice)
                        if "推荐" in advice or "买入" in advice:
                            advice_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif "观望" in advice:
                            advice_item.setForeground(QBrush(QColor(100, 100, 100)))
                        elif "回避" in advice:
                            advice_item.setForeground(QBrush(QColor(0, 255, 0)))
                        trend_table.setItem(row, 5, advice_item)
                    
                    # 设置表格属性
                    trend_table.setEditTriggers(QTableWidget.NoEditTriggers)
                    trend_table.setSelectionBehavior(QTableWidget.SelectRows)
                    trend_table.setAlternatingRowColors(True)
                    trend_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                    
                    # 添加双击事件，查看行业股票
                    trend_table.cellDoubleClicked.connect(lambda row, col: self._show_industry_stocks(future_industries[row]['name']))
                    
                    trend_layout.addWidget(trend_table)
                    
                    # 添加说明
                    trend_note = QLabel("预测基于多因子模型，考虑了动量持续性、资金流向、市场环境等因素，仅供参考。双击行业可查看详情。")
                    trend_note.setWordWrap(True)
                    trend_note.setStyleSheet("color: #666; margin-top: 10px;")
                    trend_layout.addWidget(trend_note)
                    
                else:
                    # 无预测数据时显示提示
                    no_data = QLabel("暂无行业趋势预测数据")
                    no_data.setAlignment(Qt.AlignCenter)
                    trend_layout.addWidget(no_data)
            except Exception as e:
                self.logger.error(f"获取行业趋势预测失败: {str(e)}")
                error_label = QLabel(f"获取行业趋势预测失败，请稍后再试")
                error_label.setAlignment(Qt.AlignCenter)
                error_label.setStyleSheet("color: red;")
                trend_layout.addWidget(error_label)
            
            # 创建热门股票Tab
            hot_stocks_tab = QWidget()
            hot_stocks_layout = QVBoxLayout(hot_stocks_tab)
            
            # 添加热门股票说明
            hot_stocks_title = QLabel("<h3>热门行业推荐股票</h3>")
            hot_stocks_title.setAlignment(Qt.AlignCenter)
            hot_stocks_layout.addWidget(hot_stocks_title)
            
            # 获取前3个热门行业
            top_3_industries = industry_data[:min(3, len(industry_data))]
            
            # 存储所有推荐股票
            all_recommended_stocks = []
            
            # 获取每个热门行业的推荐股票
            for industry in top_3_industries:
                try:
                    industry_name = industry['name']
                    industry_stocks = self.data_fetcher.get_industry_stocks(industry_name)
                    
                    if industry_stocks is not None and not industry_stocks.empty:
                        # 添加技术评分和推荐
                        industry_stocks = self.data_fetcher._add_stock_scores(industry_stocks)
                        industry_stocks = self.data_fetcher._add_trading_recommendations(industry_stocks)
                        
                        # 获取评分最高的前5只股票
                        top_stocks = industry_stocks.sort_values(by='composite_score', ascending=False).head(3)
                        
                        # 添加行业名称列
                        top_stocks['industry'] = industry_name
                        
                        # 添加到所有推荐股票列表
                        all_recommended_stocks.append(top_stocks)
                except Exception as e:
                    self.logger.error(f"获取{industry['name']}推荐股票失败: {str(e)}")
                    continue
            
            # 如果有推荐股票
            if all_recommended_stocks:
                # 合并所有推荐股票
                try:
                    all_stocks_df = pd.concat(all_recommended_stocks)
                    
                    # 创建热门股票表格
                    hot_stocks_table = QTableWidget()
                    hot_stocks_table.setColumnCount(8)
                    hot_stocks_table.setHorizontalHeaderLabels(["行业", "股票代码", "股票名称", "最新价", "涨跌幅", "综合评分", "推荐理由", "投资建议"])
                    hot_stocks_table.setRowCount(len(all_stocks_df))
                    
                    for row, (_, stock) in enumerate(all_stocks_df.iterrows()):
                        # 行业
                        industry_item = QTableWidgetItem(stock.get('industry', ''))
                        hot_stocks_table.setItem(row, 0, industry_item)
                        
                        # 股票代码
                        code_item = QTableWidgetItem(stock.get('ts_code', ''))
                        hot_stocks_table.setItem(row, 1, code_item)
                        
                        # 股票名称
                        name_item = QTableWidgetItem(stock.get('name', ''))
                        name_item.setFont(QFont("", -1, QFont.Bold))
                        hot_stocks_table.setItem(row, 2, name_item)
                        
                        # 最新价
                        price = stock.get('current', stock.get('close', 0))
                        price_item = QTableWidgetItem(f"{price:.2f}")
                        hot_stocks_table.setItem(row, 3, price_item)
                        
                        # 涨跌幅
                        change = stock.get('change', stock.get('pct_chg', 0))
                        change_item = QTableWidgetItem(f"{change:.2f}%")
                        if change > 0:
                            change_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif change < 0:
                            change_item.setForeground(QBrush(QColor(0, 255, 0)))
                        hot_stocks_table.setItem(row, 4, change_item)
                        
                        # 综合评分
                        score = stock.get('composite_score', 0)
                        score_item = QTableWidgetItem(f"{score:.1f}")
                        if score > 80:
                            score_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif score > 60:
                            score_item.setForeground(QBrush(QColor(255, 165, 0)))
                        hot_stocks_table.setItem(row, 5, score_item)
                        
                        # 推荐理由（基于评分构建）
                        reason = ""
                        if stock.get('technical_score', 0) > 70:
                            reason += "技术面强势"
                        if stock.get('value_score', 0) > 70:
                            reason += ", 价值评分高"
                        if stock.get('momentum_score', 0) > 70:
                            reason += ", 动能强劲"
                        if "主力资金" in stock.get('advice_detail', ''):
                            reason += ", 主力资金流入"
                        if not reason:
                            reason = "综合评分高"
                        
                        reason_item = QTableWidgetItem(reason.lstrip(", "))
                        hot_stocks_table.setItem(row, 6, reason_item)
                        
                        # 投资建议
                        advice = stock.get('trading_advice', '持有观望')
                        advice_item = QTableWidgetItem(advice)
                        if "买入" in advice:
                            advice_item.setForeground(QBrush(QColor(255, 0, 0)))
                        elif "卖出" in advice or "减持" in advice:
                            advice_item.setForeground(QBrush(QColor(0, 255, 0)))
                        hot_stocks_table.setItem(row, 7, advice_item)
                    
                    # 设置表格属性
                    hot_stocks_table.setEditTriggers(QTableWidget.NoEditTriggers)
                    hot_stocks_table.setSelectionBehavior(QTableWidget.SelectRows)
                    hot_stocks_table.setAlternatingRowColors(True)
                    hot_stocks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                    
                    # 添加双击事件查看股票详情
                    hot_stocks_table.cellDoubleClicked.connect(
                        lambda row, col: self._show_stock_details(
                            hot_stocks_table.item(row, 1).text(), 
                            hot_stocks_table.item(row, 2).text()
                        )
                    )
                    
                    hot_stocks_layout.addWidget(hot_stocks_table)
                    
                    # 添加说明
                    stock_note = QLabel("此表显示了热门行业中表现最佳的股票。双击股票可查看详细分析。")
                    stock_note.setStyleSheet("color: blue;")
                    hot_stocks_layout.addWidget(stock_note)
                    
                except Exception as concat_error:
                    self.logger.error(f"合并推荐股票失败: {str(concat_error)}")
                    error_label = QLabel("获取推荐股票数据失败，请稍后再试")
                    error_label.setAlignment(Qt.AlignCenter)
                    error_label.setStyleSheet("color: red;")
                    hot_stocks_layout.addWidget(error_label)
            else:
                no_stocks = QLabel("暂无热门股票推荐数据")
                no_stocks.setAlignment(Qt.AlignCenter)
                hot_stocks_layout.addWidget(no_stocks)
                
            # 添加各个Tab
            tab_widget.addTab(ranking_tab, "热门行业排行")
            tab_widget.addTab(chart_tab, "行业评分对比")
            tab_widget.addTab(trend_tab, "未来趋势预测")
            tab_widget.addTab(hot_stocks_tab, "热门推荐股票")
            
            main_layout.addWidget(tab_widget)
            
            # 添加底部按钮
            button_layout = QHBoxLayout()
            
            # 导出数据按钮
            export_btn = QPushButton("导出分析报告")
            export_btn.clicked.connect(lambda: self._export_industry_data(industry_data))
            button_layout.addWidget(export_btn)
            
            # 添加刷新按钮
            refresh_btn = QPushButton("刷新数据")
            refresh_btn.clicked.connect(self._on_hot_industries_clicked)
            button_layout.addWidget(refresh_btn)
            
            # 添加说明
            tip_label = QLabel("提示: 双击行业或股票可查看详细信息")
            tip_label.setStyleSheet("color: blue;")
            button_layout.addWidget(tip_label)
            
            main_layout.addLayout(button_layout)
            
            dialog.setLayout(main_layout)
            dialog.exec_()
            
        except Exception as e:
            self.logger.error(f"显示热门行业分析结果时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示热门行业分析结果时出错: {str(e)}")
    
    def _export_industry_data(self, industry_data):
        """导出行业数据到CSV文件"""
        try:
            # 选择保存路径
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "导出行业数据", 
                f"热门行业分析_{datetime.now().strftime('%Y%m%d')}.csv",
                "CSV Files (*.csv)", options=options)
            
            if not file_name:
                return
                
            # 创建DataFrame
            df = pd.DataFrame(industry_data)
            
            # 保存到CSV
            df.to_csv(file_name, index=False, encoding='utf-8-sig')
            
            # 显示成功消息
            QMessageBox.information(self, "导出成功", f"行业数据已成功导出到 {file_name}")
            
        except Exception as e:
            self.logger.error(f"导出行业数据时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"导出行业数据时出错: {str(e)}")
            
            # 连接应用按钮
            apply_button.clicked.connect(apply_filters)
            
            # 连接回车键搜索
            search_input.returnPressed.connect(apply_filters)
            
            # 添加表格到布局
            stocks_layout.addWidget(table)
            
            # 添加注释信息
            note_label = QLabel("双击股票行可查看详细信息和K线图")
            note_label.setStyleSheet("color: blue;")
            stocks_layout.addWidget(note_label)
            
            # Tab2: 评分分析
            if 'composite_score' in stocks.columns:
                score_tab = QWidget()
                score_layout = QVBoxLayout(score_tab)
                
                # 评分说明
                score_info = QLabel("<h4>股票评分系统说明</h4>"
                                   "<p>系统从价值、技术和动量三个维度对股票进行综合评分，满分100分。评分越高表示投资价值越大。</p>"
                                   "<ul>"
                                   "<li><b>价值评分</b>：基于PE、PB等基本面指标，评估股票内在价值</li>"
                                   "<li><b>技术评分</b>：基于均线、MACD、KDJ、RSI等技术指标，评估股票走势</li>"
                                   "<li><b>动量评分</b>：基于涨跌幅、换手率、资金流向等因素，评估股票动量</li>"
                                   "</ul>")
                score_info.setWordWrap(True)
                score_layout.addWidget(score_info)
                
                # 创建评分分布统计
                dist_title = QLabel("<h4>评分分布统计</h4>")
                score_layout.addWidget(dist_title)
                
                # 评分分布图表
                dist_widget = QWidget()
                dist_widget.setFixedHeight(200)
                dist_widget.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd;")
                dist_layout = QHBoxLayout(dist_widget)
                
                # 计算评分分布
                score_ranges = [
                    (0, 20, "极差"),
                    (20, 40, "较差"),
                    (40, 60, "一般"),
                    (60, 80, "较好"),
                    (80, 100, "优秀")
                ]
                
                # 计算各分数段的股票数量
                score_counts = []
                score_percents = []
                
                for low, high, _ in score_ranges:
                    count = len(stocks[(stocks['composite_score'] >= low) & (stocks['composite_score'] < high)])
                    score_counts.append(count)
                    score_percents.append(count / len(stocks) * 100 if len(stocks) > 0 else 0)
                
                # 创建分布图表（简化版）
                bar_chart = QWidget()
                bar_layout = QVBoxLayout(bar_chart)
                
                # 为每个分数段创建条形
                for i, (low, high, label) in enumerate(score_ranges):
                    # 创建行布局
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(5, 2, 5, 2)
                    
                    # 标签
                    range_label = QLabel(f"{label} ({low}-{high}分)")
                    range_label.setFixedWidth(100)
                    range_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    # 进度条
                    bar = QProgressBar()
                    bar.setFixedHeight(20)
                    bar.setRange(0, 100)
                    bar.setValue(int(score_percents[i]))
                    
                    # 根据分数段设置不同的颜色
                    colors = ["#FF6666", "#FFCC66", "#66CCFF", "#66CC99", "#CC66FF"]
                    bar.setStyleSheet(f"QProgressBar {{ background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 2px; }} "
                                     f"QProgressBar::chunk {{ background-color: {colors[i]}; }}")
                    
                    # 数量标签
                    count_label = QLabel(f"{score_counts[i]}只 ({score_percents[i]:.1f}%)")
                    count_label.setFixedWidth(100)
                    
                    # 添加到行布局
                    row_layout.addWidget(range_label)
                    row_layout.addWidget(bar)
                    row_layout.addWidget(count_label)
                    
                    # 添加到条形图布局
                    bar_layout.addWidget(row_widget)
                
                dist_layout.addWidget(bar_chart)
                score_layout.addWidget(dist_widget)
                
                # 添加评分排名前10的股票列表
                top_title = QLabel("<h4>评分排名前10的股票</h4>")
                score_layout.addWidget(top_title)
                
                # 创建表格
                top_table = QTableWidget()
                top_table.setColumnCount(5)
                top_table.setHorizontalHeaderLabels(["排名", "股票代码", "股票名称", "综合评分", "交易建议"])
                
                # 获取评分前10的股票
                top_stocks = stocks.sort_values(by='composite_score', ascending=False).head(10)
                top_table.setRowCount(len(top_stocks))
                
                for i, (_, stock) in enumerate(top_stocks.iterrows()):
                    # 排名
                    rank_item = QTableWidgetItem(f"{i+1}")
                    top_table.setItem(i, 0, rank_item)
                    
                    # 股票代码
                    code_item = QTableWidgetItem(stock['ts_code'])
                    top_table.setItem(i, 1, code_item)
                    
                    # 股票名称
                    name_item = QTableWidgetItem(stock['name'])
                    top_table.setItem(i, 2, name_item)
                    
                    # 综合评分
                    score = stock['composite_score']
                    score_item = QTableWidgetItem(f"{score:.0f}")
                    if score >= 80:
                        score_item.setForeground(QBrush(QColor(255, 0, 0)))
                    elif score >= 60:
                        score_item.setForeground(QBrush(QColor(255, 165, 0)))
                    top_table.setItem(i, 3, score_item)
                    
                    # 交易建议
                    advice = stock['trading_advice']
                    advice_item = QTableWidgetItem(advice)
                    if "强烈推荐" in advice:
                        advice_item.setForeground(QBrush(QColor(255, 0, 0)))
                        advice_item.setFont(QFont("", -1, QFont.Bold))
                    elif "建议买入" in advice:
                        advice_item.setForeground(QBrush(QColor(255, 90, 0)))
                    top_table.setItem(i, 4, advice_item)
                
                # 设置表格属性
                top_table.setEditTriggers(QTableWidget.NoEditTriggers)
                top_table.setAlternatingRowColors(True)
                top_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                
                # 添加双击事件
                top_table.cellDoubleClicked.connect(lambda row, col: self._show_stock_details(
                    top_stocks.iloc[row]['ts_code'], top_stocks.iloc[row]['name']))
                
                score_layout.addWidget(top_table)
                
                # 添加交易建议说明
                advice_info = QLabel("<h4>交易建议说明</h4>"
                                    "<ul>"
                                    "<li><span style='color:red;'><b>强烈推荐买入</b></span>：评分≥85分，各项指标优异，短中期有较大上涨空间</li>"
                                    "<li><span style='color:#FF5722;'>建议买入</span>：评分70-84分，基本面良好，技术形态较好</li>"
                                    "<li><span style='color:#2196F3;'>适量买入</span>：评分55-69分，有一定投资价值，可小仓位介入</li>"
                                    "<li><span style='color:#4CAF50;'>持有观望</span>：评分45-54分，投资价值一般，建议观望为主</li>"
                                    "<li><span style='color:#9E9E9E;'>减持/卖出</span>：评分＜45分，风险较大或下跌趋势明显</li>"
                                    "</ul>")
                advice_info.setWordWrap(True)
                score_layout.addWidget(advice_info)
                
                # 添加免责声明
                disclaimer = QLabel("<p style='color:gray; font-size:12px;'>免责声明：评分系统仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>")
                score_layout.addWidget(disclaimer)
                
                # 添加到Tab
                tab_widget.addTab(score_tab, "评分分析")
            
            # 添加Tab控件到主布局
            tab_widget.addTab(stocks_tab, "股票列表")
            layout.addWidget(tab_widget)
            
            # 添加导出和关闭按钮
            button_layout = QHBoxLayout()
            
            # 导出按钮
            export_button = QPushButton("导出数据")
            export_button.clicked.connect(lambda: self._export_stocks_data(filtered_stocks, industry_name))
            button_layout.addWidget(export_button)
            
            # 添加自选按钮
            add_favorites_button = QPushButton("添加到自选")
            add_favorites_button.clicked.connect(lambda: self._add_to_favorites(filtered_stocks, table))
            button_layout.addWidget(add_favorites_button)
            
            stocks_layout.addLayout(buttons_layout)
            
            # 添加标签页
            tab_widget.addTab(stocks_tab, "行业成分股")
            
            # 添加标签页到布局
            layout.addWidget(tab_widget)
            
            # 设置对话框布局
            dialog.setLayout(layout)
            
            # 隐藏进度对话框
            progress.close()
            
            # 显示对话框
            dialog.exec_()
            
        except Exception as e:
            self.logger.error(f"显示行业股票列表时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示行业股票列表时出错: {str(e)}")
    
    def _export_stocks_data(self, stocks_df, industry_name):
        """导出股票数据到CSV文件"""
        try:
            # 选择保存路径
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "导出股票数据", 
                f"{industry_name}_股票列表_{datetime.now().strftime('%Y%m%d')}.csv",
                "CSV Files (*.csv)", options=options)
            
            if not file_name:
                return
                
            # 保存到CSV
            stocks_df.to_csv(file_name, index=False, encoding='utf-8-sig')
            
            # 显示成功消息
            QMessageBox.information(self, "导出成功", f"股票数据已成功导出到 {file_name}")
            
        except Exception as e:
            self.logger.error(f"导出股票数据时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"导出股票数据时出错: {str(e)}")
    
    def _add_to_favorites(self, stocks_df, table):
        """添加选中股票到自选"""
        try:
            selected_rows = table.selectionModel().selectedRows()
            
            if not selected_rows:
                QMessageBox.information(self, "提示", "请先选择要添加的股票")
                return
            
            selected_indices = [index.row() for index in selected_rows]
            selected_stocks = stocks_df.iloc[selected_indices]
            
            # 在这里实现添加到自选的逻辑
            # 可以将股票信息保存到配置文件或数据库中
            stock_codes = []
            for _, stock in selected_stocks.iterrows():
                stock_codes.append(f"{stock['ts_code']} - {stock['name']}")
            
            # 简单地显示添加的股票信息
            stocks_text = "\n".join(stock_codes[:10])
            if len(stock_codes) > 10:
                stocks_text += f"\n... 等 {len(stock_codes)} 只股票"
                
            QMessageBox.information(self, "添加成功", 
                                  f"已成功添加以下股票到自选：\n\n{stocks_text}")
            
        except Exception as e:
            self.logger.error(f"添加股票到自选时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"添加股票到自选时出错: {str(e)}")
    
    def _show_stock_details(self, ts_code, stock_name):
        """显示个股详情"""
        try:
            # 显示进度对话框
            progress = QProgressDialog(f"正在获取 {stock_name} 详细数据...", "取消", 0, 100, self)
            progress.setWindowTitle("股票详情")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            progress.show()
            
            QApplication.processEvents()
            
            # 获取股票K线数据
            k_data = self.data_fetcher.get_k_line_data(ts_code, None, None)
            
            progress.setValue(70)
            
            if progress.wasCanceled():
                return
                
            if k_data is None or k_data.empty:
                QMessageBox.information(self, "提示", f"未找到 {stock_name}({ts_code}) 的历史数据")
                return
                
            progress.setValue(90)
            
            # 创建股票详情对话框
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{stock_name}({ts_code}) - 股票详情")
            dialog.resize(1000, 700)
            
            # 创建主布局
            main_layout = QVBoxLayout()
            
            # 添加基本信息区域
            info_layout = QHBoxLayout()
            
            # 左侧基础信息
            basic_group = QGroupBox("基本信息")
            basic_layout = QVBoxLayout()
            
            # 获取最新交易日数据
            latest_row = k_data.iloc[-1] if not k_data.empty else None
            prev_row = k_data.iloc[-2] if len(k_data) > 1 else None
            
            # 处理基本信息字段
            if latest_row is not None:
                # 价格信息
                close_price = latest_row['close'] if 'close' in latest_row else 0
                open_price = latest_row['open'] if 'open' in latest_row else 0
                high_price = latest_row['high'] if 'high' in latest_row else 0
                low_price = latest_row['low'] if 'low' in latest_row else 0
                
                # 计算涨跌幅
                change_pct = latest_row['pct_chg'] if 'pct_chg' in latest_row else 0
                
                # 成交量成交额
                volume = latest_row['vol'] if 'vol' in latest_row else 0
                amount = latest_row['amount'] if 'amount' in latest_row else 0
                
                # 组织显示信息
                price_info = QLabel(f"<p><b>最新价：</b><span style='color:{'red' if change_pct >= 0 else 'green'};font-size:16px;'>{close_price:.2f}</span></p>")
                basic_layout.addWidget(price_info)
                
                change_info = QLabel(f"<p><b>涨跌幅：</b><span style='color:{'red' if change_pct >= 0 else 'green'};'>{change_pct:.2f}%</span></p>")
                basic_layout.addWidget(change_info)
                
                trade_date = latest_row['trade_date'] if 'trade_date' in latest_row else ''
                date_info = QLabel(f"<p><b>交易日期：</b>{trade_date}</p>")
                basic_layout.addWidget(date_info)
                
                ohlc_info = QLabel(f"<p><b>开盘价：</b>{open_price:.2f} | <b>最高价：</b>{high_price:.2f} | <b>最低价：</b>{low_price:.2f}</p>")
                basic_layout.addWidget(ohlc_info)
                
                vol_info = QLabel(f"<p><b>成交量：</b>{volume/100:.0f}手 | <b>成交额：</b>{amount/10000:.2f}万元</p>")
                basic_layout.addWidget(vol_info)
                
                # 计算简单技术指标
                # 5日均线
                ma5 = k_data['close'].rolling(5).mean().iloc[-1] if len(k_data) >= 5 else None
                ma10 = k_data['close'].rolling(10).mean().iloc[-1] if len(k_data) >= 10 else None
                ma20 = k_data['close'].rolling(20).mean().iloc[-1] if len(k_data) >= 20 else None
                
                ma_info = QLabel("<p><b>均线：</b>")
                if ma5 is not None:
                    ma_info.setText(ma_info.text() + f"MA5: {ma5:.2f} ")
                if ma10 is not None:
                    ma_info.setText(ma_info.text() + f"| MA10: {ma10:.2f} ")
                if ma20 is not None:
                    ma_info.setText(ma_info.text() + f"| MA20: {ma20:.2f}")
                ma_info.setText(ma_info.text() + "</p>")
                basic_layout.addWidget(ma_info)
                
                # 添加价格位置信息
                if ma5 is not None and ma20 is not None:
                    if close_price > ma5 > ma20:
                        trend_info = QLabel("<p><b>趋势：</b><span style='color:red;'>上升趋势 ↗</span></p>")
                    elif close_price < ma5 < ma20:
                        trend_info = QLabel("<p><b>趋势：</b><span style='color:green;'>下降趋势 ↘</span></p>")
                    else:
                        trend_info = QLabel("<p><b>趋势：</b><span style='color:blue;'>震荡整理 ↔</span></p>")
                    basic_layout.addWidget(trend_info)
            
            basic_group.setLayout(basic_layout)
            info_layout.addWidget(basic_group)
            
            # 右侧评级信息
            rating_group = QGroupBox("投资评级")
            rating_layout = QVBoxLayout()
            
            # 添加评分和建议
            # 模拟生成综合评分 (实际应用中应从数据获取)
            technical_score = 65  # 示例得分
            value_score = 72
            momentum_score = 58
            composite_score = (technical_score * 0.3 + value_score * 0.4 + momentum_score * 0.3)
            
            # 评分显示
            score_info = QLabel(f"<p><b>综合评分：</b><span style='font-size:18px;font-weight:bold;color:{'red' if composite_score >= 70 else 'blue'};'>{composite_score:.0f}</span> / 100</p>")
            rating_layout.addWidget(score_info)
            
            # 细分评分
            sub_scores = QLabel(f"<p><b>价值评分：</b>{value_score} | <b>技术评分：</b>{technical_score} | <b>动量评分：</b>{momentum_score}</p>")
            rating_layout.addWidget(sub_scores)
            
            # 交易建议
            if composite_score >= 80:
                advice = "强烈推荐买入"
                advice_color = "red"
            elif composite_score >= 70:
                advice = "建议买入"
                advice_color = "#FF5722"  # 橙色
            elif composite_score >= 60:
                advice = "适量买入"
                advice_color = "#2196F3"  # 蓝色
            elif composite_score >= 45:
                advice = "持有观望"
                advice_color = "#4CAF50"  # 绿色
            else:
                advice = "建议卖出"
                advice_color = "#9E9E9E"  # 灰色
                
            advice_info = QLabel(f"<p><b>交易建议：</b><span style='font-size:16px;font-weight:bold;color:{advice_color};'>{advice}</span></p>")
            rating_layout.addWidget(advice_info)
            
            # 价格预期
            # 基于最近走势简单预测价格
            if latest_row is not None and prev_row is not None:
                momentum = (latest_row['close'] - prev_row['close']) / prev_row['close']
                pred_change = momentum * (5 + (random.random() * 5))  # 简化的预测模型
                pred_price = latest_row['close'] * (1 + pred_change)
                
                price_pred = QLabel(f"<p><b>目标价：</b><span style='color:{'red' if pred_change > 0 else 'green'};'>{pred_price:.2f}</span> (<span style='color:{'red' if pred_change > 0 else 'green'};'>{pred_change*100:.2f}%</span>)</p>")
                rating_layout.addWidget(price_pred)
                
                # 支撑与压力位
                support = latest_row['close'] * 0.95
                resistance = latest_row['close'] * 1.05
                
                support_resistance = QLabel(f"<p><b>支撑位：</b>{support:.2f} | <b>压力位：</b>{resistance:.2f}</p>")
                rating_layout.addWidget(support_resistance)
            
            # 添加评级分析建议
            advice_detail = QTextEdit()
            advice_detail.setReadOnly(True)
            advice_detail.setMaximumHeight(100)
            
            # 生成详细建议 (实际应用中应根据实际数据分析)
            detail_text = f"【{stock_name}】综合评分 {composite_score:.0f}，"
            
            if composite_score >= 70:
                detail_text += "各项指标较好，"
                if technical_score > 70:
                    detail_text += "技术面走强，K线形态良好，短期可能继续上涨；"
                if value_score > 70:
                    detail_text += "基本面扎实，估值合理；"
                if momentum_score > 70:
                    detail_text += "动量强劲，资金面支持良好；"
                detail_text += f"建议{advice}。"
            elif composite_score >= 45:
                detail_text += "表现一般，"
                if technical_score < 50:
                    detail_text += "技术指标偏弱，可能存在短期调整压力；"
                else:
                    detail_text += "技术面中性；"
                    
                if value_score < 50:
                    detail_text += "估值偏高；"
                else:
                    detail_text += "价值评估合理；"
                    
                detail_text += f"建议{advice}。"
            else:
                detail_text += "整体表现较弱，"
                if technical_score < 40:
                    detail_text += "技术形态走坏，下跌趋势明显；"
                if value_score < 40:
                    detail_text += "估值过高或基本面恶化；"
                if momentum_score < 40:
                    detail_text += "资金持续流出；"
                detail_text += f"建议{advice}。"
            
            advice_detail.setText(detail_text)
            rating_layout.addWidget(advice_detail)
            
            rating_group.setLayout(rating_layout)
            info_layout.addWidget(rating_group)
            
            main_layout.addLayout(info_layout)
            
            # 添加Tab控件，用于显示不同类型的信息
            tab_widget = QTabWidget()
            
            # Tab1: K线图
            chart_tab = QWidget()
            chart_layout = QVBoxLayout(chart_tab)
            
            # K线图区域
            chart_placeholder = QLabel("K线图区域 - 实际应用中应集成专业图表组件")
            chart_placeholder.setAlignment(Qt.AlignCenter)
            chart_placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd; padding: 50px;")
            chart_placeholder.setMinimumHeight(300)
            chart_layout.addWidget(chart_placeholder)
            
            tab_widget.addTab(chart_tab, "K线图")
            
            # Tab2: 交易数据
            data_tab = QWidget()
            data_layout = QVBoxLayout(data_tab)
            
            # 创建表格显示最近的交易数据
            data_table = QTableWidget()
            data_table.setColumnCount(7)
            data_table.setHorizontalHeaderLabels(["日期", "开盘价", "最高价", "最低价", "收盘价", "涨跌幅", "成交量(手)"])
            
            # 显示最近20条数据
            display_rows = min(20, len(k_data))
            data_table.setRowCount(display_rows)
            
            # 填充数据
            for i in range(display_rows):
                row_data = k_data.iloc[-(i+1)]  # 倒序显示，最新的在最上面
                
                # 日期
                date_item = QTableWidgetItem(str(row_data['trade_date']))
                data_table.setItem(i, 0, date_item)
                
                # 开盘价
                open_item = QTableWidgetItem(f"{row_data['open']:.2f}")
                data_table.setItem(i, 1, open_item)
                
                # 最高价
                high_item = QTableWidgetItem(f"{row_data['high']:.2f}")
                data_table.setItem(i, 2, high_item)
                
                # 最低价
                low_item = QTableWidgetItem(f"{row_data['low']:.2f}")
                data_table.setItem(i, 3, low_item)
                
                # 收盘价
                close_item = QTableWidgetItem(f"{row_data['close']:.2f}")
                data_table.setItem(i, 4, close_item)
                
                # 涨跌幅
                pct_chg = row_data['pct_chg'] if 'pct_chg' in row_data else 0
                pct_item = QTableWidgetItem(f"{pct_chg:.2f}%")
                if pct_chg > 0:
                    pct_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif pct_chg < 0:
                    pct_item.setForeground(QBrush(QColor(0, 255, 0)))
                data_table.setItem(i, 5, pct_item)
                
                # 成交量
                vol = row_data['vol'] if 'vol' in row_data else 0
                vol_item = QTableWidgetItem(f"{vol/100:.0f}")
                data_table.setItem(i, 6, vol_item)
            
            # 设置表格属性
            data_table.setEditTriggers(QTableWidget.NoEditTriggers)
            data_table.setAlternatingRowColors(True)
            data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            data_layout.addWidget(data_table)
            tab_widget.addTab(data_tab, "历史数据")
            
            # Tab3: 技术指标
            indicator_tab = QWidget()
            indicator_layout = QVBoxLayout(indicator_tab)
            
            # 添加一些技术指标的简要说明
            indicator_desc = QLabel("<h4>技术指标分析</h4>")
            indicator_layout.addWidget(indicator_desc)
            
            # 生成简单的技术指标表格
            indicator_table = QTableWidget()
            indicator_table.setColumnCount(4)
            indicator_table.setRowCount(6)
            indicator_table.setHorizontalHeaderLabels(["指标", "数值", "状态", "解读"])
            
            # 模拟一些指标数据 (实际应用中应从实际计算获取)
            indicators = [
                ("MA(5,10,20)", f"{ma5:.2f}, {ma10:.2f}, {ma20:.2f}" if all(x is not None for x in [ma5, ma10, ma20]) else "N/A", 
                 "多头排列" if ma5 > ma10 > ma20 else "空头排列" if ma5 < ma10 < ma20 else "交叉整理",
                 "短期均线高于长期均线，呈现上升趋势" if ma5 > ma10 > ma20 else "短期均线低于长期均线，呈现下降趋势" if ma5 < ma10 < ma20 else "均线交叉，趋势不明确"),
                
                ("MACD", "0.35", "金叉", "MACD金叉，买入信号"),
                
                ("KDJ", "75,80,85", "超买", "KDJ指标位于超买区域，注意回调风险"),
                
                ("RSI", "60", "中性偏强", "RSI处于中性偏强区域，上升动能较强"),
                
                ("BOLL", "上轨:25.4 中轨:24.2 下轨:23.0", "上轨附近", "股价运行在布林上轨附近，上涨动能强"),
                
                ("量比", "1.2", "温和放量", "成交量温和放大，市场活跃度增加")
            ]
            
            # 填充指标数据
            for i, (name, value, status, interpret) in enumerate(indicators):
                # 指标名称
                name_item = QTableWidgetItem(name)
                indicator_table.setItem(i, 0, name_item)
                
                # 数值
                value_item = QTableWidgetItem(value)
                indicator_table.setItem(i, 1, value_item)
                
                # 状态
                status_item = QTableWidgetItem(status)
                if "多头" in status or "金叉" in status or "偏强" in status:
                    status_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif "空头" in status or "死叉" in status or "偏弱" in status:
                    status_item.setForeground(QBrush(QColor(0, 255, 0)))
                indicator_table.setItem(i, 2, status_item)
                
                # 解读
                interpret_item = QTableWidgetItem(interpret)
                indicator_table.setItem(i, 3, interpret_item)
            
            # 设置表格属性
            indicator_table.setEditTriggers(QTableWidget.NoEditTriggers)
            indicator_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            indicator_table.verticalHeader().setVisible(False)
            
            indicator_layout.addWidget(indicator_table)
            
            # 添加免责声明
            disclaimer = QLabel("<p style='color:gray;font-size:12px;'>注：技术指标仅供参考，不构成投资建议。投资有风险，交易需谨慎。</p>")
            indicator_layout.addWidget(disclaimer)
            
            tab_widget.addTab(indicator_tab, "技术指标")
            
            # 添加Tab控件到主布局
            main_layout.addWidget(tab_widget)
            
            # 添加按钮区域
            button_layout = QHBoxLayout()
            
            # 添加自选按钮
            add_favorite_btn = QPushButton("添加到自选")
            button_layout.addWidget(add_favorite_btn)
            
            # 设置提醒按钮
            set_alert_btn = QPushButton("设置价格提醒")
            button_layout.addWidget(set_alert_btn)
            
            # 查看公司信息按钮
            company_info_btn = QPushButton("公司信息")
            button_layout.addWidget(company_info_btn)
            
            # 关闭按钮
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            
            main_layout.addLayout(button_layout)
            
            # 设置对话框布局
            dialog.setLayout(main_layout)
            
            progress.setValue(100)
            dialog.exec_()
            
        except Exception as e:
            self.logger.error(f"显示个股详情时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示个股详情时出错: {str(e)}")
    
    def _on_select_stock_clicked(self):
        """处理选股按钮点击事件"""
        self.logger.info("用户点击了选股按钮")
        # TODO: 实现选股功能
        QMessageBox.information(self, "选股", "选股功能将在后续版本中实现")
    
    def _on_analysis_clicked(self):
        """处理技术分析按钮点击事件"""
        self.logger.info("用户点击了技术分析按钮")
        # TODO: 实现技术分析功能
        QMessageBox.information(self, "技术分析", "技术分析功能将在后续版本中实现")
    
    def _on_prediction_clicked(self):
        """处理AI预测按钮点击事件"""
        self.logger.info("用户点击了AI预测按钮")
        # TODO: 实现AI预测功能
        QMessageBox.information(self, "AI预测", "AI预测功能将在后续版本中实现")
    
    def _on_settings_clicked(self):
        """处理设置按钮点击事件"""
        self.logger.info("用户点击了设置按钮")
        # TODO: 实现设置功能
        QMessageBox.information(self, "设置", "设置功能将在后续版本中实现")

    def _show_industry_stocks(self, industry_name):
        """显示行业内的股票"""
        try:
            self.logger.info(f"查看行业 {industry_name} 的股票")
            
            # 获取行业股票
            industry_stocks = self.data_fetcher.get_industry_stocks(industry_name)
            
            if industry_stocks is None or industry_stocks.empty:
                QMessageBox.information(self, "提示", f"未找到{industry_name}行业的股票数据")
                return
            
            # 处理股票数据，添加技术指标和评级
            if 'ts_code' in industry_stocks.columns:
                # 添加必要的列
                if 'close' not in industry_stocks.columns:
                    industry_stocks['close'] = 0
                if 'change' not in industry_stocks.columns:
                    industry_stocks['change'] = 0
                if 'vol' not in industry_stocks.columns:
                    industry_stocks['vol'] = 0
                if 'amount' not in industry_stocks.columns:
                    industry_stocks['amount'] = 0
                if 'pe' not in industry_stocks.columns:
                    industry_stocks['pe'] = 0
                if 'pb' not in industry_stocks.columns:
                    industry_stocks['pb'] = 0
                
                try:
                    # 添加技术评分和推荐
                    industry_stocks = self.data_fetcher._add_technical_indicators(industry_stocks)
                    industry_stocks = self.data_fetcher._add_money_flow(industry_stocks)
                    industry_stocks = self.data_fetcher._add_stock_scores(industry_stocks)
                    industry_stocks = self.data_fetcher._add_trading_recommendations(industry_stocks)
                except Exception as tech_error:
                    self.logger.warning(f"添加技术指标和评分失败: {str(tech_error)}")
                
                # 确保必要的列存在
                if 'composite_score' not in industry_stocks.columns:
                    industry_stocks['composite_score'] = 50  # 默认中等评分
                
                if 'trading_advice' not in industry_stocks.columns:
                    industry_stocks['trading_advice'] = "持有观望"  # 默认建议
                
                # 添加量比数据，如果不存在
                if 'vol_ratio' not in industry_stocks.columns:
                    # 计算量比 (最近成交量/5日平均成交量)
                    try:
                        # 尝试获取5日平均成交量
                        avg_vol = 0
                        for code in industry_stocks['ts_code'].unique():
                            code_data = self.data_fetcher.get_stock_kline(code, start_date=None, end_date=None, count=6)
                            if code_data is not None and len(code_data) >= 5:
                                # 计算前5日平均成交量
                                prev_5d_avg = code_data['vol'].iloc[:-1].mean()
                                # 当日成交量
                                today_vol = code_data['vol'].iloc[-1]
                                # 更新该股票的量比
                                mask = industry_stocks['ts_code'] == code
                                industry_stocks.loc[mask, 'vol_ratio'] = today_vol / prev_5d_avg if prev_5d_avg > 0 else 1.0
                        
                        # 对于未计算的股票，使用全行业平均量比
                        avg_vol_ratio = industry_stocks['vol_ratio'].mean()
                        if not pd.isna(avg_vol_ratio) and avg_vol_ratio > 0:
                            industry_stocks['vol_ratio'].fillna(avg_vol_ratio, inplace=True)
                        else:
                            industry_stocks['vol_ratio'].fillna(1.0, inplace=True)
                    except Exception as vol_err:
                        self.logger.warning(f"计算量比失败: {str(vol_err)}")
                        # 使用简化方法计算量比
                        avg_vol = industry_stocks['vol'].mean()
                        industry_stocks['vol_ratio'] = industry_stocks['vol'] / (avg_vol if avg_vol > 0 else 1)
                
                # 计算买入价、卖出价和止损价
                industry_stocks['buy_price'] = industry_stocks.apply(
                    lambda x: round(x['close'] * 0.98, 2) if x['composite_score'] >= 60 else 
                             (round(x['close'] * 0.95, 2) if x['composite_score'] >= 50 else 
                              round(x['close'] * 0.97, 2)), 
                    axis=1
                )
                
                industry_stocks['sell_price'] = industry_stocks.apply(
                    lambda x: round(x['close'] * 1.1, 2) if x['composite_score'] >= 70 else 
                             (round(x['close'] * 1.05, 2) if x['composite_score'] >= 50 else 
                              round(x['close'] * 1.03, 2)), 
                    axis=1
                )
                
                industry_stocks['stop_loss'] = industry_stocks.apply(
                    lambda x: round(x['close'] * 0.93, 2) if x['composite_score'] >= 70 else 
                             (round(x['close'] * 0.95, 2) if x['composite_score'] >= 50 else 
                              round(x['close'] * 0.97, 2)), 
                    axis=1
                )
                
                # 排序
                try:
                    industry_stocks = industry_stocks.sort_values(by='composite_score', ascending=False)
                except Exception as sort_error:
                    self.logger.warning(f"排序股票数据失败: {str(sort_error)}")
            
            # 创建对话框
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{industry_name}行业股票分析")
            dialog.resize(1200, 650)
            
            # 主布局
            layout = QVBoxLayout(dialog)
            
            # 添加标题
            title = QLabel(f"<h3>{industry_name}行业股票分析</h3>")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # 行业数据概览
            up_count = len(industry_stocks[industry_stocks['change'] > 0])
            down_count = len(industry_stocks[industry_stocks['change'] < 0])
            flat_count = len(industry_stocks) - up_count - down_count
            
            overview = QLabel(f"<p>行业股票总数: {len(industry_stocks)} | "
                              f"上涨家数: <span style='color:red;'>{up_count}</span> | "
                              f"下跌家数: <span style='color:green;'>{down_count}</span> | "
                              f"平盘家数: {flat_count}</p>")
            overview.setAlignment(Qt.AlignCenter)
            layout.addWidget(overview)
            
            # 行业交易参考信息
            avg_score = industry_stocks['composite_score'].mean()
            top_stocks = industry_stocks.nlargest(3, 'composite_score')
            top_stocks_text = ", ".join([f"{s.get('name', '')}({s.get('composite_score', 0):.1f}分)" for _, s in top_stocks.iterrows()])
            
            trading_info = QLabel(f"<p>行业平均评分: {avg_score:.1f} | "
                                 f"推荐关注: {top_stocks_text}</p>")
            trading_info.setAlignment(Qt.AlignCenter)
            layout.addWidget(trading_info)
            
            # 创建表格
            table = QTableWidget()
            columns = ["股票代码", "股票名称", "最新价", "涨跌幅", "量比", "技术评分", "价值评分", "动量评分", "综合评分", "建议买入价", "建议卖出价", "止损价", "投资建议"]
            table.setColumnCount(len(columns))
            table.setHorizontalHeaderLabels(columns)
            
            # 添加数据
            table.setRowCount(len(industry_stocks))
            
            for row, (_, stock) in enumerate(industry_stocks.iterrows()):
                # 获取现价，确保是一个有效的数值
                price = stock.get('current', stock.get('close', 0))
                if pd.isna(price) or price == 0:
                    # 如果现价无效，尝试获取其他价格数据
                    for price_field in ['close', 'last_close', 'open']:
                        temp_price = stock.get(price_field, 0)
                        if not pd.isna(temp_price) and temp_price > 0:
                            price = temp_price
                            break
                    # 如果仍然没有有效价格，设置一个默认值
                    if pd.isna(price) or price == 0:
                        price = 10.0  # 设置一个默认价格
                
                # 股票代码
                code_item = QTableWidgetItem(stock.get('ts_code', ''))
                table.setItem(row, 0, code_item)
                
                # 股票名称
                name_item = QTableWidgetItem(stock.get('name', ''))
                name_item.setFont(QFont("", -1, QFont.Bold))
                table.setItem(row, 1, name_item)
                
                # 最新价
                price_item = QTableWidgetItem(f"{price:.2f}")
                table.setItem(row, 2, price_item)
                
                # 涨跌幅
                change = stock.get('change', stock.get('pct_chg', 0))
                if pd.isna(change):
                    change = 0
                change_item = QTableWidgetItem(f"{change:.2f}%")
                if change > 0:
                    change_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif change < 0:
                    change_item.setForeground(QBrush(QColor(0, 255, 0)))
                table.setItem(row, 3, change_item)
                
                # 量比
                vol_ratio = stock.get('vol_ratio', 1.0)
                if pd.isna(vol_ratio):
                    vol_ratio = 1.0
                vol_item = QTableWidgetItem(f"{vol_ratio:.2f}")
                if vol_ratio > 2:
                    vol_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif vol_ratio > 1.5:
                    vol_item.setForeground(QBrush(QColor(255, 165, 0)))
                table.setItem(row, 4, vol_item)
                
                # 技术评分
                tech_score = stock.get('technical_score', 0)
                if pd.isna(tech_score):
                    tech_score = 50
                tech_score_item = QTableWidgetItem(f"{tech_score:.1f}")
                if tech_score > 80:
                    tech_score_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif tech_score > 60:
                    tech_score_item.setForeground(QBrush(QColor(255, 165, 0)))
                table.setItem(row, 5, tech_score_item)
                
                # 价值评分
                value_score = stock.get('value_score', 0)
                if pd.isna(value_score):
                    value_score = 50
                value_score_item = QTableWidgetItem(f"{value_score:.1f}")
                if value_score > 80:
                    value_score_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif value_score > 60:
                    value_score_item.setForeground(QBrush(QColor(255, 165, 0)))
                table.setItem(row, 6, value_score_item)
                
                # 动量评分
                momentum_score = stock.get('momentum_score', 0)
                if pd.isna(momentum_score):
                    momentum_score = 50
                momentum_score_item = QTableWidgetItem(f"{momentum_score:.1f}")
                if momentum_score > 80:
                    momentum_score_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif momentum_score > 60:
                    momentum_score_item.setForeground(QBrush(QColor(255, 165, 0)))
                table.setItem(row, 7, momentum_score_item)
                
                # 综合评分
                score = stock.get('composite_score', 0)
                if pd.isna(score):
                    score = 50
                score_item = QTableWidgetItem(f"{score:.1f}")
                if score > 80:
                    score_item.setForeground(QBrush(QColor(255, 0, 0)))
                    score_item.setFont(QFont("", -1, QFont.Bold))
                elif score > 60:
                    score_item.setForeground(QBrush(QColor(255, 165, 0)))
                table.setItem(row, 8, score_item)
                
                # 建议买入价
                buy_price = stock.get('buy_price', 0)
                if pd.isna(buy_price) or buy_price == 0:
                    # 如果买入价不存在或为0，根据现价和评分计算
                    if score >= 60:
                        buy_price = round(price * 0.98, 2)
                    elif score >= 50:
                        buy_price = round(price * 0.95, 2)
                    else:
                        buy_price = round(price * 0.97, 2)
                buy_item = QTableWidgetItem(f"{buy_price:.2f}")
                buy_item.setForeground(QBrush(QColor(255, 0, 0)))
                table.setItem(row, 9, buy_item)
                
                # 建议卖出价
                sell_price = stock.get('sell_price', 0)
                if pd.isna(sell_price) or sell_price == 0:
                    # 如果卖出价不存在或为0，根据现价和评分计算
                    if score >= 70:
                        sell_price = round(price * 1.1, 2)
                    elif score >= 50:
                        sell_price = round(price * 1.05, 2)
                    else:
                        sell_price = round(price * 1.03, 2)
                sell_item = QTableWidgetItem(f"{sell_price:.2f}")
                sell_item.setForeground(QBrush(QColor(0, 128, 0)))
                table.setItem(row, 10, sell_item)
                
                # 止损价
                stop_loss = stock.get('stop_loss', 0)
                if pd.isna(stop_loss) or stop_loss == 0:
                    # 如果止损价不存在或为0，根据现价和评分计算
                    if score >= 70:
                        stop_loss = round(price * 0.93, 2)
                    elif score >= 50:
                        stop_loss = round(price * 0.95, 2)
                    else:
                        stop_loss = round(price * 0.97, 2)
                stop_item = QTableWidgetItem(f"{stop_loss:.2f}")
                stop_item.setForeground(QBrush(QColor(255, 0, 0)))
                table.setItem(row, 11, stop_item)
                
                # 投资建议
                advice = stock.get('trading_advice', '持有观望')
                if pd.isna(advice) or not advice:
                    # 如果没有建议，根据评分生成
                    if score >= 80:
                        advice = "建议买入"
                    elif score >= 65:
                        advice = "适量买入"
                    elif score <= 30:
                        advice = "建议卖出"
                    elif score <= 40:
                        advice = "减持观望"
                    else:
                        advice = "持有观望"
                advice_item = QTableWidgetItem(advice)
                if "买入" in advice:
                    advice_item.setForeground(QBrush(QColor(255, 0, 0)))
                elif "卖出" in advice or "减持" in advice:
                    advice_item.setForeground(QBrush(QColor(0, 128, 0)))
                table.setItem(row, 12, advice_item)
                
            # 表格属性设置
            table.setEditTriggers(QTableWidget.NoEditTriggers)  # 不可编辑
            table.setSelectionBehavior(QTableWidget.SelectRows)  # 整行选择
            table.setAlternatingRowColors(True)  # 交替行颜色
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  # 列宽自适应内容
            table.horizontalHeader().setStretchLastSection(True)  # 最后一列自动拉伸
            
            # 添加双击事件
            table.cellDoubleClicked.connect(lambda row, col: 
                self._show_stock_details(table.item(row, 0).text(), table.item(row, 1).text()))
                
            # 添加表格和操作按钮
            layout.addWidget(table, 1)
            
            # 创建按钮布局
            button_layout = QHBoxLayout()
            
            # 添加到自选按钮
            add_favorites_btn = QPushButton("添加到自选")
            add_favorites_btn.clicked.connect(lambda: self._add_to_favorites(industry_stocks, table))
            button_layout.addWidget(add_favorites_btn)
            
            # 导出数据按钮
            export_btn = QPushButton("导出数据")
            export_btn.clicked.connect(lambda: self._export_industry_stocks(industry_name, industry_stocks))
            button_layout.addWidget(export_btn)
            
            # 查看细节提示
            detail_label = QLabel("双击股票查看详细分析")
            detail_label.setStyleSheet("color: blue;")
            button_layout.addWidget(detail_label)
            
            # 添加操作说明
            trade_guide = QLabel("<span style='color:#666;'><b>操作说明:</b> 买入价为建议买入价格，卖出价为短期目标价，止损价为建议止损点位</span>")
            button_layout.addWidget(trade_guide)
            
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            self.logger.error(f"显示行业股票时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示行业股票时出错: {str(e)}")
    
    def _export_industry_stocks(self, industry_name, stocks_df):
        """导出行业股票数据"""
        try:
            # 选择保存路径
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "导出行业股票数据", 
                f"{industry_name}行业股票_{datetime.now().strftime('%Y%m%d')}.csv",
                "CSV Files (*.csv)", options=options)
            
            if file_name:
                # 保存为CSV
                stocks_df.to_csv(file_name, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"已成功导出{industry_name}行业股票数据到: {file_name}")
                
        except Exception as e:
            self.logger.error(f"导出行业股票数据失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"导出行业股票数据失败: {str(e)}")
    
    def _generate_market_overview(self):
        """生成市场概况分析"""
        try:
            # 创建市场概况组件
            overview_group = QGroupBox("市场整体概况")
            overview_layout = QVBoxLayout(overview_group)
            
            # 尝试获取大盘数据
            market_trend = "震荡市"
            market_change = 0.0
            try:
                market_data = self.data_fetcher.get_stock_kline('000001.SH')
                if market_data is not None and not market_data.empty and len(market_data) > 1:
                    # 计算当日涨跌幅
                    market_change = (market_data['close'].iloc[-1] / market_data['close'].iloc[-2] - 1) * 100
                    
                    # 计算20日均线
                    if len(market_data) >= 20:
                        market_data['ma20'] = market_data['close'].rolling(window=20).mean()
                        current_price = market_data['close'].iloc[-1]
                        ma20 = market_data['ma20'].iloc[-1]
                        
                        # 判断趋势
                        if current_price > ma20 * 1.05:
                            market_trend = "偏强"
                        elif current_price < ma20 * 0.95:
                            market_trend = "偏弱"
                        else:
                            market_trend = "震荡"
            except Exception as e:
                self.logger.warning(f"获取市场数据失败: {str(e)}")
            
            # 市场概况文本
            market_text = f"""
            <p style='font-size: 14px;'>
            当前市场状态: <b>{market_trend}</b> | 大盘涨跌幅: <span style='color: {"red" if market_change > 0 else "green"};'>{market_change:.2f}%</span><br>
            市场特征: {self._get_market_feature(market_trend, market_change)}<br>
            行业轮动: {self._get_active_industry_count()} 个行业活跃度较高<br>
            投资策略建议: {self._get_market_strategy(market_trend, market_change)}
            </p>
            """
            
            market_label = QLabel(market_text)
            market_label.setWordWrap(True)
            overview_layout.addWidget(market_label)
            
            # 添加热点题材分析
            hot_topics = self._get_hot_topics()
            if hot_topics:
                topics_text = "<p style='font-size: 14px;'><b>近期市场热点题材:</b><br>"
                for topic in hot_topics:
                    topics_text += f"• {topic}<br>"
                topics_text += "</p>"
                
                topics_label = QLabel(topics_text)
                topics_label.setWordWrap(True)
                overview_layout.addWidget(topics_label)
            
            return overview_group
        
        except Exception as e:
            self.logger.error(f"生成市场概况失败: {str(e)}")
            # 返回一个空组件
            empty_group = QGroupBox("市场概况")
            QVBoxLayout(empty_group).addWidget(QLabel("无法获取市场概况数据"))
            return empty_group
    
    def _get_active_industry_count(self):
        """获取活跃行业数量"""
        try:
            industries = self.data_fetcher.get_industry_list()
            active_count = min(5, len([ind for ind in industries if ind != '全部']))
            return active_count
        except:
            return 3  # 默认值
    
    def _get_market_feature(self, trend, change):
        """获取市场特征描述"""
        if trend == "偏强" and change > 1.5:
            return "普涨行情，市场情绪高涨，成交量明显放大"
        elif trend == "偏强" and change > 0:
            return "强势震荡，热点轮动活跃，赚钱效应明显"
        elif trend == "震荡" and change > 0:
            return "结构性行情，板块轮动明显，个股分化严重"
        elif trend == "震荡" and change < 0:
            return "弱势震荡，观望情绪浓厚，交投较为清淡"
        elif trend == "偏弱" and change < -1.5:
            return "普跌行情，避险情绪浓厚，抛压较重"
        else:
            return "市场情绪中性，存在结构性机会"
    
    def _get_market_strategy(self, trend, change):
        """获取市场策略建议"""
        if trend == "偏强" and change > 1.5:
            return "把握强势板块机会，适度参与热点轮动，注意高位风险"
        elif trend == "偏强" and change > 0:
            return "积极把握强势板块机会，关注资金流向变化"
        elif trend == "震荡" and change > 0:
            return "精选个股，关注行业龙头，波段操作为主"
        elif trend == "震荡" and change < 0:
            return "控制仓位，观望为主，等待市场企稳信号"
        elif trend == "偏弱" and change < -1.5:
            return "降低仓位，规避风险，等待市场企稳"
        else:
            return "适度参与，注重防御，关注政策面变化"
    
    def _get_hot_topics(self):
        """获取当前市场热点题材"""
        # 实际应用中应从数据API获取
        # 这里返回模拟数据
        return [
            "人工智能与大数据应用",
            "新能源汽车产业链",
            "国产替代与科技创新",
            "数字经济与云计算",
            "医疗健康创新与医药研发"
        ]
    
    def _generate_industry_recommendation(self, industry, rank):
        """生成行业投资建议"""
        # 获取行业数据
        name = industry['name']
        score = industry.get('composite_score', 0)
        momentum = industry.get('momentum_score', 0)
        change = industry.get('change', 0)
        up_ratio = industry.get('up_ratio', 0)
        relative_strength = industry.get('relative_strength', 0) if 'relative_strength' in industry else 0
        leading_stock = industry.get('leading_stock', '')
        
        # 生成行业评级
        if score > 7:
            rating = "强烈推荐"
            rating_color = "red"
        elif score > 5:
            rating = "推荐关注"
            rating_color = "orange"
        elif score > 3:
            rating = "中性评级"
            rating_color = "blue"
        else:
            rating = "谨慎参与"
            rating_color = "gray"
        
        # 生成行业优势/特点分析
        strengths = []
        
        if momentum > 3:
            strengths.append("动量强劲")
        if up_ratio > 60:
            strengths.append("个股普涨")
        if relative_strength > 1:
            strengths.append("跑赢大盘")
        if change > 2:
            strengths.append("涨幅明显")
            
        if not strengths:
            if score > 5:
                strengths.append("综合表现稳健")
            else:
                strengths.append("暂无明显优势")
        
        strengths_text = "、".join(strengths[:3])
        
        # 生成投资建议
        if score > 7:
            advice = f"建议积极配置该行业龙头股，把握短线和中线机会，跟踪{leading_stock}等核心标的"
        elif score > 5:
            advice = f"关注行业优质个股，适当配置，可重点跟踪{leading_stock}等龙头表现"
        elif score > 3:
            advice = f"可少量试探性配置，以{leading_stock}等龙头为主，注意控制仓位"
        else:
            advice = "建议观望为主，暂不建议重仓配置，等待更明确的信号"
        
        # 生成操作策略
        if momentum > 3 and change > 2:
            strategy = "短期可积极参与，注意高位获利了结"
        elif momentum > 1.5:
            strategy = "波段操作为主，关注量能变化确认"
        elif relative_strength > 1:
            strategy = "相对收益策略，配置优于大盘的标的"
        else:
            strategy = "谨慎为主，低吸为主，设置止损位"
            
        # 构建完整推荐文本
        recommendation = f"""
        <p><span style='color: {rating_color}; font-weight: bold;'>【{rank}】{name} - {rating}</span></p>
        <p><b>综合评分:</b> {score:.2f} | <b>行业特点:</b> {strengths_text}</p>
        <p><b>龙头股:</b> {leading_stock}</p>
        <p><b>投资建议:</b> {advice}</p>
        <p><b>操作策略:</b> {strategy}</p>
        """
        
        return recommendation
    
    def _on_market_overview_clicked(self):
        """市场概览按钮点击事件"""
        QMessageBox.information(self, "市场概览", "市场概览功能将在后续版本中实现")
    
    def _on_index_analysis_clicked(self):
        """指数分析按钮点击事件"""
        QMessageBox.information(self, "指数分析", "指数分析功能将在后续版本中实现")
    
    def _on_market_sentiment_clicked(self):
        """市场情绪按钮点击事件"""
        QMessageBox.information(self, "市场情绪", "市场情绪分析功能将在后续版本中实现")
        
    def _on_north_fund_clicked(self):
        """北向资金按钮点击事件"""
        QMessageBox.information(self, "北向资金", "北向资金分析功能将在后续版本中实现")
        
    def _on_market_breadth_clicked(self):
        """市场宽度按钮点击事件"""
        QMessageBox.information(self, "市场宽度", "市场宽度分析功能将在后续版本中实现")
    
    def _on_ai_search_clicked(self):
        """处理AI预测页面的股票搜索"""
        search_text = self.pred_stock_search.text().strip()
        if not search_text:
            QMessageBox.warning(self, "警告", "请输入股票代码或名称！")
            return
        QMessageBox.information(self, "AI搜索", "AI预测搜索功能将在后续版本中实现")
    
    def _on_ai_stock_selected(self, row, column):
        """处理AI预测页面的股票选择"""
        QMessageBox.information(self, "股票选择", "AI预测股票选择功能将在后续版本中实现")