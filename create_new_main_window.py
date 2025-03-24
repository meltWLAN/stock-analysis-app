#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成改进版main_window.py文件的脚本
"""

import os

# main_window.py的内容
content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"\"\"
主窗口模块 - 应用程序的主界面
\"\"\"

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QMessageBox, QDialog, QTabWidget,
    QTextEdit, QLineEdit, QGroupBox, QProgressDialog, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush

class MainWindow(QMainWindow):
    \"\"\"主窗口类\"\"\"
    
    def __init__(self, system_components=None):
        \"\"\"初始化主窗口\"\"\"
        super().__init__()
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 保存系统组件
        self.system_components = system_components or {}
        
        # 初始化数据
        self._init_data()
        
        # 初始化UI
        self._init_ui()
        
        # 记录初始化完成
        self.logger.info("主窗口初始化完成")
        
    def _init_data(self):
        \"\"\"初始化数据和配置\"\"\"
        # 设置应用程序目录
        self.app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 配置各种目录
        self.data_dir = os.path.join(self.app_dir, "data")
        self.cache_dir = os.path.join(self.app_dir, "cache")
        self.log_dir = os.path.join(self.app_dir, "logs")
        self.results_dir = os.path.join(self.app_dir, "results")
        
        # 创建必要的目录
        for directory in [self.data_dir, self.cache_dir, self.log_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # 初始化数据获取器
        if 'data_fetcher' in self.system_components:
            self.data_fetcher = self.system_components['data_fetcher']
        else:
            # 如果没有提供，则创建一个默认的
            from data.stock_data import StockDataFetcher
            self.data_fetcher = StockDataFetcher()
            
        # 加载自选股列表
        self.favorites = []
        self._load_favorites()
        
    def _init_ui(self):
        \"\"\"初始化UI\"\"\"
        # 设置窗口基本属性
        self.setWindowTitle("股票分析系统")
        self.setGeometry(100, 100, 1280, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建功能区布局
        function_frame = QFrame()
        function_frame.setFrameShape(QFrame.StyledPanel)
        function_frame.setMaximumHeight(100)
        function_layout = QHBoxLayout(function_frame)
        
        # 添加功能按钮
        self.btn_market_overview = QPushButton("市场概览")
        self.btn_hot_industry = QPushButton("热门行业")
        self.btn_stock_selection = QPushButton("选股策略")
        self.btn_favorites = QPushButton("自选股")
        self.btn_data_analysis = QPushButton("数据分析")
        self.btn_settings = QPushButton("设置")
        
        # 添加按钮到布局
        function_layout.addWidget(self.btn_market_overview)
        function_layout.addWidget(self.btn_hot_industry)
        function_layout.addWidget(self.btn_stock_selection)
        function_layout.addWidget(self.btn_favorites)
        function_layout.addWidget(self.btn_data_analysis)
        function_layout.addWidget(self.btn_settings)
        
        # 添加功能区到主布局
        main_layout.addWidget(function_frame)
        
        # 工作区框架
        work_frame = QFrame()
        work_frame.setFrameShape(QFrame.StyledPanel)
        work_layout = QVBoxLayout(work_frame)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # 创建各个选项卡内容
        self._create_market_tab()
        self._create_industry_tab()
        self._create_stock_tab()
        self._create_favorites_tab()
        self._create_analysis_tab()
        
        # 添加选项卡到工作区
        work_layout.addWidget(self.tab_widget)
        
        # 添加工作区到主布局
        main_layout.addWidget(work_frame)
        
        # 连接信号和槽
        self._connect_signals()
        
        # 设置状态栏
        self.statusBar().showMessage("就绪")
        
    def _create_market_tab(self):
        \"\"\"创建市场概览选项卡\"\"\"
        market_tab = QWidget()
        market_layout = QVBoxLayout(market_tab)
        
        # 市场指数区域
        index_group = QGroupBox("市场指数")
        index_layout = QHBoxLayout(index_group)
        
        # 上证指数
        self.sh_index_label = QLabel("上证指数: --")
        self.sh_index_change_label = QLabel("涨跌幅: --")
        
        # 深证成指
        self.sz_index_label = QLabel("深证成指: --")
        self.sz_index_change_label = QLabel("涨跌幅: --")
        
        # 创业板指
        self.cyb_index_label = QLabel("创业板指: --")
        self.cyb_index_change_label = QLabel("涨跌幅: --")
        
        # 添加到指数布局
        index_layout.addWidget(self.sh_index_label)
        index_layout.addWidget(self.sh_index_change_label)
        index_layout.addWidget(self.sz_index_label)
        index_layout.addWidget(self.sz_index_change_label)
        index_layout.addWidget(self.cyb_index_label)
        index_layout.addWidget(self.cyb_index_change_label)
        
        # 市场资金流向区域
        flow_group = QGroupBox("资金流向")
        flow_layout = QVBoxLayout(flow_group)
        
        # 北向资金
        self.north_money_label = QLabel("北向资金: --")
        flow_layout.addWidget(self.north_money_label)
        
        # 更新按钮
        self.refresh_market_btn = QPushButton("刷新数据")
        
        # 添加到市场选项卡
        market_layout.addWidget(index_group)
        market_layout.addWidget(flow_group)
        market_layout.addWidget(self.refresh_market_btn)
        market_layout.addStretch()
        
        # 添加到选项卡组件
        self.tab_widget.addTab(market_tab, "市场概览")
        
    def _create_industry_tab(self):
        \"\"\"创建行业分析选项卡\"\"\"
        industry_tab = QWidget()
        industry_layout = QVBoxLayout(industry_tab)
        
        # 行业选择区域
        industry_select_group = QGroupBox("行业选择")
        industry_select_layout = QHBoxLayout(industry_select_group)
        
        self.industry_combo = QComboBox()
        self.industry_combo.addItem("请选择行业")
        
        self.load_industry_btn = QPushButton("加载行业数据")
        industry_select_layout.addWidget(self.industry_combo)
        industry_select_layout.addWidget(self.load_industry_btn)
        
        # 热门行业分析按钮
        self.hot_industry_btn = QPushButton("热门行业分析")
        
        # 行业股票列表
        industry_stocks_group = QGroupBox("行业股票")
        industry_stocks_layout = QVBoxLayout(industry_stocks_group)
        
        self.industry_stocks_table = QTableWidget()
        self.industry_stocks_table.setColumnCount(6)
        self.industry_stocks_table.setHorizontalHeaderLabels(["股票代码", "股票名称", "现价", "涨跌幅", "成交量", "成交额"])
        self.industry_stocks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        industry_stocks_layout.addWidget(self.industry_stocks_table)
        
        # 添加到行业选项卡
        industry_layout.addWidget(industry_select_group)
        industry_layout.addWidget(self.hot_industry_btn)
        industry_layout.addWidget(industry_stocks_group)
        
        # 添加到选项卡组件
        self.tab_widget.addTab(industry_tab, "行业分析")
        
    def _create_stock_tab(self):
        \"\"\"创建个股分析选项卡\"\"\"
        stock_tab = QWidget()
        stock_layout = QVBoxLayout(stock_tab)
        
        # 股票选择区域
        stock_select_group = QGroupBox("股票选择")
        stock_select_layout = QHBoxLayout(stock_select_group)
        
        self.stock_code_input = QLineEdit()
        self.stock_code_input.setPlaceholderText("输入股票代码")
        
        self.load_stock_btn = QPushButton("加载股票数据")
        stock_select_layout.addWidget(self.stock_code_input)
        stock_select_layout.addWidget(self.load_stock_btn)
        
        # 添加到自选按钮
        self.add_to_favorites_btn = QPushButton("添加到自选")
        
        # 股票信息区域
        stock_info_group = QGroupBox("股票信息")
        stock_info_layout = QVBoxLayout(stock_info_group)
        
        self.stock_info_table = QTableWidget()
        self.stock_info_table.setColumnCount(2)
        self.stock_info_table.setHorizontalHeaderLabels(["指标", "值"])
        self.stock_info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stock_info_table.setRowCount(5)
        
        stock_info_layout.addWidget(self.stock_info_table)
        
        # 添加到股票选项卡
        stock_layout.addWidget(stock_select_group)
        stock_layout.addWidget(self.add_to_favorites_btn)
        stock_layout.addWidget(stock_info_group)
        
        # 添加到选项卡组件
        self.tab_widget.addTab(stock_tab, "个股分析")
        
    def _create_favorites_tab(self):
        \"\"\"创建自选股选项卡\"\"\"
        favorites_tab = QWidget()
        favorites_layout = QVBoxLayout(favorites_tab)
        
        # 自选股列表
        favorites_group = QGroupBox("自选股列表")
        favorites_layout_group = QVBoxLayout(favorites_group)
        
        self.favorites_table = QTableWidget()
        self.favorites_table.setColumnCount(7)
        self.favorites_table.setHorizontalHeaderLabels(["股票代码", "股票名称", "现价", "涨跌幅", "建议买入价", "建议卖出价", "操作"])
        self.favorites_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        favorites_layout_group.addWidget(self.favorites_table)
        
        # 批量操作按钮
        batch_btns_layout = QHBoxLayout()
        self.refresh_favorites_btn = QPushButton("刷新数据")
        self.remove_selected_btn = QPushButton("删除选中")
        self.analyze_favorites_btn = QPushButton("分析自选股")
        
        batch_btns_layout.addWidget(self.refresh_favorites_btn)
        batch_btns_layout.addWidget(self.remove_selected_btn)
        batch_btns_layout.addWidget(self.analyze_favorites_btn)
        
        # 添加到自选股选项卡
        favorites_layout.addWidget(favorites_group)
        favorites_layout.addLayout(batch_btns_layout)
        
        # 添加到选项卡组件
        self.tab_widget.addTab(favorites_tab, "自选股")
        
    def _create_analysis_tab(self):
        \"\"\"创建数据分析选项卡\"\"\"
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        
        # 分析类型选择
        analysis_type_group = QGroupBox("分析类型")
        analysis_type_layout = QHBoxLayout(analysis_type_group)
        
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems(["技术分析", "基本面分析", "资金流向分析", "预测分析"])
        
        analysis_type_layout.addWidget(self.analysis_type_combo)
        
        # 分析参数设置
        analysis_params_group = QGroupBox("分析参数")
        analysis_params_layout = QVBoxLayout(analysis_params_group)
        
        # 参数将根据分析类型动态生成
        
        # 执行分析按钮
        self.run_analysis_btn = QPushButton("执行分析")
        
        # 添加到数据分析选项卡
        analysis_layout.addWidget(analysis_type_group)
        analysis_layout.addWidget(analysis_params_group)
        analysis_layout.addWidget(self.run_analysis_btn)
        analysis_layout.addStretch()
        
        # 添加到选项卡组件
        self.tab_widget.addTab(analysis_tab, "数据分析")
        
    def _connect_signals(self):
        \"\"\"连接信号和槽\"\"\"
        # 功能按钮
        self.btn_market_overview.clicked.connect(lambda: self.tab_widget.setCurrentIndex(0))
        self.btn_hot_industry.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        self.btn_stock_selection.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        self.btn_favorites.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        self.btn_data_analysis.clicked.connect(lambda: self.tab_widget.setCurrentIndex(4))
        self.btn_settings.clicked.connect(self._show_settings)
        
        # 行业分析
        self.load_industry_btn.clicked.connect(lambda: self.logger.info("加载行业数据"))
        self.industry_stocks_table.doubleClicked.connect(self._on_industry_stock_double_clicked)
        self.hot_industry_btn.clicked.connect(lambda: self.logger.info("热门行业分析"))
        
        # 个股分析
        self.load_stock_btn.clicked.connect(lambda: self.logger.info("加载股票数据"))
        self.add_to_favorites_btn.clicked.connect(lambda: self.logger.info("添加到自选股"))
        
        # 自选股
        self.refresh_favorites_btn.clicked.connect(lambda: self.logger.info("刷新自选股数据"))
        self.remove_selected_btn.clicked.connect(lambda: self.logger.info("删除选中的自选股"))
        self.analyze_favorites_btn.clicked.connect(lambda: self.logger.info("分析自选股"))
        
        # 数据分析
        self.run_analysis_btn.clicked.connect(lambda: self.logger.info("执行分析"))
        
    def _on_industry_stock_double_clicked(self, index):
        \"\"\"处理行业股票表格双击事件\"\"\"
        row = index.row()
        if self.industry_stocks_table.rowCount() > row:
            stock_code_item = self.industry_stocks_table.item(row, 0)
            stock_name_item = self.industry_stocks_table.item(row, 1)
            
            if stock_code_item and stock_name_item:
                stock_code = stock_code_item.text()
                stock_name = stock_name_item.text()
                
                # 显示股票详情
                self._show_stock_details(stock_code, stock_name)
    
    def _show_settings(self):
        \"\"\"显示设置对话框\"\"\"
        QMessageBox.information(self, "设置", "设置功能开发中...")
        
    def _load_favorites(self):
        \"\"\"加载自选股列表\"\"\"
        favorites_file = os.path.join(self.data_dir, "favorites.csv")
        if os.path.exists(favorites_file):
            try:
                df = pd.read_csv(favorites_file)
                self.favorites = df.to_dict('records')
                self.logger.info(f"已加载 {len(self.favorites)} 个自选股")
            except Exception as e:
                self.logger.error(f"加载自选股列表失败: {e}")
                self.favorites = []
        else:
            self.logger.info("未找到自选股文件，创建新的自选股列表")
            self.favorites = []
            
    def _save_favorites(self):
        \"\"\"保存自选股列表\"\"\"
        favorites_file = os.path.join(self.data_dir, "favorites.csv")
        try:
            df = pd.DataFrame(self.favorites)
            df.to_csv(favorites_file, index=False)
            self.logger.info(f"已保存 {len(self.favorites)} 个自选股")
        except Exception as e:
            self.logger.error(f"保存自选股列表失败: {e}")
            
    def _add_single_to_favorites(self, stock_code, stock_name="", buy_price=0, sell_price=0, stop_loss=0, score=0):
        \"\"\"添加单个股票到自选列表\"\"\"
        # 检查是否已存在
        for item in self.favorites:
            if item["stock_code"] == stock_code:
                # 已存在，更新数据
                item["stock_name"] = stock_name
                item["buy_price"] = buy_price
                item["sell_price"] = sell_price
                item["stop_loss"] = stop_loss
                item["score"] = score
                self.logger.info(f"更新自选股: {stock_code}")
                self._save_favorites()
                return
                
        # 不存在，添加新的
        self.favorites.append({
            "stock_code": stock_code,
            "stock_name": stock_name,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "stop_loss": stop_loss,
            "score": score,
            "add_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        self.logger.info(f"添加自选股: {stock_code}")
        self._save_favorites()
            
    def _format_market_cap(self, value):
        \"\"\"格式化市值显示\"\"\"
        if pd.isna(value) or value == 0:
            return "未知"
            
        value = float(value)
        if value >= 100000000000:  # 1000亿
            return f"{value/100000000000:.2f}千亿"
        elif value >= 100000000:  # 1亿
            return f"{value/100000000:.2f}亿"
        elif value >= 10000:  # 1万
            return f"{value/10000:.2f}万"
        else:
            return f"{value:.2f}"
            
    def _show_stock_details(self, stock_code, stock_name=""):
        \"\"\"显示股票详情\"\"\"
        try:
            if not stock_code:
                QMessageBox.warning(self, "警告", "股票代码不能为空")
                return
                
            # 创建进度对话框
            progress = QProgressDialog(f"正在获取 {stock_name} 详细数据...", "取消", 0, 100, self)
            progress.setWindowTitle("股票详情")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            progress.show()
            
            QApplication.processEvents()
            
            # 获取股票详细数据
            stock_data = self.data_fetcher.get_stock_details(stock_code)
            
            if progress.wasCanceled():
                return
                
            progress.setValue(40)
            QApplication.processEvents()
            
            # 获取K线数据
            kline_data = self.data_fetcher.get_stock_kline(stock_code)
            
            if progress.wasCanceled():
                return
                
            progress.setValue(70)
            QApplication.processEvents()
            
            # 获取股票名称
            if not stock_name and 'name' in stock_data:
                stock_name = stock_data['name']
                
            # 创建详情对话框
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{stock_name}({stock_code}) - 股票详情")
            dialog.resize(800, 600)
            
            # 创建布局
            layout = QVBoxLayout()
            dialog.setLayout(layout)
            
            # 基本信息区域
            base_info_group = QGroupBox("基本信息")
            base_info_layout = QVBoxLayout()
            base_info_group.setLayout(base_info_layout)
            
            # 基本信息标签
            info_label = QLabel(f"股票代码: {stock_code}\\n"
                               f"股票名称: {stock_name}\\n"
                               f"行业: {stock_data.get('industry', '未知')}")
            base_info_layout.addWidget(info_label)
            
            layout.addWidget(base_info_group)
            
            # 底部按钮
            button_layout = QHBoxLayout()
            
            # 添加到自选按钮
            add_btn = QPushButton("添加到自选")
            add_btn.clicked.connect(lambda: self._add_single_to_favorites(
                stock_code, stock_name
            ))
            button_layout.addWidget(add_btn)
            
            # 关闭按钮
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            # 完成
            progress.setValue(100)
            
            # 显示对话框
            dialog.exec_()
            
        except Exception as e:
            self.logger.error(f"显示股票详情时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示股票详情时出错: {str(e)}")
"""

# 确保目标目录存在
os.makedirs(os.path.join(os.path.dirname(__file__), "ui"), exist_ok=True)

# 写入文件
with open(os.path.join(os.path.dirname(__file__), "ui", "main_window.py"), "w", encoding="utf-8") as f:
    f.write(content)

print("改进的main_window.py文件已生成") 