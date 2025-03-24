# 智能股票分析平台

基于AI的股票数据分析和预测平台，支持基本股票数据分析、趋势预测和投资建议。

## 功能特点

- 市场概览：展示主要指数、热点板块和市场情绪指标
- 股票详情：提供股票基本信息、价格走势和财务数据
- AI预测：使用机器学习算法预测股票未来价格趋势
- 股票筛选：根据多种技术指标和基本面指标筛选股票
- 持仓组合：创建和管理自己的股票投资组合
- 市场监控：设置价格提醒和异常波动提醒

## 技术栈

- 后端：Flask (Python)
- 前端：Bootstrap 5, JavaScript, ApexCharts
- 数据分析：NumPy, Pandas, Scikit-learn

## 快速开始

### 环境要求

- Python 3.8+

### 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/stock-analysis-platform.git
cd stock-analysis-platform
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
python app.py
```

4. 浏览器访问
```
http://localhost:5000
```

## 移动端支持

系统支持响应式设计，自动适应不同设备屏幕大小。移动端特别优化：

- 底部导航栏，便于单手操作
- 简化图表显示，适应小屏幕
- 优化触摸交互

## 示例截图

![首页截图](screenshot1.png)
![股票详情](screenshot2.png)

## 数据来源

系统目前使用模拟数据，实际应用中可对接以下数据源：

- 免费数据：TuShare、AKShare
- 付费数据：万得(Wind)、同花顺、东方财富等

## 许可证

MIT