#!/bin/bash
# 最小化部署脚本 - 在PythonAnywhere的Bash控制台运行

# 设置变量
USERNAME="WZ2025"
PROJECT_NAME="wz250323001"
PROJECT_PATH="/home/$USERNAME/$PROJECT_NAME"

echo "===== 开始部署最小化应用到PythonAnywhere ====="

# 清理旧目录（如果存在）
echo "清理旧目录..."
cd ~
rm -rf "$PROJECT_NAME"

# 创建项目目录
echo "创建项目目录..."
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# 创建最小化应用
echo "创建最小化应用..."
cat > minimal_app.py << 'EOL'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最小化Flask应用 - 用于PythonAnywhere部署测试
"""

from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    """简单首页"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>智能股票分析平台</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #333; }
            .nav-button {
                display: inline-block;
                padding: 10px 15px;
                margin: 5px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>智能股票分析平台</h1>
        <p>最小化版本已成功加载！</p>
        
        <div>
            <a href="/" class="nav-button">首页</a>
            <a href="/about" class="nav-button">关于</a>
        </div>
    </body>
    </html>
    """)

@app.route('/about')
def about():
    """关于页面"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>关于 - 智能股票分析平台</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 800px; 
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #333; }
            .nav-button {
                display: inline-block;
                padding: 10px 15px;
                margin: 5px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>关于本平台</h1>
        <p>这是一个基于AI的股票数据分析和预测平台。</p>
        <p>当前版本：最小化测试版本</p>
        
        <div>
            <a href="/" class="nav-button">返回首页</a>
        </div>
    </body>
    </html>
    """)
EOL

# 创建wsgi文件
echo "创建wsgi文件内容..."
cat > wsgi.py << 'EOL'
import sys
import os

# 添加应用目录到路径
path = '/home/WZ2025/wz250323001'
if path not in sys.path:
    sys.path.append(path)

# 导入最小化应用
from minimal_app import app as application
EOL

# 安装最小依赖
echo "安装Flask..."
pip3 install --user flask==2.3.3

echo "===== 部署完成 ====="
echo "请在PythonAnywhere Web选项卡中点击Reload按钮重新加载应用"
echo "===== 完成 =====" 