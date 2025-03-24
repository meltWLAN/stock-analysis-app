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

# 应用入口点
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000) 