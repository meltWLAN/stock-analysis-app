#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PythonAnywhere WSGI配置文件
"""

import sys
import os

# 添加应用目录到路径
path = '/home/WZ2025/wz250323001'
if path not in sys.path:
    sys.path.append(path)

# 导入最小化应用
try:
    from minimal_app import app as application
except ImportError as e:
    print(f"导入错误: {e}")
    
    # 如果导入失败，创建一个简单的应用返回错误信息
    from flask import Flask, render_template_string
    
    application = Flask(__name__)
    
    @application.route('/')
    def error_page():
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>导入错误</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #d9534f; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>应用加载错误</h1>
            <p>无法加载应用，请检查日志文件以获取更多信息。</p>
            <p>当前工作目录: {{ cwd }}</p>
            <p>Python路径:</p>
            <pre>{{ pythonpath }}</pre>
            <p>导入错误: {{ error }}</p>
        </body>
        </html>
        """, cwd=os.getcwd(), pythonpath="\n".join(sys.path), error=str(e)) 