from web_app.routes import app

# 如果直接运行此文件，进入调试模式
if __name__ == '__main__':
    # 开发环境中使用
    app.run(debug=True, host='0.0.0.0', port=5000)