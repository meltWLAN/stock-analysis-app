from web_app.routes import app

# 如果直接运行此文件，进入调试模式
if __name__ == '__main__':
    # 开发环境中使用
    # 在生产环境中一定要禁用debug模式
    app.run(debug=False, host='0.0.0.0', port=5000)