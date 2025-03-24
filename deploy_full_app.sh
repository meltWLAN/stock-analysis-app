#!/bin/bash
# 完整应用部署脚本 - 在PythonAnywhere的Bash控制台运行

# 设置变量
USERNAME="WZ2025"
PROJECT_NAME="wz250323001"
PROJECT_PATH="/home/$USERNAME/$PROJECT_NAME"
GIT_REPO="https://github.com/meltWLAN/stock-analysis-app.git"

echo "===== 开始部署完整应用到PythonAnywhere ====="

# 清理旧目录（如果存在）
echo "清理旧目录..."
cd ~
rm -rf "$PROJECT_NAME"

# 从GitHub克隆代码
echo "从GitHub克隆代码..."
git clone "$GIT_REPO" "$PROJECT_NAME"

# 进入项目目录
echo "进入项目目录..."
cd "$PROJECT_NAME"

# 创建主应用文件
echo "创建主应用文件..."
cat > app.py << 'EOL'
from web_app.routes import app

# 如果直接运行此文件，进入调试模式
if __name__ == '__main__':
    # 在生产环境中禁用调试模式
    app.run(debug=False, host='0.0.0.0', port=5000)
EOL

# 创建WSGI文件
echo "创建WSGI文件..."
cat > wsgi.py << 'EOL'
import sys
import os

# 添加应用目录到路径
path = '/home/WZ2025/wz250323001'
if path not in sys.path:
    sys.path.append(path)

# 导入应用
from app import app as application
EOL

# 安装基本依赖
echo "安装基本依赖..."
pip3 install --user flask==2.3.3 Werkzeug==2.3.7 Jinja2==3.1.2 MarkupSafe==2.1.3 itsdangerous==2.1.2 click==8.1.7

# 安装数据处理依赖
echo "安装数据处理依赖..."
pip3 install --user numpy>=1.20.0,<1.24.0 pandas>=1.5.0,<2.0.0 matplotlib>=3.4.0 plotly>=5.5.0

# 安装其他必要依赖
echo "安装其他必要依赖..."
pip3 install --user gunicorn==21.2.0 pyecharts>=1.9.0 scipy>=1.7.0

# 确保目录存在
echo "创建日志目录..."
mkdir -p logs

echo "===== 部署完成 ====="
echo "请确保PythonAnywhere Web配置正确:"
echo "1. 源代码目录: $PROJECT_PATH"
echo "2. 静态文件:"
echo "   - URL: /static/"
echo "   - 目录: $PROJECT_PATH/web_app/static"
echo "3. 然后点击Reload按钮重新加载Web应用"
echo ""
echo "===== 完成 =====" 