#!/bin/bash
# 部署脚本 - 在PythonAnywhere的Bash控制台运行

# 设置变量
USERNAME="WZ2025"
PROJECT_NAME="wz250323001"
PROJECT_PATH="/home/$USERNAME/$PROJECT_NAME"
GIT_REPO="https://github.com/meltWLAN/stock-analysis-app.git"

echo "===== 开始部署项目到 PythonAnywhere ====="

# 检查 git 是否可用
if ! command -v git &> /dev/null; then
    echo "错误: git 命令不可用，请确保已安装 git"
    exit 1
fi

# 清理旧目录（如果存在）
if [ -d "$PROJECT_PATH" ]; then
    echo "删除旧项目目录..."
    rm -rf "$PROJECT_PATH"
fi

# 创建项目目录
echo "创建新项目目录..."
mkdir -p "$PROJECT_PATH"

# 从Git克隆代码
echo "从Git克隆代码: $GIT_REPO"
if ! git clone $GIT_REPO "$PROJECT_PATH"; then
    echo "错误: Git克隆失败，请检查仓库URL和网络连接"
    exit 1
fi

# 进入项目目录
echo "进入项目目录..."
cd "$PROJECT_PATH" || {
    echo "错误: 无法进入项目目录"
    exit 1
}

# 检查项目结构
echo "检查项目结构..."
if [ ! -f "app.py" ]; then
    echo "警告: 未找到 app.py 文件"
    echo "这可能是由于仓库结构与预期不同"
    echo "尝试查找项目中的主应用文件..."
    
    # 尝试在子目录中查找
    APP_FILES=$(find . -name "app.py" | sort)
    if [ -n "$APP_FILES" ]; then
        echo "找到以下 app.py 文件:"
        echo "$APP_FILES"
        echo "使用第一个找到的文件"
        # 如果需要，可以将项目内容移到根目录
    else
        echo "警告: 未找到 app.py 文件，部署可能会失败"
    fi
fi

# 安装依赖
echo "安装项目依赖..."
if [ -f "requirements.txt" ]; then
    pip3 install --user -r requirements.txt
else
    echo "警告: 未找到 requirements.txt 文件，跳过依赖安装"
fi

# 确保WSGI文件正确配置
echo "配置WSGI文件..."
cat > "$PROJECT_PATH/wsgi.py" << EOL
import sys
import os

# 添加应用目录到路径
path = '$PROJECT_PATH'
if path not in sys.path:
    sys.path.append(path)

# 调试信息
print("Python 路径:")
for p in sys.path:
    print(" - " + p)

print("当前目录: " + os.getcwd())
print("目录内容:")
for f in os.listdir(path):
    print(" - " + f)

# 导入应用
from app import app as application  # noqa
EOL

echo "===== 部署步骤 ====="
echo "请手动完成以下步骤:"
echo "1. 登录 PythonAnywhere 并访问 Web 选项卡"
echo "2. 添加新的 Web 应用"
echo "   - 域名: $USERNAME.pythonanywhere.com"
echo "   - Python 版本: 3.8 或更高"
echo "   - Flask 框架"
echo "   - 源代码目录: $PROJECT_PATH"
echo "3. 配置静态文件"
echo "   - URL: /static/"
echo "   - 目录: $PROJECT_PATH/web_app/static"
echo "4. 重新加载 Web 应用"
echo ""
echo "访问应用: http://$USERNAME.pythonanywhere.com"
echo ""
echo "===== 部署脚本执行完成! =====" 