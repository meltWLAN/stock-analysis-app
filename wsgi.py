import sys

# 添加应用目录到路径
path = '/home/WZ2025/stock-analysis-app'
if path not in sys.path:
    sys.path.append(path)

from app import app as application  # noqa

# 如果使用了.env文件或环境变量
# 建议在PythonAnywhere的Web选项卡中设置环境变量
# 而不是在这里设置，以确保安全 