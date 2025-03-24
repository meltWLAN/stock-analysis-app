import sys
import os

# 添加应用目录到路径
path = '/home/WZ2025/wz250323001'
if path not in sys.path:
    sys.path.append(path)

# 调试信息（部署后可以注释掉这些行）
print("Python 路径:")
for p in sys.path:
    print(" - " + p)

print("当前目录: " + os.getcwd())
print("目录内容:")
for f in os.listdir(path):
    print(" - " + f)

# 导入应用
from app import app as application  # noqa

# 如果使用了.env文件或环境变量
# 建议在PythonAnywhere的Web选项卡中设置环境变量
# 而不是在这里设置，以确保安全 