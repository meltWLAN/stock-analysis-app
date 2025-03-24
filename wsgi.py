import sys
import os

# 添加应用目录到路径
path = '/home/WZ2025/wz250323001'
if path not in sys.path:
    sys.path.append(path)

# 创建一个最简单的Flask应用
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <h1>智能股票分析平台</h1>
    <p>基本Flask应用已成功加载!</p>
    <p>当前路径: """ + os.getcwd() + """</p>
    <p>Python路径:</p>
    <ul>
    """ + "".join(["<li>" + p + "</li>" for p in sys.path]) + """
    </ul>
    """

# 应用入口
application = app 