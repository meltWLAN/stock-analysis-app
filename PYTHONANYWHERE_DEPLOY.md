# PythonAnywhere 部署指南

本文档提供在PythonAnywhere上部署股票分析系统的详细步骤，以便通过手机访问所有功能。

## 步骤1: 注册PythonAnywhere账号

1. 访问 [PythonAnywhere](https://www.pythonanywhere.com/) 并注册一个账号
2. 免费账号可以满足初期需求，如需更多功能，可以升级到付费计划

## 步骤2: 设置Web应用

1. 登录PythonAnywhere后，点击仪表板中的**Web**选项卡
2. 点击**添加一个新的Web应用**按钮
3. 选择域名（免费账号将使用yourname.pythonanywhere.com格式）
4. 选择Python版本（推荐Python 3.8或更高版本）
5. 选择**Flask**框架
6. 修改配置路径：在源代码目录下输入`/home/YOUR_USERNAME/stock-analysis-app`

## 步骤3: 上传代码到PythonAnywhere

### 方法1: 使用Git（推荐）

1. 打开PythonAnywhere的Bash终端（在**控制台**选项卡下）
2. 执行以下命令：

```bash
# 进入home目录
cd ~

# 克隆仓库
git clone https://github.com/meltWLAN/stock-analysis-app.git

# 进入项目目录
cd stock-analysis-app

# 安装依赖包（使用pip3确保使用Python 3）
pip3 install --user -r requirements.txt
```

### 方法2: 上传ZIP文件

1. 在本地将项目打包为ZIP文件
2. 在PythonAnywhere的**文件**选项卡中上传ZIP文件
3. 在Bash控制台中解压文件：
```bash
cd ~
unzip stock-analysis-app.zip -d stock-analysis-app
cd stock-analysis-app
pip3 install --user -r requirements.txt
```

## 步骤4: 配置WSGI文件

1. 在Web选项卡中，点击WSGI配置文件的链接
2. 替换内容为以下代码（记得替换`YOUR_USERNAME`为您的实际用户名）：

```python
import sys

# 添加应用目录到路径
path = '/home/YOUR_USERNAME/stock-analysis-app'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
```

## 步骤5: 配置静态文件

1. 在Web选项卡的**Static Files**部分添加以下配置：
   - URL: `/static/`
   - Directory: `/home/YOUR_USERNAME/stock-analysis-app/web_app/static`

## 步骤6: 重载Web应用

1. 在Web选项卡中点击**重载**按钮
2. 等待几秒钟，然后访问您的域名（例如：`yourname.pythonanywhere.com`）

## 步骤7: 调试和问题解决

如果应用无法正常运行：

1. 检查错误日志（在Web选项卡中的**日志文件**部分）
2. 确保所有依赖都已安装
3. 检查文件路径是否正确配置

## 移动端优化

本应用已经针对移动端进行了优化:

1. 使用了响应式设计，自动适应不同设备屏幕
2. 针对触摸操作进行了优化
3. 简化了图表显示，适合小屏幕

## 访问地址

部署完成后，您可以通过以下地址访问应用：

- 网址: `http://YOUR_USERNAME.pythonanywhere.com`
- 使用任何现代手机浏览器访问以获得最佳体验

## 自动化更新（可选）

如果您需要定期从GitHub更新代码，可以设置每日任务：

1. 在**控制台**选项卡中选择**计划任务**
2. 添加一个每日任务，命令如下：

```bash
cd ~/stock-analysis-app && git pull && pip3 install --user -r requirements.txt && touch /var/www/YOUR_USERNAME_pythonanywhere_com_wsgi.py
```

这将每天从GitHub拉取最新代码并重新加载应用。 