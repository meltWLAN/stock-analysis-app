# PythonAnywhere 自动部署指南

这是一个使用 Git 快速部署项目到 PythonAnywhere 的指南。

## 准备工作

1. 确保您已经有一个 PythonAnywhere 账户（用户名: WZ2025）
2. 确保您有访问 GitHub 仓库的权限：https://github.com/meltWLAN/stock-analysis-app

## 部署步骤

### 第一步：上传部署脚本到 PythonAnywhere

1. 登录您的 PythonAnywhere 账户：[https://www.pythonanywhere.com/login/](https://www.pythonanywhere.com/login/)
2. 使用用户名 `WZ2025` 和您的密码登录
3. 点击页面顶部的 **Files** 选项卡
4. 上传 `deploy_to_pythonanywhere.sh` 部署脚本到您的主目录

### 第二步：运行部署脚本

1. 点击页面顶部的 **Consoles** 选项卡
2. 选择 **Bash** 控制台
3. 在控制台中执行以下命令：
   ```bash
   # 添加执行权限
   chmod +x deploy_to_pythonanywhere.sh
   
   # 执行部署脚本
   ./deploy_to_pythonanywhere.sh
   ```
4. 脚本将自动从 GitHub 克隆仓库并配置必要的文件

### 第三步：设置 Web 应用

1. 点击页面顶部的 **Web** 选项卡
2. 点击 **Add a new web app** 按钮
3. 在向导中选择以下选项：
   - 域名：保持默认（wz2025.pythonanywhere.com）
   - Python 版本：选择 3.8 或更高版本
   - 框架：选择 Flask
   - 路径设置：输入 `/home/WZ2025/wz250323001`

### 第四步：配置静态文件

1. 在 **Web** 选项卡的 **Static files** 部分：
2. 点击 **Enter URL** 并输入 `/static/`
3. 点击 **Enter path** 并输入 `/home/WZ2025/wz250323001/web_app/static`
4. 点击 **Add** 按钮

### 第五步：重新加载应用

1. 在 **Web** 选项卡上，点击 **Reload** 按钮
2. 等待几秒钟后，您的应用应该已经部署完成

### 第六步：访问应用

您的应用现在应该可以通过以下URL访问：
- http://wz2025.pythonanywhere.com

## 自动更新设置（可选）

如果您需要定期从 GitHub 更新代码，可以设置每日任务：

1. 在 **Consoles** 选项卡中选择 **Schedule**
2. 添加一个每日任务，命令如下：
   ```bash
   cd ~/wz250323001 && git pull && pip3 install --user -r requirements.txt && touch /var/www/wz2025_pythonanywhere_com_wsgi.py
   ```

## 常见问题解决

如果您的应用没有正确部署或无法访问，请检查：

1. **错误日志**：在 Web 选项卡中，点击 **Error log** 链接查看错误信息
2. **依赖问题**：确保所有必要的依赖已安装（可通过控制台手动安装缺失的依赖）
   ```bash
   pip3 install --user <package_name>
   ```
3. **文件权限**：确保文件有正确的读取权限
   ```bash
   chmod -R 755 /home/WZ2025/wz250323001
   ```
4. **WSGI 配置**：检查 WSGI 文件是否正确配置（点击 Web 选项卡中的 WSGI 配置文件链接） 