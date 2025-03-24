#!/bin/bash
# 准备部署包 - 在本地执行

# 提交更改到Git
echo "===== 提交更改到Git ====="
git add .
git commit -m "更新依赖配置，降低依赖版本要求"
git push origin main

echo "===== 创建PythonAnywhere部署命令 ====="
echo "登录PythonAnywhere控制台后，运行以下命令："
echo ""
echo "# 清理旧代码"
echo "cd ~"
echo "rm -rf wz250323001"
echo ""
echo "# 克隆最新代码"
echo "git clone https://github.com/meltWLAN/stock-analysis-app.git wz250323001"
echo ""
echo "# 分批安装依赖以避免内存问题"
echo "cd wz250323001"
echo "pip3 install --user flask==2.3.3 Werkzeug==2.3.7 Jinja2==3.1.2 MarkupSafe==2.1.3 itsdangerous==2.1.2 click==8.1.7"
echo "pip3 install --user numpy>=1.20.0,<1.24.0 matplotlib>=3.4.0 plotly>=5.5.0"
echo "pip3 install --user pandas>=1.5.0,<2.0.0 scipy>=1.7.0"
echo "pip3 install --user gunicorn==21.2.0 pyecharts>=1.9.0"
echo "pip3 install --user scikit-learn>=1.0.0,<1.3.0"
echo ""
echo "# 重新加载Web应用（在Web选项卡中点击Reload按钮）"
echo ""
echo "===== 完成 =====" 