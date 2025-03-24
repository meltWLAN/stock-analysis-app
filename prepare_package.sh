#!/bin/bash
# 准备部署包 - 在本地执行

# 提交更改到Git
echo "===== 提交更改到Git ====="
git add .
git commit -m "更新部署配置"
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
echo "# 安装依赖"
echo "cd wz250323001"
echo "pip3 install --user -r requirements.txt"
echo ""
echo "# 重新加载Web应用（在Web选项卡中点击Reload按钮）"
echo ""
echo "===== 完成 =====" 