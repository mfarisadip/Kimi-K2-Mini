#\!/bin/bash
# 使用CloudExe运行COCONUT训练脚本

# 设置API key
export CLOUDEXE_APIKEY='eyJleHBpcnlfZGF0ZSI6IjIwMjYtMDctMTUgMDA6MDA6MDAiLCJ1c2VyaWQiOiJjOGI5NmUxZS0xODVkLTRkNDUtOTY3Mi0xYTVmZTVjYjc0NGUifQ==.FlhyVUxVixEBH0tM1zakUokaIAIV3Wq4ydR-pVY5XrM='

echo '🚀 使用CloudExe运行COCONUT训练'
echo '================================'

# 检查cloudexe是否可用
if \! command -v cloudexe &> /dev/null; then
    echo '❌ cloudexe未找到，请先安装'
    exit 1
fi

echo '✅ CloudExe已安装'
cloudexe --version

# 运行速度测试
echo ''
echo '运行网络速度测试...'
cloudexe --speedtest

# 准备训练命令
TRAINING_CMD='/usr/bin/python3 /root/Kimi-K2-Mini/train_coconut_lora.py --epochs 1 --batch_size 1 --lr 2e-4'

echo ''
echo '开始使用CloudExe运行训练...'
echo 'GPU规格: H100x1'
echo "命令: $TRAINING_CMD"

# 使用CloudExe运行训练
cloudexe --gpuspec H100x1 --pythonpath /usr/lib/python3.10 --printstats 10 -- $TRAINING_CMD

echo ''
echo '✅ 训练完成！'
