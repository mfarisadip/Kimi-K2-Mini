#\!/bin/bash
# Kimi-K2-Mini COCONUT快速开始脚本

echo "🚀 Kimi-K2-Mini COCONUT 快速开始"
echo "================================"

# 1. 检查环境
echo "
1. 检查环境..."
python test_coconut_integration.py

# 2. 准备数据
echo "
2. 准备训练数据..."
python -c "from train_coconut_lora import prepare_gsm8k_data; prepare_gsm8k_data()"

# 3. 开始LoRA微调（演示模式，实际训练需要更多数据）
echo "
3. 开始LoRA微调..."
echo "命令: python train_coconut_lora.py --epochs 1 --batch_size 1"
echo "(实际训练请使用: python train_coconut_lora.py --epochs 3 --batch_size 4)"

# 4. 测试推理
echo "
4. 测试推理..."
echo "命令: python train_coconut_lora.py --test_only"

echo "
✅ 设置完成！"
echo "
下一步:")
echo "1. 下载完整GSM8K数据集"
echo "2. 运行完整训练: python train_coconut_lora.py"
echo "3. 使用cloudexe在H100上运行获得最佳性能"
