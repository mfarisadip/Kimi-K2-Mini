#\!/usr/bin/env python3
"""
K2-Mini 快速测试脚本
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 测试各个模块
print("=== K2-Mini 项目测试 ===")
print("\n1. 测试项目结构...")

# 检查文件
required_files = [
    "README.md",
    "requirements.txt", 
    "configs/k2_mini_config.json",
    "src/expert_selector.py",
    "src/quantization.py",
    "src/inference.py",
    "scripts/analyze_layers.py",
    "scripts/convert_to_mini.py",
    "scripts/quantize.py"
]

all_exist = True
for file in required_files:
    path = Path(file)
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  [{status}] {file}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✓ 所有文件都已创建")
else:
    print("\n✗ 某些文件缺失")

# 测试模块导入
print("\n2. 测试模块导入...")
try:
    from src.expert_selector import ExpertSelector, test_expert_selector
    print("  ✓ expert_selector 导入成功")
    
    from src.quantization import K2MiniQuantizer, test_quantization
    print("  ✓ quantization 导入成功")
    
    from src.inference import K2MiniInference, test_inference
    print("  ✓ inference 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")

# 运行模块测试
print("\n3. 运行模块测试...")
print("\n--- Expert Selector 测试 ---")
test_expert_selector()

print("\n--- Quantization 测试 ---")
test_quantization()

print("\n--- Inference 测试 ---")
test_inference()

# 显示使用说明
print("\n=== 使用说明 ===")
print("\n1. 分析原始K2模型:")
print("   python scripts/analyze_layers.py --model-path /root/kimi-k2-instruct")

print("\n2. 转换为K2-Mini:")
print("   python scripts/convert_to_mini.py --source-model /root/kimi-k2-instruct --output-path ./k2-mini")

print("\n3. (可选) 量化模型:")
print("   python scripts/quantize.py --model-path ./k2-mini --output-path ./k2-mini-int8 --quantization int8")

print("\n4. 使用模型:")
print("""   from src.inference import K2MiniInference
   
   # 加载模型
   engine = K2MiniInference("./k2-mini")
   
   # 生成文本
   response = engine.generate("你好，请介绍一下自己")
   print(response)
""")

print("\n项目测试完成！")
