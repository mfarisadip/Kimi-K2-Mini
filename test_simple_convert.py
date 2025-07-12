#\!/usr/bin/env python3
"""简单测试转换流程"""
import json
from pathlib import Path

# 创建测试配置
def test_conversion_setup():
    print("测试转换设置...")
    
    # 1. 加载源配置
    with open('/root/kimi-k2-instruct/config.json', 'r') as f:
        config = json.load(f)
    
    print(f"源模型: {config['num_hidden_layers']} 层, {config['n_routed_experts']} 专家/层")
    
    # 2. 模拟选择层和专家
    selected_layers = [0, 10, 20, 30, 40, 50]  # 选6层做测试
    selected_experts = list(range(8))  # 每层选8个专家
    
    # 3. 计算大小
    params_per_expert_mb = 84  # 从之前的输出得知
    base_size_gb = 70.6 / 61 * len(selected_layers)  # 按比例缩放
    expert_size_gb = params_per_expert_mb * len(selected_experts) * len(selected_layers) / 1024
    total_size_gb = base_size_gb + expert_size_gb
    
    print(f"\n测试配置:")
    print(f"  选择层数: {len(selected_layers)}")
    print(f"  每层专家: {len(selected_experts)}")
    print(f"  预计大小: {total_size_gb:.1f} GB")
    print(f"  选择的层: {selected_layers}")
    
    return selected_layers, selected_experts

if __name__ == '__main__':
    test_conversion_setup()
