import torch
import json
import os
from pathlib import Path
from safetensors import safe_open

print('🔍 K2-Mini 模型状态检查')
print('='*50)

model_path = Path('k2-mini')

# 1. 检查文件
print('\n1. 模型文件:')
for f in sorted(model_path.glob('*.safetensors')):
    size_gb = f.stat().st_size / (1024**3)
    print(f'  {f.name}: {size_gb:.1f} GB')

# 2. 检查配置
print('\n2. 模型配置:')
with open(model_path / 'config.json') as f:
    config = json.load(f)
    print(f'  架构: {config.get("architectures", ["Unknown"])[0]}')
    print(f'  层数: {config.get("num_hidden_layers")}')
    print(f'  路由专家数: {config.get("n_routed_experts")}')
    print(f'  共享专家数: {config.get("n_shared_experts")}')
    print(f'  隐藏维度: {config.get("hidden_size")}')
    print(f'  每个token使用专家数: {config.get("num_experts_per_tok")}')

# 3. 检查权重索引
print('\n3. 权重分布:')
with open(model_path / 'model.safetensors.index.json') as f:
    index = json.load(f)
    weight_map = index['weight_map']
    
    # 统计每个文件的权重数
    file_counts = {}
    for weight_name, file_name in weight_map.items():
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
    
    for file_name, count in sorted(file_counts.items()):
        print(f'  {file_name}: {count} 个权重')

# 4. 检查关键权重
print('\n4. 检查关键权重存在性:')
key_patterns = [
    'model.embed_tokens.weight',
    'model.layers.0.self_attn.q_proj.weight',
    'model.layers.0.mlp.gate.weight',
    'model.layers.0.mlp.experts.0.down_proj.weight',
    'model.layers.0.mlp.shared_experts.down_proj.weight',
    'lm_head.weight'
]

for pattern in key_patterns:
    exists = any(pattern in k for k in weight_map.keys())
    status = '✓' if exists else '✗'
    print(f'  {status} {pattern}')

# 5. 检查共享专家权重
print('\n5. 共享专家权重状态:')
shared_expert_weights = [k for k in weight_map.keys() if 'shared_experts' in k]
print(f'  找到 {len(shared_expert_weights)} 个共享专家权重')
if shared_expert_weights:
    print('  示例:')
    for weight in shared_expert_weights[:3]:
        print(f'    - {weight}')

print('\n检查完成\!')
