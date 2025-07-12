#\!/usr/bin/env python3
import torch
import json
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import shutil

print('🔧 全面修复K2-Mini权重文件...')

model_path = Path('k2-mini')

# 加载索引
with open(model_path / 'model.safetensors.index.json', 'r') as f:
    index = json.load(f)

# 需要修复的权重模式
fix_patterns = {
    'gate.weight': (384, 16),  # 从384个专家到16个
    'gate.e_score_correction_bias': (384, 16),  # 已经修复过，但再检查一次
}

# 找出需要修复的权重
weights_to_fix = {}
for key in index['weight_map']:
    for pattern, (old_dim, new_dim) in fix_patterns.items():
        if pattern in key:
            if key not in weights_to_fix:
                weights_to_fix[key] = (old_dim, new_dim)

print(f'\n找到 {len(weights_to_fix)} 个需要检查的权重')

# 按文件分组
files_to_fix = {}
for key, dims in weights_to_fix.items():
    file_name = index['weight_map'][key]
    if file_name not in files_to_fix:
        files_to_fix[file_name] = []
    files_to_fix[file_name].append((key, dims))

# 修复每个文件
fixed_count = 0
for file_name, keys_dims in files_to_fix.items():
    print(f'\n检查 {file_name}...')
    file_path = model_path / file_name
    
    # 加载所有权重
    weights = {}
    needs_save = False
    
    with safe_open(file_path, framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # 检查是否需要修复
            need_fix = False
            for fix_key, (old_dim, new_dim) in keys_dims:
                if key == fix_key:
                    # 检查形状
                    if tensor.shape[0] == old_dim:
                        print(f'  修复 {key}: {tensor.shape} -> ', end='')
                        # 截取前new_dim个
                        tensor = tensor[:new_dim]
                        print(f'{tensor.shape}')
                        needs_save = True
                        fixed_count += 1
                    else:
                        print(f'  ✓ {key} 已经是正确形状: {tensor.shape}')
                    break
            
            weights[key] = tensor
    
    # 如果有修改，保存文件
    if needs_save:
        save_file(weights, file_path)
        print(f'  ✅ {file_name} 已更新')
    else:
        print(f'  ✓ {file_name} 无需修改')

print(f'\n✅ 修复完成！共修复 {fixed_count} 个权重')

# 验证修复结果
print('\n验证权重形状...')
with open(model_path / 'model.safetensors.index.json', 'r') as f:
    index = json.load(f)

sample_keys = ['model.layers.1.mlp.gate.weight', 'model.layers.1.mlp.gate.e_score_correction_bias']
for key in sample_keys:
    if key in index['weight_map']:
        file_name = index['weight_map'][key]
        with safe_open(model_path / file_name, framework='pt') as f:
            if key in f.keys():
                tensor = f.get_tensor(key)
                print(f'  {key}: {tensor.shape}')
