#\!/usr/bin/env python3
import torch
import json
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
import shutil

print('🔧 修复K2-Mini权重文件...')

model_path = Path('k2-mini')
backup_path = Path('k2-mini-backup-weights')

# 创建备份
if not backup_path.exists():
    print('创建权重备份...')
    shutil.copytree(model_path, backup_path)
    print('✅ 备份完成')

# 加载索引
with open(model_path / 'model.safetensors.index.json', 'r') as f:
    index = json.load(f)

# 找出需要修复的权重
bias_keys = [k for k in index['weight_map'] if 'e_score_correction_bias' in k]
print(f'\n找到 {len(bias_keys)} 个需要修复的权重')

# 按文件分组
files_to_fix = {}
for key in bias_keys:
    file_name = index['weight_map'][key]
    if file_name not in files_to_fix:
        files_to_fix[file_name] = []
    files_to_fix[file_name].append(key)

# 修复每个文件
for file_name, keys in files_to_fix.items():
    print(f'\n修复 {file_name}...')
    file_path = model_path / file_name
    
    # 加载所有权重
    weights = {}
    with safe_open(file_path, framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in keys:
                # 截断到16维
                print(f'  修复 {key}: {tensor.shape} -> torch.Size([16])')
                weights[key] = tensor[:16]
            else:
                weights[key] = tensor
    
    # 保存修复后的文件
    save_file(weights, file_path)
    print(f'  ✅ {file_name} 修复完成')

print('\n✅ 所有权重文件修复完成！')
