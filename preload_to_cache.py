#\!/usr/bin/env python3
"""预加载模型文件到内存缓存"""
import os
from pathlib import Path
from tqdm import tqdm

model_path = Path('/root/kimi-k2-instruct')
total_size = 0

print('预加载 Kimi-K2 模型到内存缓存...')
print('这会利用 Linux 页面缓存机制')

# 找出所有 safetensor 文件
files = list(model_path.glob('*.safetensors'))
print(f'\n找到 {len(files)} 个模型文件')

# 预读文件到内存
for file in tqdm(files, desc='加载到内存'):
    size = file.stat().st_size
    total_size += size
    
    # 读取文件（会被缓存到内存）
    with open(file, 'rb') as f:
        # 每次读 100MB
        while f.read(100 * 1024 * 1024):
            pass

print(f'\n✓ 已加载 {total_size / 1024**3:.1f} GB 到内存缓存')
print('后续访问这些文件将从内存读取，速度提升 100x+')
