#\!/usr/bin/env python3
import json
import os

model_path = '/root/Kimi-K2-Mini/k2-mini'

print('📊 K2-Mini Model Information')
print('='*50)

# 读取配置
config_path = os.path.join(model_path, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

print('\nModel Configuration:')
print(f'- Model type: {config.get("model_type", "unknown")}')
print(f'- Hidden size: {config.get("hidden_size", "unknown")}')
print(f'- Number of layers: {config.get("num_hidden_layers", "unknown")}')
print(f'- Vocab size: {config.get("vocab_size", "unknown")}')
print(f'- Number of experts: {config.get("n_routed_experts", "unknown")}')
print(f'- Shared experts: {config.get("n_shared_experts", "unknown")}')
print(f'- Top-k experts: {config.get("num_experts_per_tok", "unknown")}')

# 检查模型文件
print('\nModel Files:')
import glob
safetensor_files = glob.glob(os.path.join(model_path, '*.safetensors'))
total_size = 0
for f in sorted(safetensor_files):
    size = os.path.getsize(f) / (1024**3)
    total_size += size
    print(f'- {os.path.basename(f)}: {size:.2f} GB')
print(f'\nTotal model size: {total_size:.2f} GB')

# 检查索引文件
index_path = os.path.join(model_path, 'model.safetensors.index.json')
if os.path.exists(index_path):
    with open(index_path, 'r') as f:
        index = json.load(f)
    print(f'\nTotal weight entries: {len(index["weight_map"])}')

print('\n✅ Model information retrieved successfully\!')
