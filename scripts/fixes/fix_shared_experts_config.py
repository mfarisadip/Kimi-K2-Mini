#\!/usr/bin/env python3
import json

print('🔧 修复共享专家配置...')

# 读取配置
with open('k2-mini/config.json', 'r') as f:
    config = json.load(f)

print(f'原始配置:')
print(f'  n_shared_experts: {config.get("n_shared_experts", "未设置")}')
print(f'  n_routed_experts: {config.get("n_routed_experts", "未设置")}')

# 修改配置 - 禁用共享专家
config['n_shared_experts'] = 0

# 保存修改
with open('k2-mini/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f'\n修改后配置:')
print(f'  n_shared_experts: {config.get("n_shared_experts")}')
print(f'  n_routed_experts: {config.get("n_routed_experts")}')

print('\n✅ 共享专家配置已修复\!')
