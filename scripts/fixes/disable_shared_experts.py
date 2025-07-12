import json
import shutil
import os

print('🔧 禁用共享专家以修复模型加载问题')

# 备份原始配置
config_path = 'k2-mini/config.json'
backup_path = 'k2-mini/config_backup.json'

with open(config_path, 'r') as f:
    config = json.load(f)

# 保存备份
with open(backup_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f'原始配置:')
print(f'  n_shared_experts: {config.get("n_shared_experts", 0)}')
print(f'  n_routed_experts: {config.get("n_routed_experts", 0)}')

# 修改配置 - 禁用共享专家
config['n_shared_experts'] = 0

# 保存修改后的配置
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f'\n修改后配置:')
print(f'  n_shared_experts: {config.get("n_shared_experts", 0)}')
print(f'  n_routed_experts: {config.get("n_routed_experts", 0)}')

print('\n✅ 配置已更新\!')
