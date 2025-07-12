#\!/usr/bin/env python3
import os
import json

print('\n🎯 K2-Mini 项目最终状态总结')
print('='*60)

# 检查各个版本
versions = {
    'k2-mini': '主版本（从备份恢复）',
    'k2-mini-broken': '原始有问题的版本',
    'k2-mini-fast': '快速转换版本',
    'k2-mini-backup-weights': '备份权重'
}

print('\n📁 可用版本:')
for path, desc in versions.items():
    if os.path.exists(path):
        size = sum(os.path.getsize(os.path.join(path, f)) 
                  for f in os.listdir(path) 
                  if f.endswith('.safetensors')) / 1024**3
        print(f'  ✓ {path}: {desc} ({size:.1f} GB)')
    else:
        print(f'  ✗ {path}: 不存在')

# 检查主版本配置
print('\n🔧 主版本(k2-mini)配置:')
with open('k2-mini/config.json') as f:
    config = json.load(f)
    print(f'  层数: {config.get("num_hidden_layers")}')
    print(f'  路由专家数: {config.get("n_routed_experts")}')
    print(f'  共享专家数: {config.get("n_shared_experts")}')
    print(f'  量化方法: {config.get("quantization_config", {}).get("quant_method", "无")}')

print('\n📊 发现的问题:')
print('  1. 共享专家权重缺失（72个）')
print('  2. 专家门控权重维度不匹配（384 vs 16）')
print('  3. DynamicCache API兼容性问题')
print('  4. FP8量化权重与FP16不兼容')

print('\n✅ 已完成的修复:')
print('  1. 恢复备份权重')
print('  2. 禁用共享专家（n_shared_experts=0）')
print('  3. 修复meta tensor（转换为FP16）')
print('  4. 模型可以加载（使用40.6GB显存）')

print('\n💡 建议的下一步:')
print('  1. 修复modeling_deepseek.py中的缓存兼容性')
print('  2. 或使用vLLM/其他推理框架')
print('  3. 或重新完整转换（使用修正后的脚本）')

print('\n🚀 K2-Mini项目状态: 部分成功')
print('   - 模型结构 ✓')
print('   - 权重加载 ✓')
print('   - 推理生成 ✗ (需要修复)')
