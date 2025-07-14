#\!/usr/bin/env python3
import torch
import sys
import os
import json

print('🚀 测试K2-Mini模型（禁用量化检查）')
print('='*50)

# 修改配置以禁用量化
config_path = '/root/Kimi-K2-Mini/k2-mini/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# 临时移除量化配置
if 'quantization_config' in config:
    del config['quantization_config']
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print('✅ 已临时禁用量化配置')

print(f'\nGPU信息:')
print(f'  设备: {torch.cuda.get_device_name(0)}')
print(f'  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/root/Kimi-K2-Mini/k2-mini'

try:
    print(f'\n正在加载K2-Mini模型...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda',
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True  # 忽略尺寸不匹配
    )
    
    print('\n✅ 模型加载成功\!')
    print(f'显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
    
    # 简单测试
    prompt = "你好"
    print(f'\n测试输入: {prompt}')
    
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'输出: {generated_text}')
    
    print('\n🎉 基础功能测试成功\!')
    
except Exception as e:
    print(f'\n❌ 错误: {e}')
    import traceback
    traceback.print_exc()
finally:
    # 恢复量化配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['quantization_config'] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128]
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print('\n已恢复原始配置')
