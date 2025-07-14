#\!/usr/bin/env python3
import sys
import os

print('🚀 在CloudExe H100上运行K2-Mini')
print('='*50)

# 这个脚本将通过CloudExe在远程H100上运行
import torch
print(f'\nGPU信息:')
print(f'  设备: {torch.cuda.get_device_name(0)}')
print(f'  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'  可用显存: {torch.cuda.memory_available() / 1024**3:.1f} GB')

# 尝试加载K2-Mini
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/root/Kimi-K2-Mini/k2-mini'

try:
    print(f'\n正在加载K2-Mini模型...')
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 使用device_map='cuda'直接加载到GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda',
        low_cpu_mem_usage=True
    )
    
    print('✅ 模型加载成功\!')
    print(f'显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
    
    # 测试生成
    prompt = "人工智能的未来发展方向是"
    print(f'\n测试生成...')
    print(f'输入: {prompt}')
    
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'\n输出: {generated_text}')
    
    print('\n✅ K2-Mini在CloudExe H100上成功运行\!')
    
except Exception as e:
    print(f'\n❌ 错误: {e}')
    import traceback
    traceback.print_exc()
