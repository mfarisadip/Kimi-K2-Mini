#\!/usr/bin/env python3
import torch
import sys
import os

print('🚀 测试修复后的K2-Mini模型')
print('='*50)

# 这个脚本将通过CloudExe在远程H100上运行
print(f'\nGPU信息:')
print(f'  设备: {torch.cuda.get_device_name(0)}')
print(f'  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/root/Kimi-K2-Mini/k2-mini'

try:
    print(f'\n正在加载K2-Mini模型...')
    print('  加载tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print('  加载模型权重...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda',
        low_cpu_mem_usage=True
    )
    
    print('\n✅ 模型加载成功\!')
    print(f'显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
    
    # 测试生成
    test_prompts = [
        "什么是人工智能？",
        "写一个Python函数计算斐波那契数列：",
        "Explain machine learning in simple terms:"
    ]
    
    print('\n📝 测试文本生成...')
    for i, prompt in enumerate(test_prompts, 1):
        print(f'\n--- 测试 {i} ---')
        print(f'输入: {prompt}')
        
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'输出: {generated_text[:200]}...')  # 只显示前200个字符
    
    # 性能总结
    print(f'\n📊 性能总结:')
    print(f'  模型: K2-Mini (32.5B参数)')
    print(f'  GPU: H100 80GB')
    print(f'  显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
    print(f'  推理状态: 正常')
    
    print('\n🎉 K2-Mini在CloudExe H100上成功运行\!')
    
except Exception as e:
    print(f'\n❌ 错误: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
