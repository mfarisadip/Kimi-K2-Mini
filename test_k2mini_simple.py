#\!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

print('\n🚀 K2-Mini on CloudExe H100 Test')
print('='*50)

# 显示GPU信息
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

model_path = '/root/Kimi-K2-Mini/k2-mini'

try:
    print('\nLoading K2-Mini model...')
    start_time = time.time()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    print(f'✅ Model loaded in {load_time:.1f} seconds')
    print(f'Memory used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
    
    # 测试生成
    print('\nTesting generation...')
    
    prompt = '人工智能的发展历史'
    print(f'Prompt: {prompt}')
    
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    gen_time = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'\nGenerated: {generated_text}')
    print(f'Generation time: {gen_time:.2f}s')
    
    print('\n✅ K2-Mini runs successfully on CloudExe H100\!')
    
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
