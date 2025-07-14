#\!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

print('🚀 K2-Mini Basic Test')
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
    
    # 加载模型 - 使用更保守的设置
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda',
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    print(f'✅ Model loaded in {load_time:.1f} seconds')
    
    # 获取显存使用
    memory_used = torch.cuda.memory_allocated() / 1024**3
    print(f'Memory used: {memory_used:.1f} GB')
    
    # 简单的tokenize测试
    print('\nTesting tokenizer...')
    test_text = 'Hello, world\! 你好，世界！'
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f'Original: {test_text}')
    print(f'Encoded: {tokens[:10]}... (length: {len(tokens)})')
    print(f'Decoded: {decoded}')
    
    # 检查模型结构
    print('\nModel structure:')
    print(f'- Number of layers: {len(model.model.layers)}')
    print(f'- Hidden size: {model.config.hidden_size}')
    print(f'- Vocab size: {model.config.vocab_size}')
    print(f'- Experts per layer: {model.config.n_routed_experts}')
    
    print('\n✅ Basic test completed successfully\!')
    
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
