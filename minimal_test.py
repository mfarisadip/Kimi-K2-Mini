import torch
import sys
import os

print('🚀 K2-Mini 最小化测试')
print('='*50)

# 检查GPU
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ 没有找到GPU')
    sys.exit(1)

# 尝试加载模型
model_path = 'k2-mini'
print(f'\n尝试从 {model_path} 加载模型...')

try:
    # 首先只加载tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print('✅ Tokenizer 加载成功')
    
    # 检查模型文件
    import json
    with open(os.path.join(model_path, 'config.json')) as f:
        config = json.load(f)
    print(f'✅ 模型配置加载成功')
    print(f'   架构: {config.get("architectures", ["Unknown"])[0]}')
    
    # 尝试使用vLLM加载（可能更适合这种大模型）
    print('\n尝试使用vLLM加载模型...')
    try:
        from vllm import LLM, SamplingParams
        
        # 使用较小的GPU内存限制
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype='float16',
            gpu_memory_utilization=0.7,
            max_model_len=2048,
            enforce_eager=True  # 禁用CUDA图以避免某些错误
        )
        
        print('✅ vLLM模型加载成功\!')
        
        # 简单测试
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        prompt = "什么是人工智能？"
        
        print(f'\n测试生成...')
        print(f'输入: {prompt}')
        
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        print(f'输出: {generated_text}')
        print('\n✅ K2-Mini 运行成功\!')
        
    except Exception as e:
        print(f'\n❌ vLLM加载失败: {e}')
        print('\n尝试使用transformers直接加载...')
        
        # 尝试不使用device_map
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None  # 不使用自动设备映射
        )
        model = model.to('cuda')
        print('✅ Transformers模型加载成功\!')
        
except Exception as e:
    print(f'\n❌ 错误: {e}')
    import traceback
    traceback.print_exc()
