#\!/usr/bin/env python3

def main():
    from vllm import LLM, SamplingParams
    import torch
    import time
    
    print('🚀 CloudExe H100 上运行 K2-Mini 测试')
    print('=====================================')
    
    # 显示GPU信息
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'初始显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
    
    model_path = '/root/Kimi-K2-Mini/k2-mini'
    
    try:
        print(f'\n正在加载 K2-Mini 模型...')
        start_time = time.time()
        
        # 使用完整GPU资源配置
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype='float16',
            gpu_memory_utilization=0.85,  # 使用85%显存
            max_model_len=4096,           # 较长序列
            tensor_parallel_size=1,
            disable_custom_all_reduce=True,
            quantization=None,
            seed=42
        )
        
        load_time = time.time() - start_time
        print(f'✅ 模型加载成功！耗时: {load_time:.1f}秒')
        print(f'加载后显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
        
        # 设置生成参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=150,
            stop=['\n\n', '\n问：', '\n答：']
        )
        
        # 测试提示
        prompts = [
            '人工智能的发展历史可以追溯到',
            '机器学习和深度学习的主要区别是',
            '请解释什么是Transformer架构：'
        ]
        
        print(f'\n📝 开始生成测试...')
        
        for i, prompt in enumerate(prompts, 1):
            print(f'\n--- 测试 {i} ---')
            print(f'输入: {prompt}')
            
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            gen_time = time.time() - start_time
            
            output = outputs[0]
            generated_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            
            print(f'输出: {generated_text}')
            print(f'生成速度: {tokens_generated/gen_time:.1f} tokens/秒')
            print(f'耗时: {gen_time:.2f}秒')
        
        print(f'\n🎉 K2-Mini 在 CloudExe H100 上运行成功！')
        print(f'\n📊 性能总结:')
        print(f'  模型: K2-Mini (32.5B参数)')
        print(f'  GPU: H100 80GB') 
        print(f'  显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
        print(f'  加载时间: {load_time:.1f}秒')
        
    except Exception as e:
        print(f'\n❌ 错误: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
