#\!/usr/bin/env python3

def main():
    from vllm import LLM, SamplingParams
    import torch
    import os
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("\n🚀 vLLM K2-Mini 测试 (修复版)\n")
    
    model_path = "./k2-mini"
    
    try:
        # 检查GPU信息
        print(f"GPU信息:")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print(f"\n正在使用vLLM加载K2-Mini模型...")
        print(f"  模型路径: {model_path}")
        
        # 创建LLM实例（更保守的设置）
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.6,  # 降低内存使用
            max_model_len=1024,          # 更小的序列长度
            tensor_parallel_size=1,
            enforce_eager=True,          # 禁用CUDA图优化
            disable_custom_all_reduce=True,
            quantization=None,           # 禁用量化
            seed=42
        )
        
        print(f"✅ vLLM模型加载成功！")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,  # 减少生成长度
            stop=["\n"]
        )
        
        # 简单测试
        prompt = "人工智能是"
        
        print(f"\n📝 测试生成: {prompt}")
        
        # 生成
        outputs = llm.generate([prompt], sampling_params)
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        print(f"\n生成结果:")
        print(f"输入: {prompt}")
        print(f"输出: {generated_text}")
        
        print(f"\n🎉 vLLM测试成功！K2-Mini模型运行正常！")
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "NVML_SUCCESS" in str(e):
            print(f"\n❌ GPU内存不足错误: {e}")
            print(f"\n💡 解决方案:")
            print(f"1. 当前MIG实例只有10GB显存，不足以运行32.5B模型")
            print(f"2. 需要完整的H100 GPU (80GB)进行推理")
            print(f"3. 或者使用CPU推理（速度会很慢）")
        else:
            print(f"\n❌ 其他CUDA错误: {e}")
            
    except Exception as e:
        print(f"\n❌ 其他错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
