#\!/usr/bin/env python3
from vllm import LLM, SamplingParams
import torch

print("\n🚀 vLLM K2-Mini 测试\n")

model_path = "./k2-mini"

try:
    # 检查GPU信息
    print(f"GPU信息:")
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n正在使用vLLM加载K2-Mini模型...")
    
    # 创建LLM实例（使用较小的GPU内存利用率）
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.8,
        max_model_len=2048,  # 限制序列长度以节省内存
        tensor_parallel_size=1,
        seed=42
    )
    
    print(f"✅ vLLM模型加载成功！")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        stop=["\n\n"]  # 简单的停止条件
    )
    
    # 测试提示
    prompts = [
        "人工智能的发展历史",
        "Python编程语言的特点",
        "什么是机器学习？"
    ]
    
    print(f"\n📝 开始生成测试...")
    
    # 批量生成
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"\n生成结果:")
    print(f"=" * 50)
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n提示 {i+1}: {prompt}")
        print(f"生成: {generated_text}")
        print(f"-" * 30)
    
    print(f"\n🎉 vLLM测试完成！模型运行正常！")
    
except Exception as e:
    print(f"\n❌ vLLM错误: {e}")
    import traceback
    traceback.print_exc()
    
    if "CUDA out of memory" in str(e):
        print(f"\n💡 解决方案:")
        print(f"1. 当前MIG实例显存不足(10GB)")
        print(f"2. 需要完整H100 GPU (80GB)来运行32.5B模型")
        print(f"3. 或使用更小的模型配置")
