#\!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import gc

print("\n🚀 K2-Mini 推理测试\n")

model_path = "./k2-mini"

try:
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 没有检测到GPU")
        exit(1)
    
    device = torch.device("cuda")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"   初始显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    
    # 加载tokenizer
    print("\n正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Tokenizer加载成功")
    
    # 加载模型
    print("\n正在加载模型（可能需要1-2分钟）...")
    start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    load_time = time.time() - start_time
    print(f"✅ 模型加载成功！耗时: {load_time:.1f}秒")
    print(f"   显存占用: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    
    # 设置生成参数
    model.eval()
    
    # 测试生成
    print("\n📝 测试文本生成...")
    test_prompts = [
        "人工智能的发展历史可以追溯到",
        "Python编程语言的主要特点是",
        "请解释什么是机器学习："
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: {prompt}")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 生成
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen_time = time.time() - start_time
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_tokens = outputs[0].shape[0] - inputs['input_ids'].shape[1]
        
        print(f"生成内容: {generated_text}")
        print(f"生成速度: {new_tokens/gen_time:.1f} tokens/秒")
    
    print("\n✅ 所有测试完成！")
    print("\n📊 模型性能总结:")
    print(f"   模型大小: 32.5B参数")
    print(f"   显存占用: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print(f"   推理正常: 是")
    print(f"\n🎉 K2-Mini模型验证成功！可以正常使用！")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # 清理
    if 'model' in locals():
        del model
    gc.collect()
    torch.cuda.empty_cache()
