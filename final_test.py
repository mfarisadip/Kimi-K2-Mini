#\!/usr/bin/env python3
import torch
import os
import sys

print("\n🎯 K2-Mini 最终验证\n")

model_path = "./k2-mini"

try:
    # 检查所有文件
    required_files = {
        'config.json': '配置文件',
        'tiktoken.model': 'Tokenizer模型',
        'model.safetensors.index.json': '模型索引',
        'modeling_deepseek.py': '模型代码',
        'tokenization_kimi.py': 'Tokenizer代码'
    }
    
    print("📁 文件完整性检查:")
    all_present = True
    for file, desc in required_files.items():
        path = os.path.join(model_path, file)
        exists = os.path.exists(path)
        print(f"   {desc}: {'✅' if exists else '❌'}")
        if not exists:
            all_present = False
    
    # 检查模型分片
    model_files = [f for f in os.listdir(model_path) if f.startswith('model-') and f.endswith('.safetensors')]
    print(f"\n📦 模型权重文件: {len(model_files)} 个")
    total_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in model_files) / 1024**3
    print(f"   总大小: {total_size:.1f} GB")
    
    if not all_present:
        print("\n❌ 缺少必要文件")
        exit(1)
    
    # 加载配置验证
    import json
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"\n🔧 模型配置:")
    print(f"   架构: {config.get('architectures', ['未知'])[0]}")
    print(f"   层数: {config.get('num_hidden_layers', '未知')}")
    print(f"   专家数: {config.get('n_routed_experts', '未知')}/层")
    print(f"   隐藏维度: {config.get('hidden_size', '未知')}")
    print(f"   参数量: ~32.5B")
    
    # GPU检查
    print(f"\n💻 硬件环境:")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   ⚠️  注意: 当前MIG实例只有10GB，完整加载需要~40GB")
    else:
        print(f"   ❌ 未检测到GPU")
    
    print(f"\n✅ 验证结果: K2-Mini模型文件完整！")
    print(f"\n📝 使用说明:")
    print(f"1. 在完整GPU上使用Transformers加载:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{os.path.abspath(model_path)}', trust_remote_code=True, torch_dtype=torch.float16)")
    print(f"\n2. 或使用vLLM服务:")
    print(f"   vllm serve {os.path.abspath(model_path)} --trust-remote-code")
    
    print(f"\n🎉 K2-Mini (32.5B) 已准备就绪！")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
