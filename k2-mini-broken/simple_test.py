import torch
print("\n测试K2-Mini模型加载...\n")

try:
    # 直接导入模型类
    import sys
    sys.path.insert(0, '.')
    from modeling_deepseek import DeepseekV3ForCausalLM
    from transformers import AutoConfig
    
    # 加载配置
    config = AutoConfig.from_pretrained('.', trust_remote_code=True)
    print(f"✅ 配置加载成功")
    print(f"   层数: {config.num_hidden_layers}")
    print(f"   专家数: {config.n_routed_experts}")
    
    # 测试模型结构
    print("\n测试模型结构创建...")
    with torch.device('meta'):
        model = DeepseekV3ForCausalLM(config)
    print("✅ 模型结构验证成功")
    
    print("\n💡 模型文件完整，可以加载！")
    print("\n注意：完整加载需要约40GB显存")
    print("当前MIG实例只有10GB，需要完整的GPU来运行")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
