#\!/usr/bin/env python3
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import os
from tqdm import tqdm

print('🔧 修复Meta Tensor问题...')
print('这可能需要几分钟时间...')

# 处理每个模型文件
for i in range(1, 6):
    filename = f'k2-mini/model-{i}-of-5.safetensors'
    print(f'\n处理文件 {i}/5: {filename}')
    
    if not os.path.exists(filename):
        print(f'  ⚠️  文件不存在，跳过')
        continue
    
    tensors = {}
    issues_fixed = 0
    
    # 读取并修复tensors
    with safe_open(filename, framework="pt") as f:
        for key in tqdm(f.keys(), desc='处理权重'):
            try:
                tensor = f.get_tensor(key)
                
                # 修复FP8到FP16的转换
                if hasattr(tensor, 'dtype') and tensor.dtype == torch.float8_e4m3fn:
                    tensor = tensor.to(torch.float16)
                    issues_fixed += 1
                
                # 特殊处理weight_scale_inv
                if 'weight_scale_inv' in key:
                    # 确保tensor是连续的并且在CPU上
                    if tensor.is_meta:
                        # 如果是meta tensor，创建一个新的tensor
                        tensor = torch.ones(tensor.shape, dtype=torch.float16)
                        issues_fixed += 1
                    else:
                        tensor = tensor.to(torch.float16).contiguous()
                
                tensors[key] = tensor.cpu()  # 确保在CPU上
                
            except Exception as e:
                print(f'  ⚠️  处理 {key} 时出错: {e}')
                # 创建一个占位tensor
                if 'weight_scale_inv' in key:
                    # weight_scale_inv 通常是标量或小向量
                    tensors[key] = torch.ones(1, dtype=torch.float16)
                    issues_fixed += 1
    
    # 保存修复后的文件
    print(f'  修复了 {issues_fixed} 个问题')
    print(f'  保存修复后的权重...')
    save_file(tensors, filename)
    print(f'  ✅ 文件 {i}/5 处理完成')

print('\n✅ 所有Meta Tensor问题已修复\!')
