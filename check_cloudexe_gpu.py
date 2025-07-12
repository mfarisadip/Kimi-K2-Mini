#\!/usr/bin/env python3
import torch

print('=== CloudExe GPU 信息 ===')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'显存总量: {props.total_memory / 1024**3:.1f} GB')
    print(f'计算能力: {props.major}.{props.minor}')
    print(f'多处理器数: {props.multi_processor_count}')
else:
    print('GPU不可用')
