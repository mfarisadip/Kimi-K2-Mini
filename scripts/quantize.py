#\!/usr/bin/env python3
"""
K2-Mini 量化脚本
"""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.quantization import K2MiniQuantizer


def main():
    parser = argparse.ArgumentParser(description='量化 K2-Mini 模型')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='K2-Mini 模型路径'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='量化模型输出路径'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        choices=['int8', 'int4', 'mixed'],
        default='int8',
        help='量化类型 (默认: int8)'
    )
    parser.add_argument(
        '--device-map',
        type=str,
        default='auto',
        help='设备映射 (默认: auto)'
    )
    
    args = parser.parse_args()
    
    # 创建量化器
    quantizer = K2MiniQuantizer(args.model_path)
    
    # 执行量化
    if args.quantization == 'int8':
        quantizer.quantize_to_int8(args.output_path, args.device_map)
    elif args.quantization == 'int4':
        quantizer.quantize_to_int4(args.output_path, args.device_map)
    elif args.quantization == 'mixed':
        quantizer.create_mixed_precision_model(args.output_path)
    
    print("\n量化完成！")


if __name__ == '__main__':
    main()
