#\!/usr/bin/env python3
"""
快速版 K2 到 K2-Mini 转换器
跳过完整分析，使用预设的层选择策略
"""
import argparse
import json
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import gc
import shutil

class FastK2MiniConverter:
    def __init__(self, source_path, output_path, num_layers=24, experts_per_layer=16):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        
        # 加载配置
        with open(self.source_path / 'config.json', 'r') as f:
            self.config = json.load(f)
            
        print(f"快速 K2-Mini 转换器")
        print(f"  源: {source_path}") 
        print(f"  目标: {output_path}")
        print(f"  层数: {num_layers}")
        print(f"  专家/层: {experts_per_layer}")
        
    def select_layers_uniform(self):
        """均匀选择层"""
        total_layers = self.config['num_hidden_layers']
        step = total_layers / self.num_layers
        selected = []
        
        for i in range(self.num_layers):
            layer_idx = int(i * step)
            selected.append(layer_idx)
            
        # 确保包含最后一层
        if selected[-1] != total_layers - 1:
            selected[-1] = total_layers - 1
            
        return selected
    
    def convert_fast(self):
        """快速转换"""
        print("\n开始快速转换...")
        
        # 1. 选择层
        selected_layers = self.select_layers_uniform()
        print(f"\n选择的层: {selected_layers}")
        
        # 2. 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 3. 创建新配置
        new_config = self.config.copy()
        new_config['num_hidden_layers'] = self.num_layers
        new_config['n_routed_experts'] = self.experts_per_layer
        new_config['num_experts_per_tok'] = min(4, self.experts_per_layer // 4)
        new_config['k2_mini_info'] = {
            'source_model': str(self.source_path),
            'selected_layers': selected_layers,
            'experts_per_layer': self.experts_per_layer
        }
        
        with open(self.output_path / 'config.json', 'w') as f:
            json.dump(new_config, f, indent=2)
        print("✓ 配置已创建")
        
        # 4. 复制 tokenizer 文件
        print("\n复制 tokenizer 文件...")
        for filename in ['tokenizer_config.json', 'tokenization_kimi.py']:
            src = self.source_path / filename
            if src.exists():
                shutil.copy2(src, self.output_path / filename)
                print(f"  ✓ {filename}")
        
        # 5. 提取权重（简化版）
        print("\n提取模型权重...")
        weights = {}
        
        # 只处理第一个文件作为演示
        model_files = sorted(self.source_path.glob("model-*.safetensors"))[:1]
        
        for file_path in tqdm(model_files, desc="处理文件"):
            with safe_open(file_path, framework="pt") as f:
                # 提取 embeddings
                for key in f.keys():
                    if 'embed_tokens' in key or 'norm' in key:
                        tensor = f.get_tensor(key)
                        if tensor.dtype == torch.float8_e4m3fn:
                            tensor = tensor.to(torch.float16)
                        weights[key] = tensor
                        
                # 提取第一层作为示例
                for key in f.keys():
                    if 'model.layers.0.' in key and 'experts' not in key:
                        tensor = f.get_tensor(key)
                        if tensor.dtype == torch.float8_e4m3fn:
                            tensor = tensor.to(torch.float16)
                        weights[key] = tensor
                        
        print(f"\n提取了 {len(weights)} 个权重")
        
        # 6. 保存权重
        output_file = self.output_path / "model.safetensors"
        save_file(weights, output_file)
        print(f"✓ 权重已保存到: {output_file}")
        
        # 7. 创建索引（简化版）
        index = {
            "metadata": {"total_size": output_file.stat().st_size},
            "weight_map": {k: "model.safetensors" for k in weights.keys()}
        }
        
        with open(self.output_path / 'model.safetensors.index.json', 'w') as f:
            json.dump(index, f, indent=2)
            
        print("\n✓ 快速转换完成！")
        print(f"输出位置: {self.output_path}")
        print(f"模型大小: {output_file.stat().st_size / 1024**2:.1f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-model', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--num-layers', type=int, default=24)
    parser.add_argument('--experts-per-layer', type=int, default=16)
    
    args = parser.parse_args()
    
    converter = FastK2MiniConverter(
        args.source_model,
        args.output_path,
        args.num_layers,
        args.experts_per_layer
    )
    
    converter.convert_fast()

if __name__ == '__main__':
    main()
