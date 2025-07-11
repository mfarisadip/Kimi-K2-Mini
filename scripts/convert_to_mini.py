#\!/usr/bin/env python3
"""
Kimi-K2 到 K2-Mini 转换器
将1.07T参数的K2模型转换为32.5B参数的K2-Mini
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import gc
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.expert_selector import ExpertSelector


class K2MiniConverter:
    """K2 到 K2-Mini 模型转换器"""
    
    def __init__(self, source_model_path: str, output_path: str,
                 num_layers: int = 24, experts_per_layer: int = 16):
        self.source_path = Path(source_model_path)
        self.output_path = Path(output_path)
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        
        # 加载源模型配置
        with open(self.source_path / 'config.json', 'r') as f:
            self.source_config = json.load(f)
            
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"K2-Mini 转换器初始化:")
        print(f"  源模型: {source_model_path}")
        print(f"  输出路径: {output_path}")
        print(f"  目标层数: {num_layers}")
        print(f"  每层专家数: {experts_per_layer}")
        
    def create_mini_config(self, selected_layers: List[int], 
                          layer_expert_mapping: Dict[int, List[int]]) -> Dict:
        """创建K2-Mini配置"""
        mini_config = self.source_config.copy()
        
        # 更新基本参数
        mini_config['model_type'] = 'kimi_k2_mini'
        mini_config['num_hidden_layers'] = self.num_layers
        mini_config['n_routed_experts'] = self.experts_per_layer
        
        # 减少每个token使用的专家数
        mini_config['num_experts_per_tok'] = min(4, self.experts_per_layer // 4)
        
        # 添加转换信息
        mini_config['k2_mini_info'] = {
            'original_model': str(self.source_path),
            'original_layers': self.source_config['num_hidden_layers'],
            'original_experts': self.source_config['n_routed_experts'],
            'selected_layers': selected_layers,
            'layer_expert_mapping': {
                str(k): v for k, v in layer_expert_mapping.items()
            },
            'compression_ratio': (
                self.source_config['num_hidden_layers'] * self.source_config['n_routed_experts'] /
                (self.num_layers * self.experts_per_layer)
            )
        }
        
        return mini_config
    
    def get_weight_files(self) -> List[Path]:
        """获取所有权重文件"""
        return sorted(self.source_path.glob("*.safetensors"))
    
    def extract_non_expert_weights(self, selected_layers: Set[int]) -> Dict[str, torch.Tensor]:
        """提取非专家权重（embeddings, attention等）"""
        non_expert_weights = {}
        
        print("\n提取非专家权重...")
        
        for weight_file in tqdm(self.get_weight_files()):
            with safe_open(weight_file, framework="pt") as f:
                for key in f.keys():
                    # 跳过专家权重
                    if 'experts' in key:
                        continue
                        
                    # 检查是否需要这个权重
                    should_include = False
                    new_key = key
                    
                    # Embeddings和输出层
                    if any(prefix in key for prefix in [
                        'model.embed_tokens', 
                        'lm_head',
                        'model.norm'
                    ]):
                        should_include = True
                        
                    # 层相关权重
                    elif 'model.layers.' in key:
                        # 提取层号
                        parts = key.split('.')
                        layer_idx_pos = parts.index('layers') + 1
                        if layer_idx_pos < len(parts) and parts[layer_idx_pos].isdigit():
                            orig_layer_idx = int(parts[layer_idx_pos])
                            
                            if orig_layer_idx in selected_layers:
                                # 重新映射层索引
                                new_layer_idx = sorted(selected_layers).index(orig_layer_idx)
                                parts[layer_idx_pos] = str(new_layer_idx)
                                new_key = '.'.join(parts)
                                should_include = True
                    
                    if should_include:
                        tensor = f.get_tensor(key)
                        non_expert_weights[new_key] = tensor
        
        print(f"提取了 {len(non_expert_weights)} 个非专家权重")
        return non_expert_weights
    
    def extract_expert_weights(self, layer_expert_mapping: Dict[int, List[int]]) -> Dict[str, torch.Tensor]:
        """提取选定的专家权重"""
        expert_weights = {}
        
        print("\n提取专家权重...")
        
        # 为每个选定的层提取专家
        for new_layer_idx, (orig_layer_idx, expert_ids) in enumerate(
            sorted(layer_expert_mapping.items())
        ):
            print(f"\n处理层 {orig_layer_idx} -> {new_layer_idx}")
            layer_expert_weights = {}
            
            # 查找包含该层专家的文件
            for weight_file in self.get_weight_files():
                with safe_open(weight_file, framework="pt") as f:
                    for key in f.keys():
                        # 匹配该层的专家权重
                        prefix = f"model.layers.{orig_layer_idx}.mlp.experts."
                        if key.startswith(prefix):
                            # 提取专家ID
                            expert_part = key[len(prefix):]
                            expert_id_str = expert_part.split('.')[0]
                            
                            if expert_id_str.isdigit():
                                expert_id = int(expert_id_str)
                                
                                if expert_id in expert_ids:
                                    # 重新映射专家索引
                                    new_expert_idx = expert_ids.index(expert_id)
                                    
                                    # 构建新的key
                                    weight_name = '.'.join(expert_part.split('.')[1:])
                                    new_key = f"model.layers.{new_layer_idx}.mlp.experts.{new_expert_idx}.{weight_name}"
                                    
                                    # 提取权重
                                    tensor = f.get_tensor(key)
                                    layer_expert_weights[new_key] = tensor
            
            print(f"  提取了 {len(layer_expert_weights)} 个专家权重")
            expert_weights.update(layer_expert_weights)
            
            # 清理内存
            del layer_expert_weights
            gc.collect()
        
        print(f"\n总共提取了 {len(expert_weights)} 个专家权重")
        return expert_weights
    
    def save_weights_in_chunks(self, weights: Dict[str, torch.Tensor], prefix: str):
        """分块保存权重（避免内存溢出）"""
        # 按大小分组权重
        max_chunk_size = 5 * 1024**3  # 5GB per chunk
        
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for key, tensor in weights.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 0
                
            current_chunk[key] = tensor
            current_size += tensor_size
            
        if current_chunk:
            chunks.append(current_chunk)
            
        # 保存每个块
        for i, chunk in enumerate(chunks):
            filename = f"{prefix}-{i+1}-of-{len(chunks)}.safetensors"
            save_file(chunk, self.output_path / filename)
            print(f"  保存 {filename} ({len(chunk)} tensors)")
            
        return len(chunks)
    
    def create_weight_index(self, num_chunks: int) -> Dict:
        """创建权重索引文件"""
        weight_map = {}
        
        # 扫描所有生成的文件
        for i in range(num_chunks):
            filename = f"model-{i+1}-of-{num_chunks}.safetensors"
            filepath = self.output_path / filename
            
            with safe_open(filepath, framework="pt") as f:
                for key in f.keys():
                    weight_map[key] = filename
                    
        index = {
            "metadata": {
                "total_size": sum(
                    os.path.getsize(self.output_path / f"model-{i+1}-of-{num_chunks}.safetensors")
                    for i in range(num_chunks)
                )
            },
            "weight_map": weight_map
        }
        
        return index
    
    def copy_tokenizer_files(self):
        """复制tokenizer相关文件"""
        tokenizer_files = [
            'tokenizer_config.json',
            'tokenizer.model',
            'tokenizer.json',
            'special_tokens_map.json',
            'tokenization_kimi.py',
            'configuration_deepseek.py',
            'modeling_deepseek.py'
        ]
        
        print("\n复制tokenizer文件...")
        for filename in tokenizer_files:
            src_file = self.source_path / filename
            if src_file.exists():
                shutil.copy2(src_file, self.output_path / filename)
                print(f"  复制 {filename}")
    
    def convert(self):
        """执行完整的转换流程"""
        print("\n=== 开始 K2 到 K2-Mini 转换 ===")
        
        # 1. 分析并选择层
        print("\n步骤 1: 分析层重要性")
        from scripts.analyze_layers import LayerAnalyzer
        
        analyzer = LayerAnalyzer(str(self.source_path))
        layer_analysis = analyzer.run_analysis(
            num_layers_to_keep=self.num_layers,
            save_results=False
        )
        selected_layers = layer_analysis['selected_layers']
        
        # 2. 选择专家
        print("\n步骤 2: 选择专家")
        expert_selector = ExpertSelector(
            str(self.source_path),
            num_experts_to_keep=self.experts_per_layer
        )
        layer_expert_mapping = expert_selector.select_all_experts(selected_layers)
        
        # 3. 创建配置
        print("\n步骤 3: 创建K2-Mini配置")
        mini_config = self.create_mini_config(selected_layers, layer_expert_mapping)
        
        with open(self.output_path / 'config.json', 'w') as f:
            json.dump(mini_config, f, indent=2)
        print("  配置已保存")
        
        # 4. 提取权重
        print("\n步骤 4: 提取权重")
        
        # 提取非专家权重
        non_expert_weights = self.extract_non_expert_weights(set(selected_layers))
        
        # 提取专家权重
        expert_weights = self.extract_expert_weights(layer_expert_mapping)
        
        # 5. 合并并保存权重
        print("\n步骤 5: 保存权重")
        all_weights = {**non_expert_weights, **expert_weights}
        
        print(f"总权重数: {len(all_weights)}")
        num_chunks = self.save_weights_in_chunks(all_weights, "model")
        
        # 6. 创建索引文件
        print("\n步骤 6: 创建索引")
        index = self.create_weight_index(num_chunks)
        
        with open(self.output_path / 'model.safetensors.index.json', 'w') as f:
            json.dump(index, f, indent=2)
            
        # 7. 复制tokenizer文件
        self.copy_tokenizer_files()
        
        # 8. 保存转换报告
        print("\n步骤 7: 生成转换报告")
        self.save_conversion_report(
            selected_layers, 
            layer_expert_mapping,
            expert_selector.estimate_model_size(layer_expert_mapping)
        )
        
        print("\n=== 转换完成！===")
        print(f"K2-Mini 模型已保存到: {self.output_path}")
        
    def save_conversion_report(self, selected_layers: List[int],
                             layer_expert_mapping: Dict[int, List[int]],
                             estimated_size_gb: float):
        """保存转换报告"""
        report = {
            "conversion_summary": {
                "source_model": str(self.source_path),
                "output_model": str(self.output_path),
                "timestamp": str(Path.ctime(Path.cwd()))
            },
            "architecture_changes": {
                "layers": {
                    "original": self.source_config['num_hidden_layers'],
                    "mini": self.num_layers,
                    "reduction": f"{(1 - self.num_layers/self.source_config['num_hidden_layers'])*100:.1f}%"
                },
                "experts_per_layer": {
                    "original": self.source_config['n_routed_experts'],
                    "mini": self.experts_per_layer,
                    "reduction": f"{(1 - self.experts_per_layer/self.source_config['n_routed_experts'])*100:.1f}%"
                }
            },
            "model_size": {
                "estimated_gb": estimated_size_gb,
                "fits_on_h100": estimated_size_gb < 75
            },
            "selected_layers": selected_layers,
            "expert_statistics": {
                "total_experts_original": self.source_config['num_hidden_layers'] * self.source_config['n_routed_experts'],
                "total_experts_mini": sum(len(experts) for experts in layer_expert_mapping.values()),
                "average_experts_per_layer": sum(len(experts) for experts in layer_expert_mapping.values()) / len(layer_expert_mapping)
            }
        }
        
        with open(self.output_path / 'conversion_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # 打印摘要
        print("\n=== 转换报告 ===")
        print(f"原始模型: {report['architecture_changes']['layers']['original']} 层, "
              f"{report['architecture_changes']['experts_per_layer']['original']} 专家/层")
        print(f"K2-Mini: {report['architecture_changes']['layers']['mini']} 层, "
              f"{report['architecture_changes']['experts_per_layer']['mini']} 专家/层")
        print(f"模型大小: ~{estimated_size_gb:.1f} GB")
        print(f"适合H100: {'是' if report['model_size']['fits_on_h100'] else '否'}")


def main():
    parser = argparse.ArgumentParser(description='将Kimi-K2转换为K2-Mini')
    parser.add_argument(
        '--source-model',
        type=str,
        required=True,
        help='源Kimi-K2模型路径'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='K2-Mini输出路径'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=24,
        help='K2-Mini层数 (默认: 24)'
    )
    parser.add_argument(
        '--experts-per-layer',
        type=int,
        default=16,
        help='每层保留的专家数 (默认: 16)'
    )
    
    args = parser.parse_args()
    
    # 创建转换器并执行
    converter = K2MiniConverter(
        args.source_model,
        args.output_path,
        args.num_layers,
        args.experts_per_layer
    )
    
    converter.convert()


if __name__ == '__main__':
    main()
