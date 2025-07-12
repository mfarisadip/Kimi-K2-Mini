#\!/usr/bin/env python3
"""
分析Kimi-K2模型各层的重要性
用于确定在K2-Mini中保留哪些层
"""
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
import matplotlib.pyplot as plt
import os

class LayerAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config_path = self.model_path / "config.json"
        
        # 加载配置
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        self.num_layers = self.config['num_hidden_layers']
        self.n_experts = self.config['n_routed_experts']
        
        print(f"模型路径: {model_path}")
        print(f"层数: {self.num_layers}")
        print(f"每层专家数: {self.n_experts}")
        
    def analyze_weight_magnitudes(self):
        """分析各层权重的大小"""
        layer_stats = {}
        
        print("\n分析权重幅度...")
        
        # 遍历所有safetensor文件
        for file_path in tqdm(list(self.model_path.glob("*.safetensors"))):
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    if "model.layers." in key:
                        # 提取层号
                        parts = key.split('.')
                        if len(parts) > 2 and parts[2].isdigit():
                            layer_idx = int(parts[2])
                            
                            if layer_idx not in layer_stats:
                                layer_stats[layer_idx] = {
                                    'total_magnitude': 0,
                                    'num_params': 0,
                                    'expert_activity': []
                                }
                            
                            # 获取张量
                            tensor = f.get_tensor(key)
                            # Handle FP8 types
                            if str(tensor.dtype).startswith("torch.float8"):
                                tensor = tensor.to(torch.float32)
                            magnitude = torch.abs(tensor).mean().item()
                            num_params = tensor.numel()
                            
                            layer_stats[layer_idx]['total_magnitude'] += magnitude * num_params
                            layer_stats[layer_idx]['num_params'] += num_params
                            
                            # 记录专家活跃度
                            if 'experts' in key:
                                expert_match = key.split('experts.')[1].split('.')[0]
                                if expert_match.isdigit():
                                    expert_idx = int(expert_match)
                                    layer_stats[layer_idx]['expert_activity'].append(
                                        (expert_idx, magnitude)
                                    )
        
        # 计算平均幅度
        for layer_idx in layer_stats:
            if layer_stats[layer_idx]['num_params'] > 0:
                layer_stats[layer_idx]['avg_magnitude'] = (
                    layer_stats[layer_idx]['total_magnitude'] / 
                    layer_stats[layer_idx]['num_params']
                )
            else:
                layer_stats[layer_idx]['avg_magnitude'] = 0
                
        return layer_stats
    
    def analyze_layer_connectivity(self):
        """分析层间连接模式"""
        # 简化版：基于层索引的重要性评分
        # 通常中间层更重要
        connectivity_scores = {}
        
        for i in range(self.num_layers):
            # 给中间层更高的分数
            position_score = 1.0 - abs(i - self.num_layers/2) / (self.num_layers/2)
            
            # 早期层对特征提取重要
            if i < 5:
                position_score += 0.3
                
            # 后期层对输出生成重要
            if i > self.num_layers - 5:
                position_score += 0.2
                
            connectivity_scores[i] = position_score
            
        return connectivity_scores
    
    def compute_layer_importance(self, weight_stats, connectivity_scores):
        """综合计算层重要性"""
        importance_scores = {}
        
        # 归一化权重幅度
        magnitudes = [stats.get('avg_magnitude', 0) for stats in weight_stats.values()]
        if magnitudes:
            max_magnitude = max(magnitudes)
            min_magnitude = min(magnitudes)
            magnitude_range = max_magnitude - min_magnitude
            
            if magnitude_range > 0:
                for layer_idx, stats in weight_stats.items():
                    normalized_magnitude = (
                        (stats.get('avg_magnitude', 0) - min_magnitude) / magnitude_range
                    )
                    
                    # 综合得分：权重幅度 + 连接性
                    importance_scores[layer_idx] = (
                        0.6 * normalized_magnitude + 
                        0.4 * connectivity_scores.get(layer_idx, 0.5)
                    )
            else:
                # 如果所有层幅度相同，只使用连接性分数
                importance_scores = connectivity_scores.copy()
        else:
            importance_scores = connectivity_scores.copy()
            
        return importance_scores
    
    def select_layers(self, importance_scores, num_layers_to_keep=24):
        """选择要保留的层"""
        # 按重要性排序
        sorted_layers = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 选择top层，但要保持一定的分布
        selected_layers = []
        
        # 确保包含首尾层
        selected_layers.extend([0, self.num_layers - 1])
        
        # 选择剩余的层
        remaining_slots = num_layers_to_keep - 2
        candidates = [l for l, _ in sorted_layers if l not in selected_layers]
        
        # 均匀分布选择
        if remaining_slots > 0 and candidates:
            step = max(1, len(candidates) // remaining_slots)
            for i in range(0, len(candidates), step):
                if len(selected_layers) < num_layers_to_keep:
                    selected_layers.append(candidates[i])
        
        # 排序
        selected_layers = sorted(list(set(selected_layers)))[:num_layers_to_keep]
        
        return selected_layers
    
    def visualize_analysis(self, importance_scores, selected_layers, output_path):
        """可视化分析结果"""
        plt.figure(figsize=(15, 8))
        
        # 子图1：层重要性
        plt.subplot(2, 1, 1)
        layers = list(range(self.num_layers))
        importances = [importance_scores.get(i, 0) for i in layers]
        
        colors = ['red' if i in selected_layers else 'blue' for i in layers]
        plt.bar(layers, importances, color=colors, alpha=0.7)
        plt.xlabel('Layer Index')
        plt.ylabel('Importance Score')
        plt.title(f'Layer Importance Analysis (Selected {len(selected_layers)} layers)')
        plt.grid(True, alpha=0.3)
        
        # 子图2：选中层的分布
        plt.subplot(2, 1, 2)
        plt.scatter(selected_layers, [1]*len(selected_layers), s=100, c='red')
        plt.xlim(-1, self.num_layers)
        plt.ylim(0.5, 1.5)
        plt.xlabel('Layer Index')
        plt.title('Selected Layers Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n分析图表已保存到: {output_path}")
    
    def run_analysis(self, num_layers_to_keep=24, save_results=True):
        """运行完整分析"""
        print("\n=== 开始层重要性分析 ===")
        
        # 1. 分析权重幅度
        weight_stats = self.analyze_weight_magnitudes()
        
        # 2. 分析连接性
        connectivity_scores = self.analyze_layer_connectivity()
        
        # 3. 计算综合重要性
        importance_scores = self.compute_layer_importance(
            weight_stats, connectivity_scores
        )
        
        # 4. 选择层
        selected_layers = self.select_layers(importance_scores, num_layers_to_keep)
        
        # 5. 保存结果
        results = {
            'num_original_layers': self.num_layers,
            'num_selected_layers': len(selected_layers),
            'selected_layers': selected_layers,
            'importance_scores': importance_scores,
            'layer_mapping': {
                'description': 'Mapping from K2-Mini layer index to original K2 layer index',
                'mapping': {i: selected_layers[i] for i in range(len(selected_layers))}
            }
        }
        
        if save_results:
            output_dir = Path('analysis_results')
            output_dir.mkdir(exist_ok=True)
            
            # 保存JSON结果
            with open(output_dir / 'layer_analysis.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # 生成可视化
            self.visualize_analysis(
                importance_scores, 
                selected_layers,
                output_dir / 'layer_importance.png'
            )
            
            print(f"\n分析结果已保存到: {output_dir}/")
        
        # 打印摘要
        print("\n=== 分析摘要 ===")
        print(f"原始层数: {self.num_layers}")
        print(f"选择层数: {len(selected_layers)}")
        print(f"选中的层: {selected_layers}")
        print(f"\n重要性最高的10层:")
        top_10 = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for layer, score in top_10:
            status = "✓" if layer in selected_layers else " "
            print(f"  [{status}] Layer {layer}: {score:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='分析Kimi-K2模型层重要性')
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Kimi-K2模型路径'
    )
    parser.add_argument(
        '--num-layers', 
        type=int, 
        default=24,
        help='K2-Mini要保留的层数 (默认: 24)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 运行分析
    analyzer = LayerAnalyzer(args.model_path)
    results = analyzer.run_analysis(
        num_layers_to_keep=args.num_layers,
        save_results=True
    )
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
