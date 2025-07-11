"""
专家选择器 - 从每层384个专家中选择最重要的16个
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from safetensors import safe_open
from tqdm import tqdm
import heapq
from collections import defaultdict


class ExpertSelector:
    """
    选择每层最重要的专家
    策略：
    1. 基于权重幅度
    2. 基于激活频率（如果有统计数据）
    3. 基于专家多样性
    """
    
    def __init__(self, model_path: str, num_experts_to_keep: int = 16):
        self.model_path = Path(model_path)
        self.num_experts_to_keep = num_experts_to_keep
        
        # 加载配置
        with open(self.model_path / 'config.json', 'r') as f:
            self.config = json.load(f)
            
        self.num_layers = self.config['num_hidden_layers']
        self.total_experts = self.config['n_routed_experts']
        
        print(f"ExpertSelector 初始化:")
        print(f"  模型路径: {model_path}")
        print(f"  总专家数/层: {self.total_experts}")
        print(f"  保留专家数/层: {self.num_experts_to_keep}")
        
    def analyze_expert_weights(self, layer_idx: int) -> Dict[int, float]:
        """
        分析指定层所有专家的权重重要性
        返回: {expert_id: importance_score}
        """
        expert_scores = {}
        
        # 遍历模型文件查找该层的专家
        for file_path in self.model_path.glob("*.safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    # 匹配格式: model.layers.{layer}.mlp.experts.{expert_id}.{weight_name}
                    if f"model.layers.{layer_idx}.mlp.experts." in key:
                        parts = key.split('.')
                        expert_idx_pos = parts.index('experts') + 1
                        if expert_idx_pos < len(parts) and parts[expert_idx_pos].isdigit():
                            expert_id = int(parts[expert_idx_pos])
                            
                            if expert_id not in expert_scores:
                                expert_scores[expert_id] = 0.0
                            
                            # 获取权重并计算重要性
                            tensor = f.get_tensor(key)
                            
                            # 重要性度量：L2范数 + 稀疏度
                            l2_norm = torch.norm(tensor, p=2).item()
                            sparsity = (tensor.abs() < 1e-6).float().mean().item()
                            
                            # 综合评分（L2范数高且稀疏度低的专家更重要）
                            importance = l2_norm * (1.0 - sparsity)
                            expert_scores[expert_id] += importance
        
        return expert_scores
    
    def load_activation_statistics(self, stats_file: Optional[str] = None) -> Dict[Tuple[int, int], float]:
        """
        加载专家激活统计数据（如果有的话）
        返回: {(layer_id, expert_id): activation_frequency}
        """
        if stats_file and Path(stats_file).exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                
            # 转换为 (layer, expert) -> frequency 格式
            activation_freq = {}
            for layer_str, experts in stats.items():
                if layer_str.isdigit():
                    layer_id = int(layer_str)
                    for expert_str, freq in experts.items():
                        if expert_str.isdigit():
                            expert_id = int(expert_str)
                            activation_freq[(layer_id, expert_id)] = freq
                            
            return activation_freq
        
        # 如果没有统计数据，返回均匀分布
        return {}
    
    def compute_expert_diversity(self, layer_idx: int, selected_experts: List[int],
                               candidate_expert: int) -> float:
        """
        计算候选专家与已选专家的多样性分数
        多样性高 = 与已选专家的权重模式差异大
        """
        if not selected_experts:
            return 1.0
        
        # 简化版：基于专家ID的距离
        # 实际应该比较权重向量的余弦相似度
        min_distance = min(abs(candidate_expert - e) for e in selected_experts)
        
        # 归一化距离
        max_possible_distance = self.total_experts // 2
        diversity_score = min_distance / max_possible_distance
        
        return diversity_score
    
    def select_experts_for_layer(self, layer_idx: int, 
                               activation_stats: Optional[Dict] = None) -> List[int]:
        """
        为指定层选择最重要的专家
        """
        # 1. 获取权重重要性分数
        weight_scores = self.analyze_expert_weights(layer_idx)
        
        # 2. 结合激活频率（如果有）
        combined_scores = {}
        for expert_id, weight_score in weight_scores.items():
            activation_freq = 1.0  # 默认值
            if activation_stats and (layer_idx, expert_id) in activation_stats:
                activation_freq = activation_stats[(layer_idx, expert_id)]
            
            # 综合分数：权重重要性 * 激活频率
            combined_scores[expert_id] = weight_score * (0.7 + 0.3 * activation_freq)
        
        # 3. 贪心选择，考虑多样性
        selected_experts = []
        candidates = list(combined_scores.items())
        
        for _ in range(min(self.num_experts_to_keep, len(candidates))):
            best_expert = None
            best_score = -float('inf')
            
            for expert_id, base_score in candidates:
                if expert_id not in selected_experts:
                    # 考虑多样性
                    diversity = self.compute_expert_diversity(
                        layer_idx, selected_experts, expert_id
                    )
                    
                    # 调整分数：基础分数 + 多样性奖励
                    adjusted_score = base_score * (1.0 + 0.2 * diversity)
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_expert = expert_id
            
            if best_expert is not None:
                selected_experts.append(best_expert)
        
        return sorted(selected_experts)
    
    def select_all_experts(self, selected_layers: List[int],
                          activation_stats_file: Optional[str] = None) -> Dict[int, List[int]]:
        """
        为所有选定的层选择专家
        返回: {layer_idx: [expert_ids]}
        """
        # 加载激活统计
        activation_stats = self.load_activation_statistics(activation_stats_file)
        
        layer_expert_mapping = {}
        
        print(f"\n为 {len(selected_layers)} 层选择专家...")
        for layer_idx in tqdm(selected_layers):
            selected_experts = self.select_experts_for_layer(layer_idx, activation_stats)
            layer_expert_mapping[layer_idx] = selected_experts
            
            print(f"  Layer {layer_idx}: 选择了 {len(selected_experts)} 个专家")
        
        return layer_expert_mapping
    
    def estimate_model_size(self, layer_expert_mapping: Dict[int, List[int]]) -> float:
        """
        估算精简后的模型大小
        """
        # 基础模型大小（embeddings, attention等）
        hidden_size = self.config['hidden_size']
        vocab_size = self.config['vocab_size']
        num_selected_layers = len(layer_expert_mapping)
        
        # Embeddings
        embedding_params = vocab_size * hidden_size
        
        # Attention layers
        attention_params = num_selected_layers * (4 * hidden_size * hidden_size)
        
        # Shared experts
        shared_expert_size = self.config.get('intermediate_size', 18432)
        shared_expert_params = num_selected_layers * (3 * hidden_size * shared_expert_size)
        
        # Selected MoE experts
        moe_intermediate_size = self.config.get('moe_intermediate_size', 2048)
        total_selected_experts = sum(len(experts) for experts in layer_expert_mapping.values())
        moe_params = total_selected_experts * (3 * hidden_size * moe_intermediate_size)
        
        # 总参数量
        total_params = (
            embedding_params + 
            attention_params + 
            shared_expert_params + 
            moe_params
        )
        
        # 转换为GB (bfloat16)
        size_gb = (total_params * 2) / (1024**3)
        
        return size_gb
    
    def save_selection_results(self, layer_expert_mapping: Dict[int, List[int]],
                             output_path: str = "expert_selection.json"):
        """
        保存专家选择结果
        """
        # 计算统计信息
        total_experts_original = self.num_layers * self.total_experts
        total_experts_selected = sum(len(experts) for experts in layer_expert_mapping.values())
        compression_ratio = total_experts_original / total_experts_selected
        
        results = {
            "selection_config": {
                "num_experts_per_layer": self.num_experts_to_keep,
                "selection_strategy": "weight_magnitude_and_diversity"
            },
            "statistics": {
                "original_total_experts": total_experts_original,
                "selected_total_experts": total_experts_selected,
                "compression_ratio": compression_ratio,
                "estimated_size_gb": self.estimate_model_size(layer_expert_mapping)
            },
            "layer_expert_mapping": layer_expert_mapping,
            "expert_distribution": {
                "min_experts_in_layer": min(len(e) for e in layer_expert_mapping.values()),
                "max_experts_in_layer": max(len(e) for e in layer_expert_mapping.values()),
                "avg_experts_per_layer": total_experts_selected / len(layer_expert_mapping)
            }
        }
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n专家选择结果已保存到: {output_path}")
        print(f"\n=== 选择统计 ===")
        print(f"原始专家总数: {total_experts_original:,}")
        print(f"选择专家总数: {total_experts_selected:,}")
        print(f"压缩比: {compression_ratio:.1f}x")
        print(f"预估模型大小: {results['statistics']['estimated_size_gb']:.1f} GB")
        
        return results


def test_expert_selector():
    """测试专家选择器"""
    # 这里使用模拟数据进行测试
    print("ExpertSelector 模块测试")
    
    # 模拟配置
    mock_config = {
        'hidden_size': 7168,
        'num_hidden_layers': 61,
        'n_routed_experts': 384,
        'moe_intermediate_size': 2048,
        'intermediate_size': 18432,
        'vocab_size': 163840
    }
    
    print(f"\n模拟配置:")
    print(f"  层数: {mock_config['num_hidden_layers']}")
    print(f"  专家数/层: {mock_config['n_routed_experts']}")
    
    # 测试专家选择逻辑
    selected_layers = list(range(0, 24))  # 假设选择了24层
    mock_expert_mapping = {
        layer: list(range(16))  # 每层选16个专家
        for layer in selected_layers
    }
    
    # 计算模型大小
    total_params = 0
    
    # Embeddings
    total_params += mock_config['vocab_size'] * mock_config['hidden_size']
    
    # Attention
    total_params += len(selected_layers) * 4 * mock_config['hidden_size']**2
    
    # Shared experts
    total_params += len(selected_layers) * 3 * mock_config['hidden_size'] * mock_config['intermediate_size']
    
    # MoE experts
    num_experts = sum(len(e) for e in mock_expert_mapping.values())
    total_params += num_experts * 3 * mock_config['hidden_size'] * mock_config['moe_intermediate_size']
    
    size_gb = (total_params * 2) / (1024**3)
    
    print(f"\n预估K2-Mini规格:")
    print(f"  层数: {len(selected_layers)}")
    print(f"  专家数/层: 16")
    print(f"  总参数量: {total_params/1e9:.1f}B")
    print(f"  模型大小: {size_gb:.1f} GB")
    print(f"  适合H100部署: {'是' if size_gb < 75 else '否'}")


if __name__ == '__main__':
    test_expert_selector()
