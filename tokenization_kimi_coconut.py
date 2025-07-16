#\!/usr/bin/env python3
"""
Kimi-K2-Mini Tokenizer with COCONUT/Pause Token Support
"""
import os
import json
import torch
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
import tiktoken

class KimiCoconutTokenizer:
    """扩展Kimi tokenizer以支持COCONUT pause tokens"""
    
    def __init__(self, base_path: str):
        # 加载基础tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            base_path, 
            trust_remote_code=True
        )
        
        # 定义COCONUT特殊tokens
        self.coconut_tokens = {
            '<|pause|>': -100,      # 主要的pause token
            '<|think|>': -101,      # 思考开始
            '<|/think|>': -102,     # 思考结束  
            '<|compute|>': -103,    # 计算标记
            '<|/compute|>': -104,   # 计算结束
            '<|latent|>': -105,     # 潜在推理标记
        }
        
        # 添加特殊tokens到tokenizer
        self._add_coconut_tokens()
        
    def _add_coconut_tokens(self):
        """添加COCONUT特殊tokens"""
        special_tokens = list(self.coconut_tokens.keys())
        
        # 检查是否已经添加过
        existing_tokens = self.base_tokenizer.additional_special_tokens or []
        new_tokens = [t for t in special_tokens if t not in existing_tokens]
        
        if new_tokens:
            self.base_tokenizer.add_special_tokens({
                'additional_special_tokens': existing_tokens + new_tokens
            })
            
        # 更新token ID映射
        for token in self.coconut_tokens:
            self.coconut_tokens[token] = self.base_tokenizer.convert_tokens_to_ids(token)
            
    def format_with_pauses(self, text: str, num_pauses: int = 3) -> str:
        """在文本中插入pause tokens"""
        pause_token = '<|pause|>'
        pauses = pause_token * num_pauses
        return f"{text}{pauses}"
        
    def format_with_thinking(self, question: str, thinking_steps: List[str], answer: str) -> str:
        """格式化带有思考过程的文本"""
        thinking = ' '.join(thinking_steps)
        return f"{question}<|think|>{thinking}<|/think|>{answer}"
        
    def encode_for_coconut(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """为COCONUT推理编码文本"""
        # 编码文本
        encoded = self.base_tokenizer(text, return_tensors='pt', **kwargs)
        
        # 识别latent token位置
        input_ids = encoded['input_ids']
        latent_positions = []
        
        for token_name, token_id in self.coconut_tokens.items():
            if 'pause' in token_name or 'latent' in token_name:
                positions = (input_ids == token_id).nonzero(as_tuple=True)[1]
                latent_positions.extend(positions.tolist())
                
        encoded['latent_positions'] = torch.tensor(latent_positions)
        return encoded
        
    def decode_skip_latent(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """解码时跳过latent tokens"""
        # 过滤掉latent相关的token
        if skip_special_tokens:
            latent_ids = set(self.coconut_tokens.values())
            filtered_ids = [tid for tid in token_ids.tolist() if tid not in latent_ids]
            token_ids = torch.tensor(filtered_ids)
            
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def pause_token_id(self):
        return self.coconut_tokens['<|pause|>']
        
    @property
    def latent_token_id(self):
        return self.coconut_tokens['<|latent|>']
        
    def __getattr__(self, name):
        """委托给基础tokenizer"""
        return getattr(self.base_tokenizer, name)


# 测试代码
if __name__ == '__main__':
    # 测试tokenizer
    tokenizer = KimiCoconutTokenizer('./k2-mini')
    
    # 测试pause token
    text = "人工智能的未来发展方向是"
    text_with_pauses = tokenizer.format_with_pauses(text, num_pauses=3)
    print(f"带pause的文本: {text_with_pauses}")
    
    # 测试思考格式
    question = "计算 25 * 17"
    thinking = ["25 * 17 = 25 * (10 + 7)", "= 250 + 175", "= 425"]
    answer = "425"
    formatted = tokenizer.format_with_thinking(question, thinking, answer)
    print(f"\n带思考过程的文本: {formatted}")
    
    # 测试编码
    encoded = tokenizer.encode_for_coconut(text_with_pauses)
    print(f"\n编码结果: {encoded['input_ids'].shape}")
    print(f"Latent位置: {encoded['latent_positions']}")
