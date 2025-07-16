#\!/usr/bin/env python3
"""
COCONUT LoRA微调脚本 for Kimi-K2-Mini
基于您提供的极简配置：LoRA r=8, lr=2e-4
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from tokenization_kimi_coconut import KimiCoconutTokenizer
import numpy as np
from typing import List, Dict
from tqdm import tqdm

class CoconutDataset(Dataset):
    """COCONUT格式的数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, num_pauses: int = 3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pauses = num_pauses
        
        # 加载数据
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        print(f"加载了 {len(self.data)} 条数据")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 格式化输入：question + pause tokens
        question = item['question']
        answer = item['answer']
        
        # 添加pause tokens
        input_text = self.tokenizer.format_with_pauses(question, self.num_pauses)
        full_text = input_text + answer
        
        # 编码
        encoding = self.tokenizer.encode_for_coconut(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # 创建标签（只在answer部分计算loss）
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # Mask掉question和pause部分
        question_with_pauses_len = len(self.tokenizer.encode(input_text))
        labels[:question_with_pauses_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels,
            'latent_positions': encoding.get('latent_positions', torch.tensor([]))
        }

def prepare_gsm8k_data(output_path: str = 'gsm8k_coconut.json'):
    """准备GSM8K数据集（简化版）"""
    # 这里使用一些示例数据，实际使用时应该下载完整的GSM8K
    sample_data = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder for  per egg. How much does she make every day?",
            "answer": "Janet sells 16 - 3 - 4 = 9 eggs per day. She makes 9 *  =  every day.",
            "steps": [
                "Janet's ducks lay 16 eggs per day",
                "She uses 3 + 4 = 7 eggs", 
                "She has 16 - 7 = 9 eggs left",
                "She makes 9 *  = "
            ]
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "It takes 2/2 = 1 bolt of white fiber. So it takes 2 + 1 = 3 bolts in total.",
            "steps": [
                "Blue fiber: 2 bolts",
                "White fiber: 2/2 = 1 bolt",
                "Total: 2 + 1 = 3 bolts"
            ]
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"保存示例数据到 {output_path}")
    return output_path

def create_lora_config():
    """创建LoRA配置（基于您提供的参数）"""
    return LoraConfig(
        r=8,                                    # LoRA秩
        lora_alpha=16,                          # LoRA alpha (通常是r的2倍)
        target_modules=[                        # 目标模块
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj",      # MLP层
        ],
        lora_dropout=0.1,                       # Dropout
        bias="none",                            # 不训练bias
        task_type=TaskType.CAUSAL_LM,          # 任务类型
    )

def train_coconut_lora(
    model_path: str = './k2-mini',
    data_path: str = 'gsm8k_coconut.json',
    output_dir: str = './k2-mini-coconut-lora',
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    num_pauses: int = 3
):
    """训练COCONUT LoRA"""
    
    print("=== 开始COCONUT LoRA训练 ===")
    
    # 1. 加载tokenizer
    print("\n1. 加载tokenizer...")
    tokenizer = KimiCoconutTokenizer(model_path)
    
    # 2. 准备数据
    print("\n2. 准备数据集...")
    if not os.path.exists(data_path):
        data_path = prepare_gsm8k_data(data_path)
    
    dataset = CoconutDataset(data_path, tokenizer, num_pauses=num_pauses)
    
    # 3. 加载模型
    print("\n3. 加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    
    # 调整token embeddings大小
    model.resize_token_embeddings(len(tokenizer.base_tokenizer), pad_to_multiple_of=64)
    
    # 4. 应用LoRA
    print("\n4. 应用LoRA配置...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    # 5. 设置训练参数
    print("\n5. 设置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",  # 不使用wandb
        dataloader_drop_last=True,
        group_by_length=True,
    )
    
    # 6. 创建Trainer
    print("\n6. 创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer.base_tokenizer,
    )
    
    # 7. 开始训练
    print("\n7. 开始训练...")
    trainer.train()
    
    # 8. 保存模型
    print("\n8. 保存LoRA权重...")
    model.save_pretrained(output_dir)
    tokenizer.base_tokenizer.save_pretrained(output_dir)
    
    # 保存COCONUT配置
    coconut_config = {
        'num_pauses': num_pauses,
        'pause_token': '<|pause|>',
        'special_tokens': list(tokenizer.coconut_tokens.keys())
    }
    with open(os.path.join(output_dir, 'coconut_config.json'), 'w') as f:
        json.dump(coconut_config, f, indent=2)
    
    print(f"\n✅ 训练完成！LoRA权重保存在: {output_dir}")
    return output_dir

def quick_inference_test(model_path: str, lora_path: str):
    """快速推理测试"""
    print("\n=== 快速推理测试 ===")
    
    # 加载模型和LoRA
    from peft import PeftModel
    
    tokenizer = KimiCoconutTokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.resize_token_embeddings(len(tokenizer.base_tokenizer), pad_to_multiple_of=64)
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    # 测试问题
    test_questions = [
        "What is 25 times 17?",
        "If a store sells apples for  each and I buy 7 apples, how much do I pay?",
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        
        # 添加pause tokens
        input_text = tokenizer.format_with_pauses(question, num_pauses=3)
        print(f"输入: {input_text}")
        
        # 生成
        inputs = tokenizer.encode_for_coconut(input_text, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # 解码
        generated = tokenizer.decode_skip_latent(outputs[0], skip_special_tokens=True)
        print(f"输出: {generated}")
        

if __name__ == '__main__':
    # 主训练流程
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./k2-mini', help='基础模型路径')
    parser.add_argument('--data_path', default='gsm8k_coconut.json', help='训练数据路径')
    parser.add_argument('--output_dir', default='./k2-mini-coconut-lora', help='输出目录')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--num_pauses', type=int, default=3, help='pause token数量')
    parser.add_argument('--test_only', action='store_true', help='只进行推理测试')
    
    args = parser.parse_args()
    
    if args.test_only:
        # 只测试
        quick_inference_test(args.model_path, args.output_dir)
    else:
        # 训练
        lora_path = train_coconut_lora(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_pauses=args.num_pauses
        )
        
        # 训练后测试
        quick_inference_test(args.model_path, lora_path)
