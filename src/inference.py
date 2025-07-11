"""
K2-Mini 推理模块
优化的推理接口，支持各种生成任务
"""
import torch
import time
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class K2MiniInference:
    """K2-Mini 推理引擎"""
    
    def __init__(self, model_path: str, device: str = "cuda", 
                 load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_path = Path(model_path)
        self.device = device
        
        print(f"初始化 K2-Mini 推理引擎...")
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")
        
        # 加载tokenizer
        print("加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 设置量化配置
        quantization_config = None
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("使用 INT8 量化")
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("使用 INT4 量化")
        
        # 加载模型
        print("加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if not (load_in_8bit or load_in_4bit) else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        print("✓ K2-Mini 推理引擎初始化完成")
        self._print_model_info()
        
    def _print_model_info(self):
        """打印模型信息"""
        if hasattr(self.model, 'config'):
            config = self.model.config
            print(f"\n模型信息:")
            print(f"  层数: {getattr(config, 'num_hidden_layers', 'N/A')}")
            print(f"  隐藏层大小: {getattr(config, 'hidden_size', 'N/A')}")
            print(f"  专家数/层: {getattr(config, 'n_routed_experts', 'N/A')}")
            
        # 估算内存使用
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"  GPU内存使用: {memory_mb:.1f} MB")
    
    def generate(self, 
                prompt: Union[str, List[str]], 
                max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                **kwargs) -> Union[str, List[str]]:
        """
        生成文本
        
        Args:
            prompt: 输入提示词（字符串或字符串列表）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本（字符串或字符串列表）
        """
        # 处理批量输入
        is_batch = isinstance(prompt, list)
        if not is_batch:
            prompt = [prompt]
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 更新生成配置
        gen_config = self.generation_config.to_dict()
        if max_new_tokens is not None:
            gen_config['max_new_tokens'] = max_new_tokens
        if temperature is not None:
            gen_config['temperature'] = temperature
        if top_p is not None:
            gen_config['top_p'] = top_p
        gen_config.update(kwargs)
        
        # 生成
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(**gen_config)
            )
        
        generation_time = time.time() - start_time
        
        # 解码输出
        generated_texts = []
        for i, output in enumerate(outputs):
            # 只解码新生成的部分
            input_length = inputs['input_ids'][i].shape[0]
            generated_ids = output[input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        # 打印统计信息
        total_tokens = sum(len(output) - inputs['input_ids'][i].shape[0] 
                         for i, output in enumerate(outputs))
        tokens_per_second = total_tokens / generation_time
        
        print(f"\n生成统计:")
        print(f"  生成时间: {generation_time:.2f}秒")
        print(f"  生成tokens: {total_tokens}")
        print(f"  速度: {tokens_per_second:.1f} tokens/秒")
        
        # 返回结果
        return generated_texts if is_batch else generated_texts[0]
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             **kwargs) -> str:
        """
        对话接口
        
        Args:
            messages: 对话历史，格式: [{"role": "user", "content": "..."}, ...]
            **kwargs: 生成参数
            
        Returns:
            助手回复
        """
        # 构建对话提示词
        prompt = self._build_chat_prompt(messages)
        
        # 生成回复
        response = self.generate(prompt, **kwargs)
        
        return response
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """构建对话提示词"""
        # 简单的对话模板
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"系统: {content}")
            elif role == 'user':
                prompt_parts.append(f"用户: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"助手: {content}")
        
        # 添加助手提示
        prompt_parts.append("助手:")
        
        return "\n\n".join(prompt_parts)
    
    def benchmark(self, 
                 test_prompts: Optional[List[str]] = None,
                 max_new_tokens: int = 100):
        """
        性能基准测试
        """
        if test_prompts is None:
            test_prompts = [
                "请解释什么是机器学习？",
                "写一个Python函数来计算斐波那契数列。",
                "翻译成英文：今天天气真好。",
                "续写故事：从前有座山，山里有座庙，",
            ]
        
        print("\n=== K2-Mini 性能基准测试 ===")
        print(f"测试prompts数: {len(test_prompts)}")
        print(f"每个prompt生成tokens: {max_new_tokens}")
        
        total_time = 0
        total_tokens = 0
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n测试 {i}/{len(test_prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            response = self.generate(prompt, max_new_tokens=max_new_tokens)
            elapsed_time = time.time() - start_time
            
            tokens_generated = len(self.tokenizer.encode(response))
            total_time += elapsed_time
            total_tokens += tokens_generated
            
            print(f"  生成时间: {elapsed_time:.2f}秒")
            print(f"  生成tokens: {tokens_generated}")
            print(f"  速度: {tokens_generated/elapsed_time:.1f} tokens/秒")
        
        # 总体统计
        print("\n=== 总体统计 ===")
        print(f"总时间: {total_time:.2f}秒")
        print(f"总tokens: {total_tokens}")
        print(f"平均速度: {total_tokens/total_time:.1f} tokens/秒")
        
        # GPU内存
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"峰值GPU内存: {memory_mb:.1f} MB")


def test_inference():
    """测试推理功能"""
    print("K2-Mini 推理模块测试")
    
    # 模拟测试（实际使用需要模型文件）
    print("\n预期性能指标:")
    print("配置         | 内存使用 | 推理速度 | 适用GPU")
    print("------------|---------|----------|--------")
    print("K2-Mini BF16| 60GB    | 30 t/s   | H100")
    print("K2-Mini INT8| 30GB    | 45 t/s   | A100 40GB")
    print("K2-Mini INT4| 15GB    | 60 t/s   | RTX 4090")
    
    print("\n使用示例:")
    print("""
# 基础使用
engine = K2MiniInference("/path/to/k2-mini")
response = engine.generate("解释什么是深度学习")

# INT8量化
engine_int8 = K2MiniInference("/path/to/k2-mini", load_in_8bit=True)

# 批量生成
responses = engine.generate([
    "写一首关于春天的诗",
    "解释相对论",
    "Python快速排序算法"
])

# 对话模式
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "请介绍一下人工智能"}
]
response = engine.chat(messages)
    """)


if __name__ == '__main__':
    test_inference()
