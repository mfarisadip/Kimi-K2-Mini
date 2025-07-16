#\!/usr/bin/env python3
'''
简单测试COCONUT功能（不加载完整模型）
'''
import json
from tokenization_kimi_coconut import KimiCoconutTokenizer

def test_coconut_tokenizer():
    print('=== COCONUT Tokenizer 测试 ===\n')
    
    # 初始化tokenizer
    tokenizer = KimiCoconutTokenizer('./k2-mini')
    
    # 加载测试数据
    with open('data/gsm8k_coconut_100.json', 'r') as f:
        data = json.load(f)
    
    # 测试前5条数据
    for i, item in enumerate(data[:5]):
        print(f'\n示例 {i+1}:')
        print(f'问题: {item["question"][:100]}...')
        
        # 添加pause tokens
        text_with_pauses = tokenizer.format_with_pauses(item['question'], num_pauses=3)
        print(f'添加pause: ...{text_with_pauses[-50:]}')
        
        # 编码
        encoded = tokenizer.encode_for_coconut(text_with_pauses)
        print(f'编码长度: {encoded["input_ids"].shape[1]}')
        print(f'Latent位置: {encoded.get("latent_positions", [])}')
        
        # 使用思考格式
        if item['steps']:
            formatted = tokenizer.format_with_thinking(
                item['question'], 
                item['steps'][:2],  # 只用前两步
                item['answer']
            )
            print(f'思考格式长度: {len(formatted)}')

def test_training_data():
    print('\n\n=== 训练数据检查 ===\n')
    
    # 检查数据文件
    import os
    data_files = [
        'data/gsm8k_coconut_100.json',
        'data/gsm8k_coconut_500.json', 
        'data/gsm8k_coconut_full.json'
    ]
    
    for file in data_files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                data = json.load(f)
            print(f'✅ {file}: {len(data)} 条数据')
            
            # 统计
            avg_question_len = sum(len(d['question']) for d in data) / len(data)
            avg_answer_len = sum(len(d['answer']) for d in data) / len(data)
            print(f'   平均问题长度: {avg_question_len:.0f} 字符')
            print(f'   平均答案长度: {avg_answer_len:.0f} 字符')

def estimate_memory_usage():
    print('\n\n=== 显存估算 ===\n')
    
    print('K2-Mini (32.5B参数):')
    print('  - FP16: ~65GB')
    print('  - INT8: ~32GB')
    print('  - LoRA训练额外: +5GB')
    print('\n当前MIG GPU只有9.8GB，需要使用CloudExe获取完整H100')
    
    print('\n建议方案:')
    print('  1. 使用CloudExe --gpuspec H100x1 (80GB)')
    print('  2. 或使用更小的模型测试（如Llama-3-8B）')
    print('  3. 或使用INT8量化')

if __name__ == '__main__':
    test_coconut_tokenizer()
    test_training_data()
    estimate_memory_usage()
    
    print('\n✅ COCONUT功能测试完成！')
    print('\n下一步: 运行 ./run_with_cloudexe.sh 使用完整H100进行训练')
