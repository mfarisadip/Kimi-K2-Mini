#\!/usr/bin/env python3
'''
将GSM8K数据转换为COCONUT格式
'''
import json
import random

def convert_jsonl_to_coconut(input_file, output_file, num_samples=100):
    '''转换GSM8K jsonl格式到COCONUT格式'''
    
    data = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line)
            
            # 提取问题和答案
            question = item['question']
            answer = item['answer']
            
            # 简化答案（去掉####后的数字之外的部分）
            if '####' in answer:
                steps = answer.split('####')[0].strip()
                final_answer = answer.split('####')[1].strip()
                
                # 创建COCONUT格式
                coconut_item = {
                    'question': question,
                    'answer': final_answer,
                    'steps': steps.split('. ') if '. ' in steps else [steps]
                }
            else:
                coconut_item = {
                    'question': question,
                    'answer': answer,
                    'steps': []
                }
            
            data.append(coconut_item)
    
    # 保存为JSON
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f'✅ 转换完成！{len(data)}条数据已保存到 {output_file}')
    return len(data)

# 转换测试集的前100条
if __name__ == '__main__':
    # 小规模测试数据（100条）
    convert_jsonl_to_coconut('data/test.jsonl', 'data/gsm8k_coconut_100.json', 100)
    
    # 中等规模数据（500条）
    convert_jsonl_to_coconut('data/test.jsonl', 'data/gsm8k_coconut_500.json', 500)
    
    # 完整测试集（1319条）
    convert_jsonl_to_coconut('data/test.jsonl', 'data/gsm8k_coconut_full.json', 2000)
