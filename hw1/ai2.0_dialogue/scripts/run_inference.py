import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

def get_gpu_stats():
    """获取GPU内存统计（单位MB）"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return {
            'initial': alloc,
            'peak': peak,
            'current': alloc
        }
    return {'initial': 0, 'peak': 0, 'current': 0}

# 初始化统计量
total_time = 0.0
total_tokens = 0
gpu_stats = []

# 记录初始显存状态
gpu_stats.append(("Before Loading", get_gpu_stats()))

# Load model
local_model_path = "../model/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path).to(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# 记录加载后显存
gpu_stats.append(("After Loading", get_gpu_stats()))

# 读取prompt
with open('../data/prompt.txt', 'r') as f:
    prompts = [p.strip() for p in f.readlines() if p.strip()]

# 创建结果目录
os.makedirs('../results', exist_ok=True)

# 处理每个prompt
for idx, prompt in enumerate(prompts):
    # 编码输入
    input_ids = tokenizer.encode(
        prompt + tokenizer.eos_token, 
        return_tensors='pt'
    ).to(model.device)
    
    # 推理前显存记录
    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
    
    # 执行推理
    start_time = time.time()
    output = model.generate(
        input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    
    # 计算指标
    inference_time = end_time - start_time
    generated_tokens = len(output[0]) - len(input_ids[0])
    
    # 更新统计
    total_time += inference_time
    total_tokens += generated_tokens
    
    # 记录显存
    gpu_stats.append((
        f"Prompt {idx+1}", 
        get_gpu_stats()
    ))
    
    # 保存响应
    response = tokenizer.decode(
        output[:, input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    with open(f'../results/response_{idx+1}.txt', 'w') as f:
        f.write(response)
    
    # 保存单个推理时间
    with open(f'../results/inference_time_{idx+1}.txt', 'w') as f:
        f.write(f"{inference_time:.2f}")

# 最终显存记录
gpu_stats.append(("Final", get_gpu_stats()))

# 保存汇总数据
with open('../results/summary.txt', 'w') as f:
    f.write(f"Total Prompts: {len(prompts)}\n")
    f.write(f"Total Inference Time: {total_time:.2f}s\n")
    f.write(f"Average Inference Time: {total_time/len(prompts):.2f}s\n")
    f.write(f"Total Generated Tokens: {total_tokens}\n")
    f.write(f"Tokens per Second: {total_tokens/total_time:.2f}\n")
    f.write("\nGPU Memory Usage (MB):\n")
    for stage, stats in gpu_stats:
        f.write(f"{stage}: {stats['peak']:.1f} (peak)\n")

# 打印最终统计
print("\n=== Final Statistics ===")
print(f"Total Prompts Processed: {len(prompts)}")
print(f"Average Inference Time: {total_time/len(prompts):.2f}s")
print(f"Total Tokens Generated: {total_tokens}")
print(f"Token Generation Rate: {total_tokens/total_time:.2f} tokens/s")
print(f"Peak GPU Memory Usage: {gpu_stats[-2][1]['peak']:.1f} MB")