import torch
import time
import cv2

def get_gpu_memory():
    """获取当前GPU内存占用（单位MB）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为MB
        cached = torch.cuda.memory_reserved() / 1024**2
        return allocated, cached
    return 0.0, 0.0

# Load model
repo_path = "/root/shuochen/FSHT-for-AI/hw1/ai1.0_target_detection/model/yolov5"
model_path = "/root/shuochen/FSHT-for-AI/hw1/ai1.0_target_detection/model/yolov5s.pt"

# 记录初始显存状态
init_alloc, init_cached = get_gpu_memory()

# 加载模型前显存基准
model = torch.hub.load(
    repo_path, 
    'custom', 
    path=model_path, 
    source='local', 
    force_reload=True
).to('cuda' if torch.cuda.is_available() else 'cpu')

# 模型加载后显存占用
model_alloc, model_cached = get_gpu_memory()

# Load image
img_path = '../data/test.jpg'
img = cv2.imread(img_path)

# 预热GPU（可选）
if torch.cuda.is_available():
    dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
    _ = model(dummy_input)
    torch.cuda.empty_cache()

# Run inference
start_time = time.time()
results = model(img)
end_time = time.time()

# 推理后显存峰值
peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

# Calculate metrics
inference_time = end_time - start_time
fps = 1 / inference_time
detection_count = len(results.pred[0])
current_alloc, current_cached = get_gpu_memory()

# 添加可视化标注
results.render()  # 生成标注图像
cv2.imwrite('../results/annotated_result.jpg', img)  # 保存带标注的图像

# 保存带显存信息的指标
with open('../results/metrics.txt', 'w') as f:
    f.write(f"Input Resolution: {img.shape[1]}x{img.shape[0]}\n")
    f.write(f"Detection Count: {detection_count}\n")
    f.write(f"Inference Time: {inference_time:.2f}s\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write(f"VRAM Usage (MB):\n")
    f.write(f"  - Initial: {init_alloc:.1f}\n")
    f.write(f"  - After Load: {model_alloc:.1f}\n")
    f.write(f"  - Peak During Inference: {peak_alloc:.1f}\n")
    f.write(f"  - Current: {current_alloc:.1f}\n")

# 打印关键信息
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Inference Time: {inference_time:.2f}s")
print(f"Peak VRAM Usage: {peak_alloc:.1f} MB")