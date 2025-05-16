import torch
import time

def gpu_usage_test(max_memory_gb):
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("GPU is not available. Exiting.")
        return

    # 将计算移至GPU
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # 确定张量大小以接近最大内存限制
    tensor_size = int((max_memory_gb * 1024**3 / 4)**0.5)
    print(f"Tensor size set to: {tensor_size}x{tensor_size}")

    # 创建随机张量
    a = torch.randn(tensor_size, tensor_size, device=device)
    b = torch.randn(tensor_size, tensor_size, device=device)

    start_time = time.time()

    # 循环执行大规模矩阵乘法以占用GPU
    while True:
        # 执行矩阵乘法并同步
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

if __name__ == "__main__":
    # 设置最大内存限制为40GB
    max_memory_gb = 7.5

    gpu_usage_test(max_memory_gb)