import torch
import time

def gpu_usage_test(tensor_size):
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("GPU is not available. Exiting.")
        return

    # 将计算移至GPU
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # 创建随机张量
    a = torch.randn(tensor_size, tensor_size, device=device)
    b = torch.randn(tensor_size, tensor_size, device=device)

    #print(f"Starting GPU usage test with tensor size {tensor_size}x{tensor_size} for {iterations} iterations.")

    start_time = time.time()

    # 循环执行大规模矩阵乘法以占用GPU
    while True:
        # 执行矩阵乘法并同步
        c = torch.matmul(a, b)
        torch.cuda.synchronize()


    #print(f"GPU usage test completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # 设置矩阵大小和迭代次数
    tensor_size = 4096 * 10 # 增大尺寸以增加GPU负载

    gpu_usage_test(tensor_size)
