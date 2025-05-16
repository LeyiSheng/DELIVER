import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 定义线性噪声调度
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# 2. 前向过程：添加噪声
def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_0)
    return sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise, noise

# 3. 定义U-Net去噪网络
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # 简化的U-Net
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# 4. 训练DDPM
def train_ddpm(model, dataloader, num_timesteps, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    betas = linear_beta_schedule(num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    model.train()
    for epoch in range(10):
        for x_0, _ in dataloader:
            x_0 = x_0.to(device)
            t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device).long()
            noisy_x, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            noise_pred = model(noisy_x, t)

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 5. 采样函数：生成新图像
def sample_ddpm(model, num_timesteps, shape, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(shape, device=device)
        betas = linear_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        for t in reversed(range(num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = sqrt_recip_alphas_cumprod[t] * (x - sqrt_one_minus_alphas_cumprod[t] * predicted_noise) + noise * betas[t]

        return x

# 6. 主程序
if __name__ == "__main__":
    # 假设有一个DataLoader `dataloader` 用于加载MNIST数据集
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # 训练模型
    num_timesteps = 1000
    train_ddpm(model, dataloader, num_timesteps, device)

    # 生成样本
    sampled_images = sample_ddpm(model, num_timesteps, (64, 1, 28, 28), device)
    sampled_images = sampled_images.cpu().numpy()
    
    # 可视化生成的样本
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 8, figsize=(20, 2))
    for i in range(8):
        axes[i].imshow(sampled_images[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.show()
