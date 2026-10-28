# unet.py
import torch
import torch.nn as nn
import math

# 核心：直接引入官方原版的 Mamba Dynamic Former
from MDFormer import MambaDynamicFormer 

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class MaskGuidedUNet(nn.Module):
    """
    SegMaDiff-Drive 核心网络: 掩码引导的扩散 U-Net，核心融合官方 MDFormer
    """
    def __init__(self, in_channels=2, mask_channels=1, base_dim=64):
        super().__init__()
        
        self.init_conv = nn.Conv2d(in_channels + mask_channels, base_dim, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.GELU(),
            nn.Linear(base_dim * 4, base_dim * 4)
        )
        
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1)
        
        # 100% 还原 Proposal：将 U-Net 的核心替换为官方 MambaDynamicFormer
        self.bottleneck = MambaDynamicFormer(in_channels=base_dim * 4)
        
        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_dim * 2 * 2, base_dim, kernel_size=4, stride=2, padding=1) 
        self.out_conv = nn.Conv2d(base_dim * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x, mask, time):
        t_emb = self.time_mlp(time)
        t_emb = t_emb[(..., ) + (None, ) * 2] 
        
        x_input = torch.cat([x, mask], dim=1) 
        x0 = self.init_conv(x_input)
        
        d1 = self.down1(x0)
        d2 = self.down2(d1)
        
        d2 = d2 + t_emb
        bottleneck_out = self.bottleneck(d2)
        
        u1 = self.up1(bottleneck_out)
        u1_cat = torch.cat([u1, d1], dim=1)
        
        u2 = self.up2(u1_cat)
        u2_cat = torch.cat([u2, x0], dim=1)
        
        return self.out_conv(u2_cat)

if __name__ == "__main__":
    batch_size = 4
    noisy_traj = torch.randn(batch_size, 2, 32, 32) 
    bev_mask = torch.randn(batch_size, 1, 32, 32)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    model = MaskGuidedUNet(in_channels=2, mask_channels=1, base_dim=64)
    predicted_noise = model(noisy_traj, bev_mask, timesteps)
    
    if noisy_traj.shape == predicted_noise.shape:
        print("✅ 官方 SegMaDiff 架构已成功迁移至自动驾驶 U-Net！")