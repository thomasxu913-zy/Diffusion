import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入我们之前写好的模块 (请确保文件名对应)
from segmadiff_dataset import SegMaDiffDriveDataset
from unet import MaskGuidedUNet

# ==========================================
# 2.5 升级版：时序轨迹解码器 (RNN-based Decoder)
# ==========================================
class TrajectoryDecoder(nn.Module):
    def __init__(self, in_channels=2, spatial_size=32, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        # 1. 空间特征提取
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * spatial_size * spatial_size, 256),
            nn.GELU()
        )
        # 2. 时序推演
        self.gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True)
        self.out_proj = nn.Linear(128, 2)
        
    def forward(self, x):
        B = x.shape[0]
        # 提取全局特征，并在时间维度上复制 pred_len 次
        feat = self.feature_extractor(x).unsqueeze(1).repeat(1, self.pred_len, 1) # [B, 10, 256]
        # 通过 GRU 引入时序依赖
        out, _ = self.gru(feat) # [B, 10, 128]
        # 输出最终坐标
        return self.out_proj(out) # [B, 10, 2]


# ==========================================
# 1. 扩散模型超参数与噪声调度 (Linear Schedule)
# ==========================================
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def q_sample(x_start, t, noise=None):
    """前向加噪过程：根据时间步 t 将真实数据 x_start 破坏为带噪数据"""
    if noise is None:
        noise = torch.randn_like(x_start)
    device = x_start.device
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# ==========================================
# 2. 维度桥接网络 (1D Trajectory <-> 2D Spatial)
# ==========================================
class TrajectoryProjector(nn.Module):
    def __init__(self, pred_len=10, out_channels=2, spatial_size=32):
        super().__init__()
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        # 将 10x2 的坐标映射到 2x32x32 的空间特征
        self.proj = nn.Linear(pred_len * 2, out_channels * spatial_size * spatial_size)
        
    def forward(self, traj):
        B = traj.shape[0]
        flat_traj = traj.view(B, -1)
        spatial_feature = self.proj(flat_traj)
        return spatial_feature.view(B, self.out_channels, self.spatial_size, self.spatial_size)

# ==========================================
# 3. 训练主循环 (Training Loop)
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动端到端联合训练，使用设备: {device}")
    
    dataset = SegMaDiffDriveDataset(data_root="./collected_data", seq_len=4, pred_len=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # 初始化三个核心组件
    traj_proj = TrajectoryProjector().to(device)
    traj_dec = TrajectoryDecoder().to(device) # 【新增】解码器
    model = MaskGuidedUNet(in_channels=2, mask_channels=1, base_dim=64).to(device)
    
    # 将三者的参数放入同一个优化器联合训练
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(traj_proj.parameters()) + list(traj_dec.parameters()), 
        lr=1e-4
    )
    
    model.train()
    traj_proj.train()
    traj_dec.train()
    
    epochs = 200 # 建议适当增加 epoch 以保证轨迹解码器收敛
    for epoch in range(epochs):
        total_loss = 0
        for step, (rgb, mask, traj) in enumerate(dataloader):
            mask = mask.to(device)
            traj = traj.to(device)
            
            # 1. 轨迹编码 (1D -> 2D 空间特征)
            x_start = traj_proj(traj)
            
            # 2. 自编码重建损失 (使用对异常值更鲁棒的 Smooth L1)
            traj_recon = traj_dec(x_start)
            loss_recon = F.smooth_l1_loss(traj_recon, traj)
            
            # 3. 【新增】运动学平滑损失：惩罚异常的加速度 (二阶导数)
            velocity = traj_recon[:, 1:] - traj_recon[:, :-1]       # 计算一阶导 (速度)
            acceleration = velocity[:, 1:] - velocity[:, :-1]       # 计算二阶导 (加速度)
            loss_smooth = torch.mean(acceleration ** 2)             # 抑制急刹、急转弯等不平滑现象
            
            # 4. 扩散模型前向加噪与去噪预测
            B = traj.shape[0]
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = q_sample(x_start, t, noise)
            
            mask_resized = F.interpolate(mask, size=(32, 32), mode='bilinear', align_corners=False)
            predicted_noise = model(x_noisy, mask_resized, t)
            
            # 5. 扩散重建损失
            loss_diff = F.mse_loss(predicted_noise, noise)
            
            # 6. 【终极修复】混合损失函数 (严格对齐物理量级)
            # 将几十上百的坐标/平滑 Loss 缩放到 1.0 左右的量级，防止扩散模型被忽视
            loss = loss_diff + 0.01 * loss_recon + 0.01 * loss_smooth
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
                
        print(f"✅ Epoch [{epoch+1}/{epochs}] 平均总 Loss: {total_loss/len(dataloader):.4f}")

    # ==========================================
    # 【新增】保存训练好的联合权重
    # ==========================================
    torch.save({
        'unet': model.state_dict(),
        'decoder': traj_dec.state_dict()
    }, 'segmadiff_drive_weights.pth')
    print("🎉 训练完成！权重已保存至 segmadiff_drive_weights.pth")

if __name__ == "__main__":
    train()