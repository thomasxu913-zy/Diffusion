import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from unet import MaskGuidedUNet
from train import TrajectoryProjector, TrajectoryDecoder  


# ==========================================
# 2. 升级版：DAAM 注意力热图提取 (累积机制)
# ==========================================
daam_heatmap_accumulator = None
attention_count = 0
record_attention = False  # 控制开关

def mamba_activation_hook(module, input, output):
    global daam_heatmap_accumulator, attention_count, record_attention
    
    # 只有在指定的黄金时间步才记录注意力
    if not record_attention:
        return
        
    # 获取当前特征图并求均值 [B, 1, H, W]
    act = output.detach().mean(dim=1, keepdim=True)
    
    # 累加到全局变量
    if daam_heatmap_accumulator is None:
        daam_heatmap_accumulator = act
    else:
        daam_heatmap_accumulator += act
        
    attention_count += 1

# ==========================================
# 3. DDIM 逆向去噪采样器 (加入开关控制)
# ==========================================
@torch.no_grad()
def ddim_sample(model, mask, ddim_steps=50, total_timesteps=1000):
    global record_attention, daam_heatmap_accumulator, attention_count
    
    # 初始化/清空累加器
    daam_heatmap_accumulator = None
    attention_count = 0
    
    device = mask.device
    B = mask.shape[0]
    
    betas = torch.linspace(1e-4, 0.02, total_timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    x_t = torch.randn(B, 2, 32, 32, device=device)
    
    step_size = total_timesteps // ddim_steps
    time_steps = list(reversed(range(0, total_timesteps, step_size)))
    
    for i, t in enumerate(time_steps):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        # 【核心修改】只在中间 20% 到 80% 的步骤累积注意力，滤除早期噪声和晚期过度拟合
        if int(ddim_steps * 0.2) <= i <= int(ddim_steps * 0.8):
            record_attention = True
        else:
            record_attention = False
            
        pred_noise = model(x_t, mask, t_batch)
        
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t - step_size] if t - step_size >= 0 else torch.tensor(1.0, device=device)
        
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        dir_xt = torch.sqrt(1 - alpha_t_prev) * pred_noise
        x_t = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
    return x_t

# ==========================================
# 3. 推理与可视化 Demo
# ==========================================
def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化模型与 Hook, 使用设备: {device}...")
    
    model = MaskGuidedUNet(in_channels=2, mask_channels=1, base_dim=64).to(device)
    decoder = TrajectoryDecoder().to(device)
    
    checkpoint_path = 'segmadiff_drive_weights.pth'
    try:
        # 添加 weights_only=True 消除警告
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['unet'])
        decoder.load_state_dict(checkpoint['decoder'])
        print(f"✅ 成功加载权重文件: {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ 找不到权重文件 {checkpoint_path}，请先运行 train.py")
        return
        
    model.eval()
    decoder.eval()
    model.bottleneck.register_forward_hook(mamba_activation_hook)
    
    # ==========================================
    # 【核心修改】读取真实的 CARLA 训练数据掩码
    # ==========================================
    mask_dir = "./collected_data/bev_mask/"
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    if not mask_files:
        print("❌ 找不到真实的 BEV 掩码数据，请检查 collected_data 目录！")
        return
        
    # 取第一帧数据进行测试
    real_mask_path = os.path.join(mask_dir, mask_files[0])
    real_mask_img = cv2.imread(real_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 按照 Dataset 里的预处理方式进行转换并 Resize 到 32x32
    real_mask_tensor = torch.from_numpy(real_mask_img).unsqueeze(0).unsqueeze(0).float() / 25.0
    real_mask_tensor = F.interpolate(real_mask_tensor, size=(32, 32), mode='nearest').to(device)
    
    print("开始 DDIM 逆向去噪采样 (50 步)...")
    spatial_traj = ddim_sample(model, real_mask_tensor, ddim_steps=50)
    predicted_waypoints = decoder(spatial_traj).cpu().detach().numpy()[0]
    
    # ==========================================
    # 【新增】CARLA 语义分割调色板 (彩色可视化)
    # ==========================================
    # CARLA 标签对应颜色: 0无, 1建筑, 2栅栏, 3其他, 4行人(红), 5电线杆, 6道路线, 7道路(灰), 8人行道, 9植被(绿), 10车辆(蓝)...
    palette = np.array([
        [0, 0, 0], [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60],
        [153, 153, 153], [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35],
        [0, 0, 142], [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81],
        [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
        [170, 120, 50], [45, 60, 150], [145, 170, 100]
    ], dtype=np.uint8)

    # 还原真实标签值 (之前为了归一化除以了 25.0)
    mask_labels = (real_mask_tensor.cpu().numpy()[0, 0] * 25.0).astype(np.uint8)
    mask_labels = np.clip(mask_labels, 0, 22) # 防止越界
    color_mask = palette[mask_labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 绘制彩色 BEV 掩码
    axes[0].imshow(color_mask)
    axes[0].set_title("Real CARLA BEV Mask")
    axes[0].axis('off')
    
    # 2. 绘制 DAAM 热图
    global daam_heatmap_accumulator, attention_count
    axes[1].imshow(color_mask, alpha=0.5) 
    
    if daam_heatmap_accumulator is not None and attention_count > 0:
        # 求平均
        avg_heatmap = daam_heatmap_accumulator / attention_count
        # 归一化到 0-1
        avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min() + 1e-8)
        
        heatmap_np = F.interpolate(avg_heatmap, size=(32, 32), mode='bilinear').cpu().numpy()[0, 0]
        axes[1].imshow(heatmap_np, cmap='jet', alpha=0.6)
    
    axes[1].set_title("DAAM: Averaged Spatial Attention")
    axes[1].axis('off')
    
    # 3. 绘制预测的轨迹点
    axes[2].plot(predicted_waypoints[:, 0], predicted_waypoints[:, 1], 'ro-', linewidth=2, markersize=6)
    axes[2].set_title("Predicted Future Trajectory (x, y)")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_filename = "segmadiff_demo_real_v2.png"
    plt.savefig(save_filename, dpi=300)
    print(f"✅ 推理完成！结果已保存为 {save_filename}")

if __name__ == "__main__":
    run_demo()