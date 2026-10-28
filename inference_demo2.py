import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

from unet import MaskGuidedUNet
from train import TrajectoryProjector, TrajectoryDecoder  

# ==========================================
# 1. 终极空间高分辨率 DAAM 提取
# ==========================================
daam_heatmap_accumulator = None
attention_count = 0
record_attention = False

def high_res_activation_hook(module, input, output):
    global daam_heatmap_accumulator, attention_count, record_attention
    if not record_attention:
        return
        
    # 【核心修改】使用 L2 范数替代均值，更锐利地提取高强度激活区域
    # output 现在的分辨率是 32x32
    act = torch.norm(output.detach(), p=2, dim=1, keepdim=True)
    
    if daam_heatmap_accumulator is None:
        daam_heatmap_accumulator = act
    else:
        daam_heatmap_accumulator += act
    attention_count += 1

# ==========================================
# 2. DDIM 逆向去噪采样器
# ==========================================
@torch.no_grad()
def ddim_sample(model, mask, ddim_steps=50, total_timesteps=1000):
    global record_attention, daam_heatmap_accumulator, attention_count
    
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
        
        # 黄金时段 (20% - 80%) 开启高分辨率 DAAM 记录
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
# 3. 推理与高级可视化 Demo
# ==========================================
# ==========================================
# 3. 推理与高级可视化 Demo
# ==========================================
def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化模型与 Hook, 使用设备: {device}...")
    
    model = MaskGuidedUNet(in_channels=2, mask_channels=1, base_dim=64).to(device)
    decoder = TrajectoryDecoder().to(device)
    
    checkpoint_path = 'segmadiff_drive_weights.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['unet'])
        decoder.load_state_dict(checkpoint['decoder'])
        print(f"✅ 成功加载权重文件: {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ 找不到权重文件 {checkpoint_path}，请先运行 train.py")
        return
        
    model.eval()
    decoder.eval()
    
    # 挂载 Mamba 高分辨率层的 Hook
    model.up2.register_forward_hook(high_res_activation_hook)
    
    mask_dir = "./collected_data/bev_mask/"
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    if not mask_files:
        print("❌ 找不到真实的 BEV 掩码数据，请检查 collected_data 目录！")
        return
        
    real_mask_path = os.path.join(mask_dir, mask_files[0])
    real_mask_img = cv2.imread(real_mask_path, cv2.IMREAD_GRAYSCALE)
    
    real_mask_tensor = torch.from_numpy(real_mask_img).unsqueeze(0).unsqueeze(0).float() / 25.0
    real_mask_tensor = F.interpolate(real_mask_tensor, size=(32, 32), mode='nearest').to(device)
    
    print("开始 DDIM 逆向去噪采样并提取高分辨率注意力 (50 步)...")
    spatial_traj = ddim_sample(model, real_mask_tensor, ddim_steps=50)
    
    with torch.no_grad():
        predicted_waypoints = decoder(spatial_traj).cpu().numpy()[0]
    
    palette = np.array([
        [0, 0, 0], [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60],
        [153, 153, 153], [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35],
        [0, 0, 142], [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81],
        [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
        [170, 120, 50], [45, 60, 150], [145, 170, 100]
    ], dtype=np.uint8)

    mask_labels = (real_mask_tensor.cpu().numpy()[0, 0] * 25.0).astype(np.uint8)
    mask_labels = np.clip(mask_labels, 0, 22)
    color_mask = palette[mask_labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(color_mask)
    axes[0].set_title("Real CARLA BEV Mask")
    axes[0].axis('off')
    
    global daam_heatmap_accumulator, attention_count
    axes[1].imshow(color_mask, alpha=0.5) 
    
    if daam_heatmap_accumulator is not None and attention_count > 0:
        # 1. 求平均
        avg_heatmap = daam_heatmap_accumulator / attention_count
        heatmap_np = avg_heatmap.cpu().numpy()[0, 0]
        
        # ==========================================
        # 【美化 1】强化百分位数截断 (平衡注意力)
        # 将截断范围扩大到 5% - 95%，压制左上角行人的极端高亮尖峰，
        # 让中间道路的注意力也能够清晰地浮现出来
        # ==========================================
        v_min, v_max = np.percentile(heatmap_np, 5), np.percentile(heatmap_np, 95)
        heatmap_np = np.clip(heatmap_np, v_min, v_max)
        
        # 重新归一化到 0-1
        heatmap_np = (heatmap_np - v_min) / (v_max - v_min + 1e-8)
        
        # ==========================================
        # 【美化 2】强效高斯平滑 (消除像素马赛克感)
        # 加大滤波核，让注意力完全变成自然的“能量云雾”
        # ==========================================
        heatmap_np = cv2.GaussianBlur(heatmap_np, (7, 7), 3.0)
        
        # 直接使用透明度叠加，彻底废弃之前生硬的 mask 相乘
        axes[1].imshow(heatmap_np, cmap='jet', alpha=0.55) # 0.55的透明度能完美透出底部的 BEV
        
        # ==========================================
        # 【美化 3】优化轨迹显示 (视觉焦点)
        # 适度放大轨迹在 32x32 像素图上的视觉映射比例，使其更加醒目
        # ==========================================
        visual_scale = 1.5 
        pixel_x = (predicted_waypoints[:, 0] * visual_scale) + 16 
        pixel_y = 16 - (predicted_waypoints[:, 1] * visual_scale)
        
        # 使用加粗的白色线条，外加轻微发光效果
        axes[1].plot(pixel_x, pixel_y, 'wo-', linewidth=3, markersize=5, alpha=0.9, label='Planned Path')
        axes[1].legend(loc='lower left', fontsize=9)
    
    axes[1].set_title("Semantic Masked Attention")
    axes[1].axis('off')
    
    axes[2].plot(predicted_waypoints[:, 0], predicted_waypoints[:, 1], 'ro-', linewidth=2, markersize=6)
    axes[2].set_title("Predicted Future Trajectory (x, y)")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_filename = "segmadiff_demo_final.png"
    plt.savefig(save_filename, dpi=300)
    print(f"✅ 推理完成！结果已保存为 {save_filename}")

if __name__ == "__main__":
    run_demo()