import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ==========================================
# 1. 动态 Tanh 激活函数 (Dynamic Tanh)
# 提供自适应缩放和逐通道仿射变换，增强深层模型稳定性
# ==========================================
class DynamicTanh(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return torch.tanh(self.alpha * x + self.beta)

# ==========================================
# 2. 高斯高通滤波器 (GHFilter)
# 用于强化车道线、行人与动态障碍物的边缘结构特征
# ==========================================
class GHFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        # 提取低频背景
        blurred = TF.gaussian_blur(x, kernel_size=[self.kernel_size, self.kernel_size], sigma=[self.sigma, self.sigma])
        # 原图减去低频得到高频边缘特征
        return x - blurred

# ==========================================
# 3. 模拟 Omni Scan (Mamba 双向扫描过渡版)
# 用水平和垂直双向 GRU 模拟状态空间模型的长距离依赖
# ==========================================
class MockOmniScan(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 使用 Bi-GRU 暂代 Mamba 扫描，捕捉全局上下文
        self.gru_h = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.gru_v = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 水平扫描 (Horizontal Sweep)
        x_h = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        out_h, _ = self.gru_h(x_h)
        out_h = out_h.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # 垂直扫描 (Vertical Sweep)
        x_v = out_h.permute(0, 3, 2, 1).reshape(B * W, H, C)
        out_v, _ = self.gru_v(x_v)
        out_v = out_v.reshape(B, W, H, C).permute(0, 3, 2, 1)
        
        return out_v

# ==========================================
# 4. 核心模块：Mamba Dynamic Former (MDF) 
# 将用于替换传统 U-Net 的中心瓶颈层
# ==========================================
class MambaDynamicFormer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 通道扩张至 2D
        self.expand = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        
        # 驱动支路 (Driver Path) 预处理
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dytanh = DynamicTanh(in_channels)
        self.ghfilter = GHFilter()
        
        # 状态空间扫描
        self.omni_scan = MockOmniScan(in_channels)
        
        # 门控输出投影
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 扩张并分离为驱动 (Driver) 和门控 (Gate)
        expanded = self.expand(x)
        driver, gate = torch.split(expanded, C, dim=1)
        
        # 2. 驱动特征处理 (提取高频结构)
        u = self.dwconv(driver)
        u = self.dytanh(u)
        u_high = self.ghfilter(u)
        
        # 3. 双向空间扫描
        y_vert = self.omni_scan(u_high)
        
        # 4. 掩码/门控调节与降维投影
        gated = y_vert * torch.sigmoid(gate)
        out = self.proj(gated)
        
        # 5. 残差连接
        return out + x

# ==========================================
# 快速测试张量流转
# ==========================================
if __name__ == "__main__":
    # 模拟在 U-Net 最深处（瓶颈层）的特征图 (Batch=2, Channels=256, H=32, W=32)
    dummy_feature = torch.randn(2, 256, 32, 32)
    
    mdf_block = MambaDynamicFormer(in_channels=256)
    output = mdf_block(dummy_feature)
    
    print(f"输入特征形状: {dummy_feature.shape}")
    print(f"输出特征形状: {output.shape}")
    if dummy_feature.shape == output.shape:
        print("✅ MDF 模块内部特征维度流转完全正确！")