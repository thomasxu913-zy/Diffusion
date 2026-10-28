# MDFormer.py (整合了高通滤波器的完整版)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# 将原本在 utils 里的高通滤波器直接内置，消除依赖
class GaussianHighPass(nn.Module):
    def __init__(self, channels, kernel_size=15, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, u):
        u_blurred = TF.gaussian_blur(u, kernel_size=[self.kernel_size, self.kernel_size], sigma=[self.sigma, self.sigma])
        return u - u_blurred

class MambaDynamicFormer(nn.Module):
    def __init__(self, in_channels, d_state=16, expand=2, kernel_size=3, dt_min=1e-3, dt_max=0.1):
        super(MambaDynamicFormer, self).__init__()
        self.C = in_channels
        self.N = d_state
        self.d_inner = expand * in_channels
        self.kernel_size = kernel_size
        self.highpass = GaussianHighPass(channels=self.d_inner, kernel_size=15, sigma=3.0)

        # 1. Input projection
        self.in_proj = nn.Conv2d(in_channels, 2*self.d_inner, kernel_size=1)
        # 2. Depthwise convolution
        self.dw_conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=kernel_size, 
                                 padding=kernel_size//2, groups=self.d_inner)

        # 3. Projection to SSM parameters
        self.dt_rank = 1
        param_channels = self.dt_rank + 2*self.N  
        self.x_proj = nn.Conv2d(self.d_inner, param_channels, kernel_size=1, bias=False)
        self.dt_proj = nn.Conv2d(self.dt_rank, self.d_inner, kernel_size=1, bias=True)
        
        # 4. State-space parameters
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.N))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        with torch.no_grad():
            self.A_log.data = torch.log(torch.logspace(0, 1, steps=self.N, base=10.0))
            self.A_log.data *= -1.0  
        
        with torch.no_grad():
            log_min, log_max = torch.log(torch.tensor(dt_min)), torch.log(torch.tensor(dt_max))
            init_dt = torch.exp(torch.rand(self.d_inner) * (log_max - log_min) + log_min)
            inv_softplus = lambda x: x + torch.log(torch.clamp(torch.exp(-x) - 1, min=1e-8))
            self.dt_proj.bias.copy_(inv_softplus(init_dt))
    
        # 5. Output projection
        self.out_proj = nn.Conv2d(self.d_inner, in_channels, kernel_size=1)
    
    def forward(self, x):
        device = x.device  
        B, C, H, W = x.size()

        u_and_z = self.in_proj(x)
        U = u_and_z[:, :self.d_inner, :, :]    
        Z = u_and_z[:, self.d_inner:, :, :]    

        U_conv = self.dw_conv(U)               
        
        # DynamicTanh 原作者实现
        alpha = nn.Parameter(torch.ones(1) * 0.5).to(device)
        weight = nn.Parameter(torch.ones(self.d_inner)).to(device)
        bias = nn.Parameter(torch.zeros(self.d_inner)).to(device)

        U_conv = torch.tanh(alpha * U_conv)  
        U_conv = U_conv * weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        U_conv = self.highpass(U_conv)
        
        params = self.x_proj(U_conv)           
        delta_in = params[:, :self.dt_rank, :, :]
        B_map    = params[:, self.dt_rank:self.dt_rank+self.N, :, :]
        C_map    = params[:, self.dt_rank+self.N:self.dt_rank+2*self.N, :, :]

        dt_val = F.softplus(self.dt_proj(delta_in))
        A = -torch.exp(self.A_log)  
        D = self.D  
        
        Y_horiz = torch.zeros(B, self.d_inner, H, W, device=x.device, dtype=x.dtype)
        Y_vert  = torch.zeros(B, self.d_inner, H, W, device=x.device, dtype=x.dtype)

        # Horizontal scan
        for i in range(H):
            H_state = torch.zeros(B, self.d_inner, self.N, device=x.device, dtype=x.dtype)
            for j in range(W):
                u_ij = U_conv[:, :, i, j]        
                dt_ij = dt_val[:, :, i, j]       
                B_ij  = B_map[:, :, i, j]        
                C_ij  = C_map[:, :, i, j]        
                expA_dt = torch.exp(A.unsqueeze(0) * dt_ij.unsqueeze(2))
                H_state = expA_dt * H_state + (u_ij * dt_ij).unsqueeze(2) * B_ij.unsqueeze(1)
                y_ij = (H_state * C_ij.unsqueeze(1)).sum(dim=2) + D * u_ij
                Y_horiz[:, :, i, j] = y_ij

        # Vertical scan
        for j in range(W):
            H_state = torch.zeros(B, self.d_inner, self.N, device=x.device, dtype=x.dtype)
            for i in range(H):
                u_ij = Y_horiz[:, :, i, j]      
                dt_ij = dt_val[:, :, i, j]      
                B_ij  = B_map[:, :, i, j]       
                C_ij  = C_map[:, :, i, j]       
                expA_dt = torch.exp(A.unsqueeze(0) * dt_ij.unsqueeze(2))
                H_state = expA_dt * H_state + (u_ij * dt_ij).unsqueeze(2) * B_ij.unsqueeze(1)
                y_ij = (H_state * C_ij.unsqueeze(1)).sum(dim=2) + D * u_ij
                Y_vert[:, :, i, j] = y_ij

        out_intermediate = Y_vert * torch.sigmoid(Z)
        out = self.out_proj(out_intermediate)
        return f + out if 'f' in locals() else x + out # 添加残差连接