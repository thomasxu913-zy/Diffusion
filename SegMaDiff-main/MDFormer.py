# Copyright 2026 Yi Zhao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GaussianHighPass

class MambaDynamicFormer(nn.Module):
    def __init__(self, in_channels, d_state=16, expand=2, kernel_size=3, dt_min=1e-3, dt_max=0.1):
        super(MambaDynamicFormer, self).__init__()
        self.C = in_channels
        self.N = d_state
        self.d_inner = expand * in_channels
        self.kernel_size = kernel_size
        self.highpass = GaussianHighPass(channels=self.d_inner, kernel_size=15, sigma=3.0)

        # 1. Input projection (1x1 conv): C -> 2*d_inner (split into U and Z)
        self.in_proj = nn.Conv2d(in_channels, 2*self.d_inner, kernel_size=1)
        # 2. Depthwise convolution for local mixing (groups = d_inner keeps each channel separate)
        self.dw_conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=kernel_size, 
                                 padding=kernel_size//2, groups=self.d_inner)

        # 3. Projection from conv output to SSM parameters (dt, B, C maps).
        self.dt_rank = 1
        param_channels = self.dt_rank + 2*self.N  # output channels for x_proj
        self.x_proj = nn.Conv2d(self.d_inner, param_channels, kernel_size=1, bias=False)
        # Linear mapping from dt_rank to d_inner for Delta (with bias for base log-Deltas)
        self.dt_proj = nn.Conv2d(self.dt_rank, self.d_inner, kernel_size=1, bias=True)
        
        # 4. State-space parameters:
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, self.N))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Initialize A_log and dt_proj biases for stability
        with torch.no_grad():
            self.A_log.data = torch.log(torch.logspace(0, 1, steps=self.N, base=10.0))
            self.A_log.data *= -1.0  # Ensure A = -exp(A_log) < 0
        
        with torch.no_grad():
            log_min, log_max = torch.log(torch.tensor(dt_min)), torch.log(torch.tensor(dt_max))
            init_dt = torch.exp(torch.rand(self.d_inner) * (log_max - log_min) + log_min)
            inv_softplus = lambda x: x + torch.log(torch.clamp(torch.exp(-x) - 1, min=1e-8))
            self.dt_proj.bias.copy_(inv_softplus(init_dt))
    
        # 5. Output projection back to in_channels (C) after gating
        self.out_proj = nn.Conv2d(self.d_inner, in_channels, kernel_size=1)
    
    def forward(self, x):
        device = x.device  
        B, C, H, W = x.size()

        # Input linear projection + split
        u_and_z = self.in_proj(x)
        U = u_and_z[:, :self.d_inner, :, :]    # (B, d_inner, H, W)
        Z = u_and_z[:, self.d_inner:, :, :]    # (B, d_inner, H, W)

        # Depthwise conv + activation on U for local spatial context
        U_conv = self.dw_conv(U)               # (B, d_inner, H, W)
        
        # Apply DynamicTanh instead of SiLU
        # DynamicTanh: Applying tanh activation + learnable scaling and bias
        alpha = nn.Parameter(torch.ones(1) * 0.5).to(device)  # Ensure alpha is on the same device as x
        weight = nn.Parameter(torch.ones(self.d_inner)).to(device)  # Ensure weight is on the same device as x
        bias = nn.Parameter(torch.zeros(self.d_inner)).to(device)  # Ensure bias is on the same device as x

        U_conv = torch.tanh(alpha * U_conv)  # Apply tanh with alpha scaling
        U_conv = U_conv * weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        U_conv = self.highpass(U_conv)
        # Compute parameter maps (Delta, B, C) from conv output
        params = self.x_proj(U_conv)           # (B, dt_rank + 2N, H, W)
        delta_in = params[:, :self.dt_rank, :, :]
        B_map    = params[:, self.dt_rank:self.dt_rank+self.N, :, :]
        C_map    = params[:, self.dt_rank+self.N:self.dt_rank+2*self.N, :, :]

        dt_val = F.softplus(self.dt_proj(delta_in))
        A = -torch.exp(self.A_log)  # (d_inner, N)
        D = self.D  # (d_inner,)
        
        Y_horiz = torch.zeros(B, self.d_inner, H, W, device=x.device, dtype=x.dtype)
        Y_vert  = torch.zeros(B, self.d_inner, H, W, device=x.device, dtype=x.dtype)

        # Horizontal scan
        for i in range(H):
            H_state = torch.zeros(B, self.d_inner, self.N, device=x.device, dtype=x.dtype)
            for j in range(W):
                u_ij = U_conv[:, :, i, j]        # (B, d_inner)
                dt_ij = dt_val[:, :, i, j]       # (B, d_inner)
                B_ij  = B_map[:, :, i, j]        # (B, N)
                C_ij  = C_map[:, :, i, j]        # (B, N)
                expA_dt = torch.exp(A.unsqueeze(0) * dt_ij.unsqueeze(2))
                H_state = expA_dt * H_state + (u_ij * dt_ij).unsqueeze(2) * B_ij.unsqueeze(1)
                y_ij = (H_state * C_ij.unsqueeze(1)).sum(dim=2) + D * u_ij
                Y_horiz[:, :, i, j] = y_ij

        # Vertical scan
        for j in range(W):
            H_state = torch.zeros(B, self.d_inner, self.N, device=x.device, dtype=x.dtype)
            for i in range(H):
                u_ij = Y_horiz[:, :, i, j]      # input is now the horizontal output
                dt_ij = dt_val[:, :, i, j]      # reuse same Delta map
                B_ij  = B_map[:, :, i, j]       # reuse same B_map
                C_ij  = C_map[:, :, i, j]       # reuse same C_map
                expA_dt = torch.exp(A.unsqueeze(0) * dt_ij.unsqueeze(2))
                H_state = expA_dt * H_state + (u_ij * dt_ij).unsqueeze(2) * B_ij.unsqueeze(1)
                y_ij = (H_state * C_ij.unsqueeze(1)).sum(dim=2) + D * u_ij
                Y_vert[:, :, i, j] = y_ij

        # Gating: multiply vertical output by SiLU of Z
        out_intermediate = Y_vert * torch.sigmoid(Z)
        out = self.out_proj(out_intermediate)
        return out

class CustomMidBlock(nn.Module):
    def __init__(self, orig_mid_block, MDFormer):
        super(CustomMidBlock, self).__init__()
        self.orig_mid_block = orig_mid_block
        self.MDFormer = MDFormer

    def forward(self, hidden_states, temb=None):
        x = self.orig_mid_block(hidden_states, temb)
        x = self.MDFormer(x)
        return x

