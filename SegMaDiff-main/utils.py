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

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid
    
class GaussianHighPass(nn.Module):
    def __init__(self, channels, kernel_size=15, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.weight = self._create_highpass_kernel().repeat(channels, 1, 1, 1)
        
    def _create_highpass_kernel(self):
        k = self.kernel_size
        sigma = self.sigma

        x = np.linspace(-k // 2, k // 2, k)
        xx, yy = np.meshgrid(x, x)
        lowpass = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        lowpass /= lowpass.sum()

        impulse = np.zeros_like(lowpass)
        impulse[k//2, k//2] = 1.0
        highpass = impulse - lowpass
        kernel = torch.tensor(highpass, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel  # shape (1, 1, k, k)

    def forward(self, x):
        return F.conv2d(x, self.weight.to(x.device), padding=self.kernel_size//2, groups=self.channels)
