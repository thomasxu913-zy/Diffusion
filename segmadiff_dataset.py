import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class SegMaDiffDriveDataset(Dataset):
    def __init__(self, data_root, seq_len=4, pred_len=10):
        """
        data_root: 存放 RGB、BEV 和 Trajectory 的根目录
        seq_len: 过去的历史帧数
        pred_len: 需要预测的未来轨迹点数
        """
        self.data_root = data_root
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 假设数据按照 frame_0001 格式命名
        self.frames = sorted([f for f in os.listdir(os.path.join(data_root, 'rgb')) if f.endswith('.png')])
        
        # 预处理：RGB去均值标准化，掩码转Tensor
        self.rgb_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # 留出预测未来的帧数空间
        return len(self.frames) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        curr_frame_idx = idx + self.seq_len - 1
        frame_name = self.frames[curr_frame_idx]
        
        # 1. 读取前视 RGB 图像
        rgb_path = os.path.join(self.data_root, 'rgb', frame_name)
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_tensor = self.rgb_transform(rgb_img)
        
        # 2. 读取 BEV 语义掩码 (作为条件引导)
        mask_path = os.path.join(self.data_root, 'bev_mask', frame_name)
        bev_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 增加通道维度使其变为 (1, H, W)，用于后续与网络特征级联或相乘
        mask_tensor = torch.from_numpy(bev_mask).unsqueeze(0).float() / 25.0 # CARLA最多25类
        
        # 3. 读取未来轨迹 (Waypoints Ground Truth)
        traj_path = os.path.join(self.data_root, 'trajectory', frame_name.replace('.png', '.json'))
        with open(traj_path, 'r') as f:
            waypoints_data = json.load(f)
        # 提取未来 pred_len 步的 (x, y) 坐标
        future_waypoints = np.array(waypoints_data['future_waypoints'][:self.pred_len])
        traj_tensor = torch.from_numpy(future_waypoints).float()
        
        return rgb_tensor, mask_tensor, traj_tensor

# 测试一下数据管道是否通畅
if __name__ == "__main__":
    # 假设你把刚采集的数据放到了这个目录
    dataset = SegMaDiffDriveDataset(data_root="./collected_data")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    for rgb, mask, traj in dataloader:
        print(f"RGB shape: {rgb.shape}")       # 期望: [4, 3, H, W]
        print(f"BEV Mask shape: {mask.shape}") # 期望: [4, 1, H, W]
        print(f"Trajectory shape: {traj.shape}") # 期望: [4, 10, 2]
        break
    print("✅ SegMaDiff-Drive 数据管道测试通过！")