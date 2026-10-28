# SegMaDiff-Drive: Mask-Guided Diffusion for Autonomous Driving

This repository contains the official PyTorch implementation for our FYP project: **End-to-End Autonomous Driving Motion Prediction based on Mamba State-Space and Diffusion Models**. 

We transferred the SegMaDiff architecture from the medical image domain (MRI) to autonomous driving (CARLA Simulator), replacing the standard U-Net bottleneck with the **Mamba Dynamic Former (MDF)** for superior spatial topology reasoning.

## 👥 Team Members
- Zhaoyang XU 
- Yongxin GUAN 
- Zixuan LIN 

## 📁 Repository Structure
- `MDFormer.py`: Core implementation of the Mamba Dynamic Former, including Gaussian High-Pass Filter and OmniScan.
- `unet.py`: The Mask-Guided Diffusion U-Net architecture integrated with MDFormer.
- `train.py`: Training script with Mixed Loss (Diffusion MSE + Coordinate Reconstruction + Kinematic Smoothness).
- `inference_demo2.py`: Inference script with DDIM sampling, high-resolution attention extraction, and semantic masking visualization.

## 🚀 Getting Started

### 1. Environment Setup
Please ensure you have Ubuntu 22.04 and a CUDA-enabled GPU.
```bash
conda activate carla
pip install torch torchvision torchaudio
pip install pandas matplotlib numpy opencv-python diffusers


### 2. Train the Model
python3 train.py
### 3. Inference & XAI Visualization
python3 inference_demo2.py
