# SegMaDiff: Segmentation-Guided Diffusion Model with Mamba-Dynamic Transformer

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<br>
<img src="logo_ICASSP.webp" width="300">

### 🎉 ACCEPTED at IEEE ICASSP 2026 (CORE B & CCF B)
### 🇪🇸 See you in Barcelona!

</div>

## Overview

**SegMaDiff** is an advanced diffusion-based image generation framework that combines segmentation guidance with a novel **Mamba-Dynamic Transformer (MDFormer)** architecture. The model enables high-quality, controllable image synthesis conditioned on semantic segmentation masks, with optional mask ablation training for robust handling of incomplete segmentation information.

### Key Features

- 🎯 **Segmentation-Guided Generation**: Precise control over image synthesis using semantic segmentation masks
- 🔄 **Mask Ablation Training**: Robust conditioning on incomplete or partially missing segmentation classes
- 🚀 **Mamba-Dynamic Transformer**: Novel hybrid architecture integrating state-space models with transformer mechanisms
- ⚡ **Dual Diffusion Schedulers**: Support for both DDPM and DDIM sampling strategies
- 🎨 **Multi-Channel Flexibility**: Compatible with 1-channel (grayscale), 3-channel (RGB), and arbitrary multi-channel medical images
- 🧬 **Medical Imaging Optimized**: Specifically designed for medical image generation tasks (MRI, CT, etc.)

### Architecture Highlights

The **MDFormer** module introduces several innovations:

- **Bidirectional State-Space Scanning**: Horizontal and vertical SSM passes for comprehensive spatial modeling
- **Dynamic Tanh Activation**: Learnable activation function for enhanced feature representation
- **Gaussian High-Pass Filtering**: Preserves fine-grained details during generation
- **Efficient Mid-Block Integration**: Seamless integration into standard U-Net architectures

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Basic Training](#basic-training)
  - [Segmentation-Guided Training](#segmentation-guided-training)
  - [Advanced Options](#advanced-options)
- [Evaluation & Sampling](#evaluation--sampling)
- [Model Architecture](#model-architecture)
- [Citation](#citation)
- [License](#license)

---

## Installation

### Prerequisites

- **Python 3.11** or higher
- **CUDA 12.2** or compatible version (check with `nvidia-smi`)
- **GPU**: Recommended 2 * RTX 4090 (48GB VRAM) for training at 256×256 resolution

### Setup

1. **Clone the repository**:

```bash
git clone https://github.com/YiZhao-Jasper/SegMaDiff.git
cd SegMaDiff
```

2. **Install PyTorch**:

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install PyTorch according to your CUDA version. This codebase was developed with:

```bash
pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

3. **Install dependencies**:

```bash
pip3 install -r requirements.txt
```

**Dependencies include**:
- `diffusers==0.21.4` - Diffusion model implementations
- `numpy==1.26.1` - Numerical operations
- `Pillow==10.0.1` - Image I/O
- `matplotlib==3.8.0` - Visualization
- `datasets==2.14.5` - Dataset management
- `tqdm==4.66.1` - Progress bars
- `safetensors` - Efficient model serialization

---

## Dataset Preparation

### Image Dataset Structure

Organize your image dataset with train/validation/test splits:

```
DATA_FOLDER/
├── train/
│   ├── tr_001.png
│   ├── tr_002.png
│   └── ...
├── val/
│   ├── val_001.png
│   ├── val_002.png
│   └── ...
└── test/
    ├── ts_001.png
    ├── ts_002.png
    └── ...
```

**Supported formats**:
- **Standard images**: `.png`, `.jpg` (PIL-readable, 1 or 3 channels)
- **Multi-channel images**: `.npy` NumPy arrays (for arbitrary channel counts)

### Segmentation Mask Structure

For segmentation-guided models, create a separate mask directory with an `all/` subdirectory:

```
MASK_FOLDER/
└── all/
    ├── train/
    │   ├── tr_001.png
    │   ├── tr_002.png
    │   └── ...
    ├── val/
    │   ├── val_001.png
    │   ├── val_002.png
    │   └── ...
    └── test/
        ├── ts_001.png
        ├── ts_002.png
        └── ...
```

**⚠️ Important Mask Requirements**:
- **Matching filenames**: Each mask must have the **exact same filename** as its corresponding image
- **Integer encoding**: Masks must contain integer class labels: `0, 1, 2, ...`
  - `0` = background class
  - `1, 2, 3, ...` = object classes
- **File format**: Save as `.png` or any PIL-compatible format with integer pixel values

**Example**: For a 4-class segmentation (background + 3 tissues):
```python
mask_values = {0: "background", 1: "tissue_A", 2: "tissue_B", 3: "tissue_C"}
```

---

## Training

### Basic Training

Train an **unconditional diffusion model** without segmentation guidance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset breast_mri \
    --img_dir /path/to/DATA_FOLDER \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 400
```

**Parameter descriptions**:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--mode` | Operation mode | `train` / `eval` / `eval_many` |
| `--model_type` | Diffusion scheduler | `DDIM` (faster) / `DDPM` (higher quality) |
| `--img_size` | Image resolution | `256`, `512` |
| `--num_img_channels` | Channel count | `1` (grayscale), `3` (RGB) |
| `--dataset` | Dataset name | `breast_mri`, `brain_mri` |
| `--img_dir` | Path to image directory | `/data/images/` |
| `--train_batch_size` | Training batch size | `16` (48GB GPU), `8` (24GB GPU) |
| `--eval_batch_size` | Evaluation batch size | `8` |
| `--num_epochs` | Total training epochs | `400` (default) |

**GPU recommendations**:
- **256×256 images**: Batch size 16 on 48GB A6000
- **512×512 images**: Batch size 4 on 48GB A6000

### Segmentation-Guided Training

Train a **segmentation-guided model** with mask conditioning:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset brain_mri \
    --img_dir /path/to/DATA_FOLDER \
    --seg_dir /path/to/MASK_FOLDER \
    --segmentation_guided \
    --num_segmentation_classes 4 \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 400
```

**Additional segmentation parameters**:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--seg_dir` | Path to mask directory | `/data/masks/` |
| `--segmentation_guided` | Enable mask conditioning | *(flag)* |
| `--num_segmentation_classes` | Total classes (including background) | `4` (background + 3 tissues) |
| `--segmentation_channel_mode` | Mask encoding mode | `single` (all classes in 1 channel) / `multi` |

### Advanced Options

#### Mask Ablation Training

Enable **mask ablation** to train models robust to incomplete segmentations:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset brain_mri \
    --img_dir /path/to/DATA_FOLDER \
    --seg_dir /path/to/MASK_FOLDER \
    --segmentation_guided \
    --num_segmentation_classes 4 \
    --use_ablated_segmentations \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 400
```

**What is mask ablation?**  
During training, random segmentation classes are removed from masks with 50% probability per class. This teaches the model to generate plausible images even when some anatomical structures are missing from the conditioning mask.

**Use cases**:
- Handling uncertain or ambiguous segmentations
- Generating images with partial anatomical information
- Improving model robustness to segmentation errors

#### Resume Training

Resume from a specific epoch checkpoint:

```bash
python3 main.py \
    --mode train \
    --resume_epoch 200 \
    [... other training args ...]
```

---

## Evaluation & Sampling

### Generate Multiple Samples

Generate **100 images** from a trained model:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset breast_mri \
    --eval_batch_size 8 \
    --eval_sample_size 100
```

**Output**: Generated images saved to `{OUTPUT_DIR}/samples_many_100/`

### Segmentation-Guided Sampling

Generate images **conditioned on test set masks**:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset brain_mri \
    --seg_dir /path/to/MASK_FOLDER \
    --segmentation_guided \
    --num_segmentation_classes 4 \
    --eval_batch_size 8 \
    --eval_sample_size 100
```

**Behavior**: Reads masks from `MASK_FOLDER/all/test/` and generates corresponding images.

### Mask Removal Visualization

Evaluate model behavior when **progressively removing anatomical structures**:

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --mode eval \
    --model_type DDIM \
    --img_size 256 \
    --num_img_channels 1 \
    --dataset brain_mri \
    --seg_dir /path/to/MASK_FOLDER \
    --segmentation_guided \
    --num_segmentation_classes 4 \
    --eval_mask_removal \
    --eval_batch_size 8
```

**Output**: PDF visualization showing generated images for:
1. Full segmentation mask
2. Mask with class 1 removed
3. Mask with class 2 removed
4. ... (for each class)

### Blank Mask Evaluation

Test model behavior with **empty conditioning**:

```bash
python3 main.py \
    --mode eval \
    --eval_blank_mask \
    [... other args ...]
```

### Evaluation Options

| Parameter | Description |
|-----------|-------------|
| `--eval_sample_size` | Number of images to generate (default: 1000) |
| `--eval_mask_removal` | Visualize progressive anatomy removal |
| `--eval_blank_mask` | Generate with empty masks (zeros) |
| `--eval_noshuffle_dataloader` | Disable dataloader shuffling for reproducibility |

---

## Model Architecture

### U-Net Backbone

Standard diffusion U-Net with attention:

```python
UNet2DModel(
    sample_size=256,
    in_channels=2,  # 1 image + 1 segmentation
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D", 
        "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D"
    )
)
```

### Mamba-Dynamic Transformer (MDFormer)

Custom mid-block module replacing standard ResNet:

```python
class MambaDynamicFormer(nn.Module):
    Components:
    - Input projection (1×1 conv) → U, Z branches
    - Depthwise convolution (local spatial mixing)
    - Dynamic Tanh activation (learnable α, weight, bias)
    - Gaussian high-pass filter (preserves details)
    - SSM parameter projection (Δ, B, C maps)
    - Bidirectional state-space scanning:
      * Horizontal scan (left-to-right)
      * Vertical scan (top-to-bottom)
    - Gating mechanism (U ⊙ σ(Z))
    - Output projection (back to residual dimension)
```

**State-space model formulation**:

```
h_{t+1} = exp(A·Δ_t) · h_t + (u_t · Δ_t) · B_t
y_t = (h_t · C_t) + D · u_t
```

Where:
- `h_t`: Hidden state (N-dimensional)
- `u_t`: Input at position t
- `A`: State transition matrix (learned)
- `Δ_t`, `B_t`, `C_t`: Position-dependent parameters
- `D`: Skip connection weight

**Key innovations**:
1. **2D spatial scanning**: Combines horizontal and vertical passes for full spatial context
2. **Dynamic activation**: Learnable Tanh scaling adapts to data distribution
3. **High-pass filtering**: Enhances edge preservation for medical images
4. **Lightweight parameters**: Efficient state-space model with `d_state=16`

### Custom Pipeline Components

- **SegGuidedDDPMPipeline**: Segmentation-conditioned DDPM sampling
- **SegGuidedDDIMPipeline**: Segmentation-conditioned DDIM sampling (faster)

Both support:
- Mask ablation during inference
- Classifier-free guidance (experimental)
- Image-to-image translation (experimental)

---

## Project Structure

```
SegMaDiff/
├── main.py                 # Main training/evaluation script
├── MDFormer.py             # Mamba-Dynamic Transformer implementation
├── training.py             # Training loop and configuration
├── eval.py                 # Evaluation pipelines and metrics
├── utils.py                # Utility functions (GaussianHighPass, image grid)
├── requirements.txt        # Python dependencies
├── LICENSE                 # Apache 2.0 License
└── README.md              # This file
```

**Key modules**:

| File | Description |
|------|-------------|
| `MDFormer.py` | Core architecture: `MambaDynamicFormer` + `CustomMidBlock` |
| `training.py` | `TrainingConfig` dataclass + `train_loop()` function |
| `eval.py` | Custom pipelines: `SegGuidedDDPMPipeline`, `SegGuidedDDIMPipeline` |
| `utils.py` | `GaussianHighPass` filter + grid visualization utilities |

---

## Output Directory Structure

After training, the following structure is created:

```
{model_type}-{dataset}-{img_size}[-segguided][-ablated]/
├── unet/
│   ├── config.json                      # Model configuration
│   ├── diffusion_pytorch_model.safetensors  # Model weights
│   └── model_index.json                 # Pipeline metadata
└── samples/
    ├── 0000.png                         # Epoch 0 samples
    ├── 0010.png                         # Epoch 10 samples
    ├── ...
    └── 0399.png                         # Final epoch samples
```

**Example directory name**:
- `ddim-brain_mri-256-segguided-ablated`

---

## Advanced Usage

### Custom Datasets

To use your own medical imaging dataset:

1. **Prepare data** following the directory structure
2. **Update `--dataset` name** (used for logging only)
3. **Set `--num_segmentation_classes`** to match your segmentation schema
4. **Adjust `--img_size`** and `--num_img_channels`** as needed

### Multi-Channel Medical Images

For images with >3 channels (e.g., multi-sequence MRI):

```bash
python3 main.py \
    --mode train \
    --num_img_channels 8 \
    --img_dir /path/to/npy_files \
    [... other args ...]
```

Images should be saved as `.npy` files with shape `(C, H, W)`.

### Tensorboard Monitoring

Training logs are saved to Tensorboard:

```bash
tensorboard --logdir=runs/
```

**Logged metrics**:
- Training loss (MSE between predicted and actual noise)
- Learning rate schedule
- Sample images (every `save_image_epochs`)

---

## Citation

If you use SegMaDiff in your research, please cite:

Y. Zhao, Q. Fan, X. Duan, and D. Chen, "SegMaDiff: Segmentation-Guided Mamba Dynamic Diffusion Model for Anatomy-Preserving MRI Generation," in Proc. 51st IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2026.

```bibtex
@inproceedings{zhao2026segmadiff,
  title={SegMaDiff: Segmentation-Guided Mamba Dynamic Diffusion Model for Anatomy-Preserving MRI Generation},
  author={Zhao, Yi and Fan, Q. and Duan, X. and Chen, D.},
  booktitle={Proc. 51st IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year={2026}
}
```

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**Key permissions**:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Patent use
- ⚠️ Trademark use (restricted)

---

## Acknowledgments

- Built with [🤗 Diffusers](https://github.com/huggingface/diffusers)
- Inspired by [Mamba](https://arxiv.org/abs/2312.00752) state-space models
- Medical imaging preprocessing with [PyTorch](https://pytorch.org)

---

## Contact

**Author**: Yi Zhao  
**GitHub**: [@YiZhao-Jasper](https://github.com/YiZhao-Jasper)

For questions, issues, or collaboration inquiries, please open an issue on GitHub.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
