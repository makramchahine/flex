# Custom Environment Generation and Data Simulation Guide

This comprehensive guide walks you through the complete process of creating custom environments, generating training data, and performing inference using the Flex framework. The system uses BLIP2 vision-language models for flight control with imitation learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Custom Environment Generation](#custom-environment-generation)
5. [Data Generation](#data-generation)
6. [Training](#training)
7. [Inference](#inference)
8. [Configuration Reference](#configuration-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Flex framework enables:
- **Vision-Language Flight Control**: Uses BLIP2 for multimodal understanding
- **Imitation Learning**: Learns from demonstration data
- **Custom Environment Support**: Create environments with custom objects and behaviors
- **Multi-behavior Modes**: Supports directional behaviors (left/right/above/below) with target objects (blue/red)

### System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Environment   │────▶│  Data Generator │────▶│ Training Data   │
│   (Simulator)   │     │                 │     │ (Images+Labels) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Inference     │◀────│  Trained Model  │◀────│    Training     │
│                 │     │                 │     │    Pipeline     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Prerequisites

### System Requirements

- Python 3.9
- CUDA-compatible GPU (recommended)
- Linux operating system (Ubuntu 20.04+ recommended)
- Minimum 16GB RAM
- 50GB+ free disk space for datasets

### Software Dependencies

Ensure you have the following installed:

```bash
# Create and activate conda environment
conda create -n flex python==3.9
conda activate flex

# Install PyTorch (check https://pytorch.org/get-started/locally/ for your CUDA version)
pip install torch torchvision torchaudio

# Install core dependencies
pip install ipdb
pip install hydra-core==1.3.2
pip install hydra_colorlog --upgrade
pip install tensorboard
pip install cvxopt
pip install pyrootutils
pip install h5py
pip install salesforce-lavis  # for BLIP
pip install vit-pytorch==1.2.4  # transformer-based policy
```

---

## Environment Setup

### Step 1: Clone and Install the Repository

```bash
git clone <repository-url>
cd flex

# Install the learning module
cd learning
pip install -e .
```

### Step 2: Configure BLIP Model

Copy the required BLIP model file:

```bash
# Copy eva_vit.py to the lavis installation directory
cp learning/src/models/components/extractors/eva_vit.py \
   ~/miniconda3/envs/flex/lib/python3.9/site-packages/lavis/models/eva_vit.py
```

### Step 3: Set Environment Variables

Create a `setup_env.sh` file or modify `learning/scripts/setup_macro.sh`:

```bash
#!/bin/bash

# Directory where your training/evaluation data is stored
export DATA_DIR="/path/to/your/BLIP2_DATASET"

# Directory for storing training outputs, logs, and checkpoints
export LOG_DIR="/path/to/output/directory"

# For Hydra debugging (shows full stack traces)
export HYDRA_FULL_ERROR=1

# OpenGL settings for headless rendering
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0}

# Add project paths
export PYTHONPATH=$PYTHONPATH:/path/to/flex/learning
```

Source the environment:

```bash
source setup_env.sh
```

---

## Custom Environment Generation

### Understanding the Data Structure

The Flex framework expects data organized in the following structure:

```
BLIP2_DATASET/
├── train/
│   ├── left_blue/
│   │   ├── run_001/
│   │   │   ├── 001.png
│   │   │   ├── 002.png
│   │   │   ├── ...
│   │   │   ├── data_out.csv
│   │   │   └── label.txt
│   │   └── run_002/
│   │       └── ...
│   ├── right_blue/
│   ├── left_red/
│   ├── right_red/
│   ├── above_blue/
│   ├── above_red/
│   ├── below_blue/
│   ├── below_red/
│   └── save_flight/
└── eval/
    └── (same structure as train)
```

### Behavior Modes

The system supports 9 behavior modes defined in the dataset loader:

| Mode | Description | Weight |
|------|-------------|--------|
| `left_blue` | Navigate left relative to blue object | 0.1 |
| `left_red` | Navigate left relative to red object | 0.1 |
| `right_blue` | Navigate right relative to blue object | 0.1 |
| `right_red` | Navigate right relative to red object | 0.1 |
| `above_blue` | Navigate above blue object | 0.1 |
| `above_red` | Navigate above red object | 0.1 |
| `below_blue` | Navigate below blue object | 0.1 |
| `below_red` | Navigate below red object | 0.1 |
| `save_flight` | Legacy flight data | 0.2 |

### Creating Custom Behavior Types

To add custom behavior types, modify the dataset loader in `learning/src/data/components/flight_il_dataset_clean.py`:

```python
class FlightILDataset(IterableDataset):
    def __init__(self, ...):
        # Add your custom behavior types here
        self.run_types = [
            'left_blue', 'left_red', 
            'right_blue', 'right_red', 
            'above_blue', 'above_red', 
            'below_blue', 'below_red', 
            'save_flight',
            'custom_behavior'  # Add your custom type
        ]
        # Adjust weights (must sum to 1.0, must match number of run_types)
        self.run_types_wts = [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.18, 0.10]
```

---

## Data Generation

### Step 1: Create Run Directory

For each training run, create a directory structure:

```bash
mkdir -p $DATA_DIR/train/your_behavior_type/run_001
```

### Step 2: Collect Images

Images should be:
- **Format**: PNG
- **Resolution**: Will be resized to 224x224 during training
- **Naming**: Sequential numbered files (e.g., `001.png`, `002.png`, etc.)

### Step 3: Create Labels File

Create `data_out.csv` with the following columns:

```csv
vx,vy,vz,yaw
0.5,0.0,0.0,0.0
0.4,0.1,0.0,0.05
...
```

**Column descriptions:**
- `vx`: Velocity in x-direction (forward)
- `vy`: Velocity in y-direction (lateral)
- `vz`: Velocity in z-direction (vertical)
- `yaw`: Rotational velocity (heading change)

### Step 4: Create Text Labels

Create `label.txt` with text descriptions:

```
direction---target_object
```

**Format**: `{direction}---{object}`

**Examples**:
```
left---blue_object
right---red_object
above---blue_sphere
```

The text label provides the language conditioning for the vision-language model.

### Step 5: Organize Multiple Trajectories

For runs with multiple trajectories in the same folder:

```
run_001/
├── 001.png
├── 002.png
├── ...
├── data_out.csv      # First trajectory labels
├── data_out1.csv     # Second trajectory labels
├── data_out2.csv     # Third trajectory labels
└── label.txt         # Contains multiple lines, one per trajectory
```

---

## Training

### Configuration Files

Training is controlled by Hydra configuration files in `learning/configs/`:

```
configs/
├── train.yaml              # Main training config
├── data/
│   └── flight.yaml         # Data loading config
├── model/
│   ├── flight.yaml         # Model architecture config
│   ├── extractor/
│   │   └── blip.yaml       # BLIP feature extractor config
│   └── policy/
│       ├── lstm.yaml       # LSTM policy config
│       └── mlp.yaml        # MLP policy config
└── experiment/
    └── flight_blip_lstm.yaml  # Experiment-specific overrides
```

### Basic Training Command

```bash
cd learning
bash scripts/train_flight.sh
```

### Custom Training Configuration

```bash
cd learning

# Train with custom parameters
python src/train.py \
    task_name=my_experiment \
    experiment=flight_blip_lstm \
    data.batch_size=32 \
    data.num_workers=8 \
    trainer.max_steps=500000 \
    seed=42
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.batch_size` | Training batch size | 32 |
| `data.num_workers` | Data loading workers | 23 |
| `trainer.max_steps` | Maximum training steps | 1000000 |
| `trainer.val_check_interval` | Validation frequency | 5000 |
| `model.optimizer.lr` | Learning rate | 0.0001 |
| `data.flag_last_num` | Number of final steps to flag | 5 |

### Monitoring Training

Training logs are saved to TensorBoard:

```bash
tensorboard --logdir $LOG_DIR
```

### Resuming Training

To resume from a checkpoint:

```bash
python src/train.py \
    experiment=flight_blip_lstm \
    ckpt_path="/path/to/checkpoint.ckpt"
```

---

## Inference

### Loading a Trained Model

```python
import torch
from omegaconf import DictConfig
import hydra

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('/path/to/checkpoint.ckpt', map_location=device)

# Initialize model
model = hydra.utils.instantiate(cfg.model)
model.load_state_dict(ckpt['state_dict'])
model.to(device)
model.eval()
```

### Running Inference on Images

```python
import torch
from PIL import Image
from torchvision import transforms

# Prepare input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open('test_image.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Create input dictionary
input_data = {
    'image': image_tensor,
    'text': ['left---blue_object']  # Direction and target
}

# Run inference
with torch.no_grad():
    output = model(input_data)

# Output contains: vx, vy, vz, yaw, stop (if configured)
print(f"Velocity X: {output['vx'].item()}")
print(f"Velocity Y: {output['vy'].item()}")
print(f"Velocity Z: {output['vz'].item()}")
print(f"Yaw: {output['yaw'].item()}")
```

### Batch Inference

```python
# For multiple images
images = torch.stack([transform(Image.open(f)) for f in image_files])
texts = ['left---blue'] * len(image_files)

input_data = {
    'image': images.to(device),
    'text': texts
}

with torch.no_grad():
    outputs = model(input_data)
```

### Evaluation Script

Use the provided evaluation script:

```bash
cd learning

python src/eval.py \
    experiment=flight_blip_lstm \
    ckpt_path="/path/to/checkpoint.ckpt" \
    save_video=true
```

---

## Configuration Reference

### Data Configuration (`configs/data/flight.yaml`)

```yaml
_target_: src.data.flight_datamodule.FlightDataModule

train_data_path:
  - ${paths.data_dir}/train

eval_data_path:
  - ${paths.data_dir}/eval

features_folder: null    # Pre-extracted features (optional)
mode: IL                 # Imitation Learning mode
batch_size: 1
num_workers: 0
pin_memory: False
use_standardize: true
use_lavis_preprocess: false
```

### Model Configuration (`configs/model/flight.yaml`)

```yaml
_target_: src.models.flight_module.FlightLitModule

optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

net:
  _target_: src.models.components.e2e_net.E2ENet
  output_names: [vx, vy, vz, yaw, stop, step_remain]

criterion:
  _target_: src.models.components.criterion.FMCriterion
  weights:
    vx: 1
    vy: 1
    vz: 1
    yaw: 1
```

### Experiment Configuration Example

Create custom experiments in `configs/experiment/`:

```yaml
# @package _global_

defaults:
  - override /data: flight.yaml
  - override /model: flight.yaml
  - override /model/extractor@model.net.extractor: blip
  - override /model/policy@model.net.policy: lstm

tags: ["custom_experiment"]
seed: 42

trainer:
  max_steps: 500000
  val_check_interval: 2500

model:
  optimizer:
    lr: 0.0001

data:
  batch_size: 16
  num_workers: 8
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python src/train.py data.batch_size=16

# Use gradient accumulation
python src/train.py trainer.accumulate_grad_batches=2
```

#### 2. Data Loading Errors

Verify your data structure:
```bash
# Check directory structure
ls -la $DATA_DIR/train/

# Verify image count matches labels
python -c "
import pandas as pd
import os

run_path = '$DATA_DIR/train/left_blue/run_001'
images = len([f for f in os.listdir(run_path) if f.endswith('.png')])
labels = len(pd.read_csv(os.path.join(run_path, 'data_out.csv')))
print(f'Images: {images}, Labels: {labels}')
"
```

#### 3. BLIP Model Loading Issues

```bash
# Ensure BLIP checkpoint is downloaded
python -c "
from lavis.models import load_model_and_preprocess
model, vis_proc, txt_proc = load_model_and_preprocess(
    name='blip2_feature_extractor',
    model_type='pretrain',
    is_eval=True,
    device='cpu'
)
print('BLIP loaded successfully')
"
```

#### 4. Worker Initialization Errors

```bash
# Reduce number of workers
python src/train.py data.num_workers=0

# Or use file system sharing strategy
export PYTHONWARNINGS='ignore'
python src/train.py pytorch_sharing_strategy=file_system
```

### Debug Mode

Enable full error traces:

```bash
export HYDRA_FULL_ERROR=1
python src/train.py debug=default
```

### Performance Optimization

1. **Use Pre-extracted Features**: Extract BLIP features once and load them during training
   ```yaml
   data:
     features_folder: /path/to/features
     load_features_directly: true
   ```

2. **Enable Mixed Precision Training**:
   ```yaml
   trainer:
     precision: 16
   ```

3. **Use Distributed Training**:
   ```yaml
   trainer:
     strategy: ddp
     devices: 4
   ```

---

## Additional Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [LAVIS Library](https://github.com/salesforce/LAVIS)

---

## Contributing

When adding new features or environment types:

1. Create appropriate configuration files in `configs/`
2. Add new behavior types to the dataset loader
3. Update this documentation with new parameters
4. Add tests in `learning/tests/`

---

*Last updated: November 2024*
