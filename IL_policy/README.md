# IL Policy (Diffusion Policy for PlanarPeg)

This directory contains the Imitation Learning (IL) module for the `PlanarPegInsertion-v0` environment, implementing a **Diffusion Policy** architecture adapted from [Chi et al. (2023)](https://diffusion-policy.cs.columbia.edu/) and perfectly aligned with the structural configuration of `manipulation_pipeline`.

## 📁 Directory Structure

```text
IL_policy/
├── configs/
│   └── base.yaml              # Single unified configuration file (Hydra)
├── models/
│   ├── conditional_unet1d.py  # 1D U-Net backbone for noise prediction
│   └── vision_encoder.py      # ResNet18-based spatial softmax vision encoder
├── utils/
│   ├── dataset.py             # Zarr-based dataset loader (PlanarPegDataset)
│   └── normalize.py           # MinMax Normalization/Unnormalization modules
├── diffusion_policy.py        # Core Diffusion Policy implementation (DDIM/DDPM)
├── train.py                   # Main training loop with WandB integration
├── eval.py                    # Evaluation script for the Gym environment
└── requirements.txt           # Required dependencies
```

## 🛠️ Architecture Highlights

- **Vision Encoder**: Utilizes `timm`'s ResNet18 without pooling, followed by a Spatial Softmax layer to extract 2D keypoints dynamically from multi-view images (`top` and `front`).
- **Diffusion Model**: Employs a 1D Conditional U-Net (`ConditionalUnet1d`) using FiLM modulation for visual and state conditioning.
- **Schedulers**: Supports both `DDPM` and `DDIM` (via Hugging Face `diffusers`). Easily swappable via `base.yaml`.
- **Normalization**: Dynamically computes Min-Max boundaries across the entire dataset (`stats.json`) and strictly bounds network inputs/outputs to `[-1, 1]`.
- **Action Buffering**: The policy maintains an internal `collections.deque` queue. It evaluates the environment once, generates an entire trajectory (`pred_horizon`), buffers `action_horizon` steps, and pops them sequentially to drastically accelerate inference.

## 🚀 Quick Start

## 📦 0. Dependencies

Ensure the local `planar_peg` conda environment has the dependencies installed:
```bash
pip install -r IL_policy/requirements.txt
```
*(Key dependencies include: `torch`, `diffusers`, `timm`, `hydra-core`, `zarr`, and `wandb`)*

### 1. Training

Training is managed via [Hydra](https://hydra.cc/). All hyperparameters (optimizer, scheduler, UNet sizing, and DDIM parameters) can be configured directly inside `configs/base.yaml` or overridden via CLI.

```bash
# Run training with default settings
python IL_policy/train.py

# Override batch size and steps via CLI
python IL_policy/train.py train.batch_size=32 train.steps=50000
```

*Note: Training will automatically create an isolated log directory under the repository root:*  
`outputs/diffusion-[YYYYMMDD]_[HHMMSS]/` containing `checkpoints/`, `wandb/`, `logs/`, and `stats.json`.

## ⚙️ Configuration (`base.yaml`)

The `base.yaml` file exposes fine-grained architectural controls. Key tunable parameters include:

- `policy.noise_scheduler.type`: Switch between `DDIM` and `DDPM`.
- `policy.noise_scheduler.num_inference_steps`: Controls DDIM generation speed vs. quality.
- `policy.unet.down_dims`: Scale the capacity of the U-Net.
- `optimizer_lr` & `scheduler_warmup_steps`: Training convergence tuning.
- `policy.obs_horizon` & `policy.action_horizon`: Context length and receding horizon length.

### 2. Evaluation

To evaluate a trained checkpoint in the `gymnasium` environment, pass the generated `eval_dir` to the evaluation script. The script will automatically load the checkpoint (`best.pth`) and the corresponding normalization statistics (`stats.json`).

```bash
python IL_policy/eval.py eval_dir=outputs/diffusion-20260616_140455
```

You can also specify a specific checkpoint file using `checkpoint_name`:
```bash
python IL_policy/eval.py eval_dir=outputs/diffusion-20260616_140455 checkpoint_name=epoch_50.pth
```
