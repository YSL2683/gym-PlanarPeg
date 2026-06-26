# IL Policy (Diffusion Policy for PlanarPeg)

This directory contains the Imitation Learning (IL) module for the `PlanarPegInsertion-v0` environment, implementing a highly optimized **Diffusion Policy** architecture adapted from [Chi et al. (2023)](https://diffusion-policy.cs.columbia.edu/).

The configuration and evaluation pipelines have been refactored to align perfectly with the structural standards of `manipulation_pipeline`.

## 📁 Directory Structure

```text
IL_policy/
├── configs/
│   └── base.yaml              # Unified Hydra configuration file with dynamic path interpolation
├── models/
│   ├── conditional_unet1d.py  # 1D U-Net backbone for noise prediction
│   └── vision_encoder.py      # ResNet18-based spatial softmax vision encoder
├── utils/
│   ├── dataset.py             # Zarr-based dataset loader with RAM caching
│   ├── normalize.py           # MinMax Normalization/Unnormalization modules
│   └── checkpoints.py         # Utilities for resolving best/latest checkpoints
├── diffusion_policy.py        # Core Diffusion Policy implementation (DDIM/DDPM)
├── train.py                   # Main training loop (AMP, EMA, GradClip, offline Validation)
├── eval.py                    # Evaluation script mirroring manipulation_pipeline
└── requirements.txt           # Required dependencies
```

## 🛠️ Architecture & Optimizations

- **Automatic Mixed Precision (AMP)**: Accelerates U-Net and ResNet computations using `torch.autocast` (`float16`/`bfloat16`) and `GradScaler`, halving VRAM usage.
- **Zarr RAM Caching**: Dramatically eliminates Disk I/O bottlenecks by pre-loading Zarr datasets directly into RAM using `cache_all: true`.
- **Stabilization**: Employs **Exponential Moving Average (EMA)** (`power: 0.75`) for stable model checkpoints, coupled with Gradient Clipping (`max_grad_norm: 1.0`).
- **Sequence-Relative Delta Actions**: Transforms world-absolute action trajectories into relative delta trajectories referenced from the robot's current state ($S_t$). This resolves out-of-distribution (OOD) "teleportation" failures when evaluating on unseen starting positions.
- **Dynamic Checkpoint Resolution**: The `eval.py` script automatically searches for the best checkpoints (`best.pth`) without hardcoded file paths.
- **Vision Encoder**: Utilizes `timm`'s ResNet18 followed by a Spatial Softmax layer to extract 2D keypoints dynamically from multi-view images (`top` and `front`).

## 🚀 Quick Start

### 📦 0. Dependencies

Ensure the local conda environment has the dependencies installed:
```bash
pip install -r IL_policy/requirements.txt
```

### 1. Training

Training is managed via [Hydra](https://hydra.cc/). Directory paths (`output_dir`, `checkpoint_dir`, `log_dir`) are dynamically interpolated in `base.yaml`.

```bash
# Run training with default settings
python IL_policy/train.py

# Override hyperparameters via CLI
python IL_policy/train.py train.batch_size=128 train.steps=50000
```

*Note: Training logs and checkpoints are automatically saved to: `outputs/planar_peg/diffusion/[SESSION_DATE_TIME]/`*

### 2. Evaluation

To evaluate a trained model, pass the checkpoint directory (or explicit path) to the `eval.py` script. The script automatically determines the optimal checkpoint.

```bash
# Automatically load the best.pth from a specific run directory
python IL_policy/eval.py checkpoint_dir=outputs/planar_peg-diffusion-20260617_150000

# Specify the exact checkpoint path
python IL_policy/eval.py checkpoint_path=outputs/planar_peg-diffusion-20260617_150000/checkpoints/step_10000.pth

# Override evaluation parameters (e.g., test 20 episodes on a specific OOD map)
python IL_policy/eval.py checkpoint_dir=outputs/... val.eval_n_episodes=20 task.grid="hard_maze"
```

## ⚙️ Configuration (`base.yaml`)

Key tunable parameters in `configs/base.yaml`:
- **task.use_delta_action**: (Crucial for OOD) Toggles sequence-relative delta action training vs absolute coordinate training. Must be `true` for generalizable policies.
- **train.use_amp / train.cache_all**: Toggles for performance optimizations.
- **train.use_ema**: Toggles Exponential Moving Average weight tracking.
- **policy.noise_scheduler.num_inference_steps**: Controls DDIM generation speed vs. quality.
- **val.eval_n_episodes**: Number of evaluation episodes run by `eval.py`.
