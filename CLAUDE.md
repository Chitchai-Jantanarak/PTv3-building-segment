# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-stage deep learning pipeline for urban 3D point cloud processing using Point Transformer V3 (PTv3). The pipeline segments buildings from LiDAR data, inpaints missing geometry, and classifies buildings according to FEMA/HAZUS codes. Python 3.9-3.11, PyTorch 2.1.0, CUDA 11.8.

## Pipeline Stages

The system runs as a sequential 4-stage pipeline where later stages depend on earlier checkpoints:

| Stage | Task Config | Purpose | Key Detail |
|-------|-------------|---------|------------|
| 0 | `mae` | Self-supervised pretraining | 75% block masking on XYZ+relZ, MSE loss |
| 1 | `seg_a` | Semantic segmentation (13 classes) | Frozen MAE encoder, focal loss |
| 2 | `seg_b_geom` / `seg_b_color` | Building inpainting | Structured wall/roof masking, Chamfer loss |
| 3 | `hazus` | FEMA building classification | 7 main + 28 sub-classes, geometry+MAE error features |

Stage dependencies are enforced at runtime via `check_dependencies()` in `runner.py`.

## Working Directory

All commands run from `urban-building/` (not the repo root).

## Commands

### Environment Setup
```bash
cd urban-building
chmod +x scripts/setup_env.sh && ./scripts/setup_env.sh
conda activate urban-building
```

### Training (via Hydra runner)
```bash
python runner.py task=mae                    # Stage 0
python runner.py task=seg_a                  # Stage 1
python runner.py task=seg_b_geom             # Stage 2 geometry
python runner.py task=seg_b_color            # Stage 2 color
python runner.py task=hazus                  # Stage 3
```

### Override parameters via CLI
```bash
python runner.py task=seg_a data=whu training.epochs=150 data.batch_size=4
```

### Other modes
```bash
python runner.py mode=preprocess data=sensat
python runner.py mode=infer task=seg_a checkpoint=checkpoints/seg_a/best.pth
python runner.py mode=pipeline              # Full SEG-A -> SEG-B -> HAZUS inference
```

### Full pipeline (all stages sequentially)
```bash
bash scripts/run_full_pipeline.sh
```

### Linting and Formatting
```bash
ruff check src/                             # Lint
ruff format src/                            # Format
```

### Tests
```bash
pytest                                      # Run all tests
pytest -m "not slow and not gpu"            # Skip slow/GPU tests
pytest tests/test_foo.py::test_bar          # Single test
```

## Architecture

### Entry Points

- **`runner.py`** — Primary Hydra entry point. Dispatches to `run_preprocess`, `run_train`, `run_inference`, `run_evaluate`, `run_export`, or `run_full_pipeline` based on `mode` config.
- **`main.py`** — Legacy entry point used by shell scripts in `scripts/`. The shell scripts mutate `configs/config.yaml` via `sed` then call `main.py`.
- **Prefer `runner.py` with CLI overrides** over running scripts that sed-edit config files.

### Configuration System (Hydra)

`configs/config.yaml` composes defaults from three config groups:
- `task/` — Training hyperparameters, loss, optimizer, scheduler per stage
- `model/` — PTv3 encoder architecture (channels, depths, patch size, flash attention)
- `data/` — Dataset paths, format (PLY/LAS/H5), class mappings

`configs/default.yaml` sets runtime defaults (seed, device, paths, CUDA cache clearing). Hydra outputs logs to `logs/{task_name}/{timestamp}/`.

### Source Code (`src/`)

- **`core/`** — Shared I/O (LAS, PLY, DEM, H5), preprocessing (voxelization, normalization), utilities (logging, seeding, checkpointing, memory tracking)
- **`datasets/`** — `base.py` defines `BasePointCloudDataset`; `builder.py` is a registry-based factory that builds datasets/dataloaders from config
- **`models/`** — Composable: PTv3 encoder (`encoders/ptv3/`) + task-specific heads (`mae/`, `seg_heads/`, `hazus_head/`). HAZUS head uses a conditioned sub-head architecture (main category -> sub-category -> attributes)
- **`train/`** — `_base.py` has generic train/validate loops, optimizer/scheduler builders. Each task file (e.g., `mae.py`, `seg_a.py`) provides a `train(cfg)` function dynamically imported by `runner.py`
- **`infer/`** — Per-task inference modules + `pipeline.py` for end-to-end orchestration
- **`losses/`** — MSE (MAE), Chamfer distance (inpainting), Focal loss (segmentation)

### Data Flow

Raw point clouds (LAS/PLY) -> preprocessing (voxelization, relZ from DEM) -> `.pth` dicts with keys: `coords`, `features`, `labels`, `batch`, `offset` -> model input.

PTv3 input: 4 channels (XYZ + relative height). Color head is optional and off by default.

### Git Submodules

- **`PointTransformerV3/`** — Official PTv3 backbone (Pointcept)
- **`PotreeConverter/`** — Point cloud visualization tool

## Key Conventions

- **Checkpoints**: Stored as `checkpoints/{task_name}/best.pth` (PyTorch state dicts with optimizer state). Later stages check for existence of prerequisite checkpoints.
- **Training modules expose `train(cfg)`** — dynamically imported via `importlib.import_module(f"src.train.{task_name}")`.
- **Datasets registered in `builder.py`** — add new datasets by registering in the builder, not by modifying task configs.
- **Ruff** is the primary linter/formatter (target: py39, line length: 88). Config in `pyproject.toml`.
- **GPU memory**: Default configs target NVIDIA A40 (48GB). For smaller GPUs, reduce `batch_size` and `max_points` via CLI overrides.
- **Precision**: FP32 by default, configurable via `run.precision`.
- **Seed**: 42 by default, set globally at startup.
