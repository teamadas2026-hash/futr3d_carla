# FUTR3D Agent Instructions

## Project Structure

- `plugin/futr3d/` - Main implementation (models, configs, core utilities)
- `mmdet3d/` - Fork of mmdetection3d 1.0.0rc6 (dataset builds, model registry)
- `mmcv/` - Bundled mmcv (do not use system-installed mmcv)
- `tools/` - Train/test scripts, data converters, misc utilities
- `configs/` - Base configs inherited by plugin configs

## Installation

```bash
pip install -v -e .
```

Dependencies (enforced at import time):
- mmcv-full >=1.5.2, <=1.7.0
- mmdet >=2.24.0, <=3.0.0
- mmseg >=0.20.0, <=1.0.0
- nuscenes-devkit

## Training

```bash
# LiDAR-only (no pretrained model needed)
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py 8

# LiDAR-Cam (requires fused pretrained model, see Model Fusion below)
bash tools/dist_train.sh plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py 8

# Cam-Radar (requires DETR3D pretrained)
bash tools/dist_train.sh plugin/futr3d/configs/cam_radar/cam_res101_radar.py 8
```

## Evaluation

```bash
bash tools/dist_test.sh <config> <checkpoint> <num_gpus> --eval bbox
```

Example:
```bash
bash tools/dist_test.sh plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_res101.py ../lidar_cam.pth 8 --eval bbox
```

## Model Fusion (for LiDAR-Cam pretraining)

Fuse separate cam-only and lidar-only checkpoints:
```bash
python tools/fuse_model.py --img <cam_checkpoint> --lidar <lidar_checkpoint> --out <output_path>
```

Note: `tools/fuse_model.py` has hardcoded paths - update before use.

## Data Preparation

Use `tools/data_converter/nuscenes_converter.py` to generate `infos.pkl`. This version adds radar info (different from upstream mmdet3d).

Follow mmdet3d nuscenes guidance: https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/advanced_guides/datasets/nuscenes.md

## Plugin Architecture

Configs set `plugin = 'plugin/futr3d'` which triggers dynamic import in `tools/train.py` / `tools/test.py`. The plugin registers:
- `FUTR3D` detector
- `FUTR3DHead`, `FUTR3DTransformer`, `FUTR3DAttention`
- `HungarianAssigner3D`, `RadarPoint` (radar point cloud type)
- Custom data loaders in `plugin/futr3d/datasets/`

## Key Config Locations

| Model Type | Config Path |
|------------|-------------|
| LiDAR-only | `plugin/futr3d/configs/lidar_only/` |
| LiDAR-Cam | `plugin/futr3d/configs/lidar_cam/` |
| Cam-Radar | `plugin/futr3d/configs/cam_radar/` |
| Full fusion | `plugin/futr3d/configs/fusion/` |

## Testing

```bash
# Single test
python tools/test.py <config> <checkpoint> --eval bbox

# With visualization
python tools/test.py <config> <checkpoint> --show --show-dir <output_dir>
```

## Lint / Typecheck

No configured lint or typecheck commands found. The project uses flake8/yapf/interrogate (see `requirements/tests.txt`) but no pre-commit or CI enforcement.

## Notes

- This is a research codebase built on OpenMMLab stack; expect minimal polish
- `mmcv/` subdirectory is bundled - do not install separate mmcv package
- Hardcoded paths in `tools/fuse_model.py` and some visualization scripts need manual editing

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
