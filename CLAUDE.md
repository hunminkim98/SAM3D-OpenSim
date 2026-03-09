# Claude Code Context

Context file for Claude Code AI assistant.

## Project Overview

SAM3D-OpenSim converts monocular video to OpenSim motion data using SAM3D Body for 3D pose estimation.

## Key Files

| File | Purpose |
|------|---------|
| `run_full_pipeline.py` | Main entry point - full video to OpenSim pipeline |
| `run_inference.py` | Stage 1: SAM3D Body inference only |
| `run_export.py` | Stage 2: Export JSON to TRC/MOT/FBX |
| `run_pipeline.py` | Combined TRC generation pipeline |
| `src/sam3d_inference.py` | SAM3D Body wrapper with SAM3/MoGe2 integration |
| `src/coordinate_transform.py` | Camera to OpenSim coordinate transform + global translation |
| `src/post_processing.py` | Butterworth smoothing, bone normalization |
| `src/keypoint_converter.py` | MHR70 to OpenSim marker mapping |
| `src/trc_exporter.py` | TRC file generation |
| `config/config.yaml` | Main configuration |

## Environments

| Environment | Purpose |
|-------------|---------|
| `sam_3d_body` | SAM3D Body inference (PyTorch, CUDA) |
| `Pose2Sim` | OpenSim IK (OpenSim Python bindings) |

## External Dependencies

| Component | Location |
|-----------|----------|
| SAM3D Body | `C:\Sam3dBody\sam-3d-body` |
| SAM3 | `C:\Sam3dBody\sam3\sam3` (nested) |
| MoGe2 | Installed in sam_3d_body env |
| Blender | `C:\Program Files\Blender Foundation\Blender 5.0\` |
| Pose2Sim | Installed in Pose2Sim env |

## Key Concepts

### MHR70 Skeleton
70-keypoint skeleton format from SAM3D Body:
- Body: indices 0-14 (COCO17-like)
- Feet: indices 15-20
- Right hand: indices 21-41
- Left hand: indices 42-62
- Extra anatomical: indices 63-69

### cam_t
Camera translation vector from SAM3D Body. Used for global translation tracking when `--global-translation` flag is set.

### Global Translation
Tracks subject movement through 3D space using cam_t. Requires `--fov moge2` for accurate results.

### Per-Frame Ground Alignment
Each frame is aligned so the lowest foot point (heel or toe) is at Y=0.

### Hand Markers
LIndex3, RIndex3, LMiddleTip, RMiddleTip markers help IK solve forearm pronation/supination.

## Common Tasks

### Run best quality pipeline
```bash
conda activate sam_3d_body
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

### Re-export with different settings
```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --global-translation --smooth 6.0
```

### Adjust smoothing
- More smooth: `--smooth 4.0`
- Less smooth: `--smooth 8.0`
- No smoothing: `--smooth 0`

## Coordinate Systems

### Camera (SAM3D Body)
- X: Right
- Y: Down
- Z: Forward

### OpenSim
- X: Forward
- Y: Up
- Z: Right

## Common Issues

### SAM3 import error
Path must be `C:/Sam3dBody/sam3/sam3` (nested structure).

### Missing dependencies for SAM3
```bash
pip install decord psutil ftfy regex triton-windows
```

### OpenSim errors
Use `Pose2Sim` environment, not `sam_3d_body`.

## Recent Changes (2026-02-06)

1. Added Butterworth smoothing (`--smooth` flag, default 6 Hz)
2. Added per-frame ground alignment (feet always touch floor)
3. Added hand markers (LIndex3, RIndex3, LMiddleTip, RMiddleTip) for arm rotation
4. Integrated SAM3 detector (`--detector sam3`)
5. Integrated MoGe2 FOV estimation (`--fov moge2`)
6. Added global translation from cam_t (`--global-translation`)
7. Created two-stage workflow (inference → export)
