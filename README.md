# SAM3D-OpenSim

Convert monocular video to OpenSim motion data using SAM3D Body for 3D pose estimation.

## Overview

This pipeline uses [SAM3D Body](https://github.com/facebookresearch/sam-3d-body) (Meta/Facebook Research) for state-of-the-art 3D pose estimation, producing OpenSim-compatible motion files for biomechanical analysis.

**Pipeline Flow:**
```
Video → SAM3 Detector → SAM3D Body → MHR70 Keypoints → TRC Markers → OpenSim IK → Joint Angles (.mot)
              ↓                              ↓                              ↓
         MoGe2 FOV                      cam_t (global pos)           Blender → FBX Animation
```

## Features

- **70 MHR70 keypoints** including body, hands, and feet
- **SAM3 Detector**: State-of-the-art person detection using Segment Anything 3
- **MoGe2 FOV Estimation**: Accurate focal length for better 3D reconstruction
- **Global translation tracking** from cam_t for walking/running movement
- **Per-frame ground alignment**: Feet always touch the floor
- **Hand markers for arm rotation**: Better internal/external rotation tracking
- **Butterworth smoothing**: Configurable low-pass filter to reduce jitter
- **OpenSim IK** with 40 DOF using Pose2Sim model
- **FBX export** via Blender with rigged skeleton template (includes forearm rotation)
- **Two-stage workflow**: Separate inference (slow) from export (fast) for rapid iteration

## Performance

Tested on NVIDIA RTX GPU with 1136 frames (37.8 sec video):

| Stage | Time | Speed |
|-------|------|-------|
| SAM3D Inference | ~25 min | ~1.3 sec/frame |
| Export (TRC/MOT/FBX) | ~10 sec | ~114 frames/sec |
| OpenSim IK | 4 sec | ~284 frames/sec |

## Quick Start

```bash
conda activate sam_3d_body
cd C:\Sam3DBodyToOpenSim

# Best quality: SAM3 detector + MoGe2 FOV + global translation
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

## Usage

### Full Pipeline (One Command)

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

### Two-Stage Workflow (Recommended for Iteration)

**Stage 1: Inference** (slow, run once)
```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2
```

**Stage 2: Export** (fast, iterate on settings)
```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 --global-translation
```

### CPU Mode

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 --device cpu
```

## Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--input, -i` | Input video file | Required |
| `--height` | Subject height (meters) | 1.75 |
| `--mass` | Subject mass (kg) | 70.0 |
| `--output, -o` | Output directory | Auto |
| `--detector` | Human detector: vitdet, yolo11, **sam3**, none | vitdet |
| `--fov` | FOV estimator: **moge2**, none | moge2 |
| `--global-translation` | Track global movement from cam_t | false |
| `--smooth` | Smoothing cutoff frequency in Hz (0 to disable) | 6.0 |
| `--skip-ik` | Skip OpenSim IK | false |
| `--skip-fbx` | Skip FBX export | false |

## Output Files

```
output_YYYYMMDD_HHMMSS_videoname/
├── frames/                           # Extracted video frames
├── video_outputs.json                # SAM3D format (keypoints, cam_t, focal_length)
├── inference_meta.json               # Video metadata (FPS, dimensions)
├── markers_videoname.trc             # OpenSim marker trajectories (22 markers)
├── markers_videoname_ik.mot          # Joint angles (40 DOF)
├── markers_videoname_model.osim      # OpenSim model with markers
└── markers_videoname.fbx             # Animated skeleton for 3D software
```

## Pipeline Stages

1. **Frame Extraction**: Video → PNG frames at target FPS
2. **Person Detection**: SAM3/VitDet bounding box detection
3. **FOV Estimation**: MoGe2 focal length estimation (first frame)
4. **3D Pose Estimation**: SAM3D Body → 70 MHR70 keypoints + cam_t
5. **Post-Processing**: Smoothing, bone normalization, coordinate transform
6. **TRC Export**: OpenSim-compatible marker trajectories
7. **Inverse Kinematics**: OpenSim IK → 40 DOF joint angles
8. **FBX Export**: Blender animated skeleton

## Documentation

- [SETUP.md](SETUP.md) - Installation guide
- [HOW_TO_RUN.md](HOW_TO_RUN.md) - Usage examples
- [DOCUMENTATION.md](DOCUMENTATION.md) - Technical reference
- [docs/FULL_DOCUMENTATION.md](docs/FULL_DOCUMENTATION.md) - Complete documentation

## Requirements

- Windows 10/11
- Python 3.11
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+
- SAM3D Body
- SAM3 (Segment Anything 3)
- MoGe2 (Monocular Geometry Estimation)
- OpenSim 4.5+ (via Pose2Sim)
- Blender 5.0+ (for FBX export)

## Project Structure

```
SAM3D-OpenSim/
├── config/                    # Configuration files
├── models/                    # OpenSim model files
├── src/                       # Source modules
├── utils/                     # Utility functions
├── scripts/                   # Blender export script
├── docs/                      # Documentation
├── run_inference.py           # Stage 1: SAM3D inference
├── run_export.py              # Stage 2: Export to TRC/MOT/FBX
├── run_pipeline.py            # Combined TRC pipeline
├── run_full_pipeline.py       # Full pipeline (TRC + IK + FBX)
├── test_imports.py            # Verify installation
└── requirements.txt           # Python dependencies
```

## License

This project is provided for research and educational purposes.

## Acknowledgments

- **SAM3D Body**: Meta/Facebook Research - https://github.com/facebookresearch/sam-3d-body
- **SAM3**: Meta/Facebook Research - https://github.com/facebookresearch/sam3
- **MoGe2**: Monocular Geometry Estimation
- **Pose2Sim**: https://github.com/perfanalytics/pose2sim
- **OpenSim**: https://opensim.stanford.edu/
