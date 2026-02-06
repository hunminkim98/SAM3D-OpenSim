# How to Run SAM3D-OpenSim

Step-by-step guide for running the video-to-OpenSim pipeline.

## Quick Start

```bash
conda activate sam_3d_body
cd C:\Sam3DBodyToOpenSim

python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

## Workflow Options

### Option 1: Full Pipeline (One Command)

Best for first-time processing:

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

**Output:** TRC, MOT, and FBX files in `output_YYYYMMDD_HHMMSS_videoname/`

### Option 2: Two-Stage Workflow (Recommended)

Best for iterating on export settings without re-running inference:

**Stage 1: Inference (slow, run once)**
```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2
```

**Stage 2: Export (fast, iterate on settings)**
```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --global-translation --smooth 6.0
```

## Configuration Examples

### Best Quality (Production)

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 \
    --fov moge2 \
    --global-translation \
    --smooth 6.0
```

### Fast Preview

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector vitdet \
    --skip-fbx
```

### Walking/Running Analysis

```bash
python run_full_pipeline.py --input walking.mp4 --height 1.69 \
    --detector sam3 \
    --fov moge2 \
    --global-translation
```

### Stationary Subject

```bash
python run_full_pipeline.py --input standing.mp4 --height 1.69 \
    --detector sam3 \
    --fov moge2
```

### Skip OpenSim IK (TRC only)

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --skip-ik --skip-fbx
```

### Adjust Smoothing

```bash
# More smoothing (slower movements)
python run_export.py --input video_outputs.json --height 1.69 --smooth 4.0

# Less smoothing (faster movements)
python run_export.py --input video_outputs.json --height 1.69 --smooth 8.0

# No smoothing
python run_export.py --input video_outputs.json --height 1.69 --smooth 0
```

## Command Reference

### run_full_pipeline.py

Full pipeline from video to OpenSim outputs.

```bash
python run_full_pipeline.py \
    --input VIDEO_PATH \           # Required: input video
    --height HEIGHT \              # Subject height in meters (default: 1.75)
    --mass MASS \                  # Subject mass in kg (default: 70.0)
    --output OUTPUT_DIR \          # Output directory (default: auto)
    --fps FPS \                    # Target FPS (default: 30)
    --detector DETECTOR \          # vitdet, yolo11, sam3, none (default: vitdet)
    --fov FOV \                    # moge2, none (default: moge2)
    --smooth CUTOFF \              # Smoothing cutoff Hz (default: 6.0, 0=disable)
    --global-translation \         # Enable global movement tracking
    --skip-inference \             # Use existing video_outputs.json
    --skip-ik \                    # Skip OpenSim IK
    --skip-fbx                     # Skip FBX export
```

### run_inference.py

Stage 1: SAM3D Body inference only.

```bash
python run_inference.py \
    --input VIDEO_PATH \
    --detector DETECTOR \
    --fov FOV \
    --output OUTPUT_DIR
```

### run_export.py

Stage 2: Export from existing JSON to TRC/MOT/FBX.

```bash
python run_export.py \
    --input VIDEO_OUTPUTS_JSON \   # Path to video_outputs.json
    --height HEIGHT \
    --global-translation \
    --smooth CUTOFF \
    --skip-ik \
    --skip-fbx
```

## Using Existing SAM3D Outputs

You can process outputs from the SAM3D Body demo:

```bash
python run_export.py \
    --input C:\Sam3dBody\sam-3d-body\outputs\my_video\video_outputs.json \
    --height 1.75 \
    --global-translation
```

## Output Files

After running, you'll find:

```
output_YYYYMMDD_HHMMSS_videoname/
├── frames/                           # Extracted PNG frames
├── video_outputs.json                # SAM3D keypoints + cam_t
├── inference_meta.json               # Video metadata
├── markers_videoname.trc             # OpenSim markers
├── markers_videoname_ik.mot          # Joint angles
├── markers_videoname_model.osim      # OpenSim model
└── markers_videoname.fbx             # Blender animation
```

## Viewing Results

### OpenSim GUI

1. Open OpenSim
2. File → Open Model → `markers_*_model.osim`
3. File → Load Motion → `markers_*_ik.mot`
4. Click Play

### Blender

1. Open Blender
2. File → Import → FBX → `markers_*.fbx`
3. Press Space to play animation

## Troubleshooting

### "Input not found"

Ensure the video path is correct and the file exists.

### "No module named 'torch'"

Activate the correct environment:
```bash
conda activate sam_3d_body
```

### CUDA out of memory

- Lower FPS: `--fps 15`
- Use smaller video resolution
- Close other GPU applications

### IK errors

- Verify TRC loads in OpenSim GUI
- Check subject height is correct
- Try `--skip-ik` to verify TRC generation

### Jittery output

- Increase smoothing: `--smooth 4.0`
- Ensure `--fov moge2` is enabled for global translation

## Tips

1. **Always use `--fov moge2`** when using `--global-translation`
2. **Use the two-stage workflow** when iterating on height or smoothing
3. **SAM3 detector** gives best results but is slower
4. **6 Hz smoothing** is a good default for walking/running
5. **Lower smoothing (8-10 Hz)** for fast movements like jumping
