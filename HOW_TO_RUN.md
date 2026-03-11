# How to Run SAM3D-OpenSim

Practical command guide for the current canonical pipeline.

## Quick Start

```bash
conda activate sam_3d_body
cd C:\Sam3DBodyToOpenSim

python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

## Recommended Workflows

### Full pipeline

Best when processing a clip for the first time:

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

### Two-stage workflow

Best when iterating on export settings without rerunning inference:

Stage 1:

```bash
python run_inference.py --input video.mp4 --detector sam3 --fov moge2
```

Stage 2:

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --global-translation \
    --ground-alignment-mode contact_aware \
    --vertical-translation-mode hybrid_support_plane \
    --save_graph
```

### Manual support surface

Use this when the real contact plane is not the room floor:

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation \
    --support-surface-mode manual_roi \
    --vertical-translation-mode hybrid_support_plane \
    --post-ik-foot-snap stance_only
```

### CPU inference

`run_full_pipeline.py` does not expose `--device`, so use the two-stage path:

```bash
python run_inference.py --input video.mp4 --device cpu
python run_export.py --input output_dir/video_outputs.json --height 1.69
```

## Common Examples

### Fast preview

```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector vitdet --skip-fbx
```

### TRC only

```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --skip-ik --skip-fbx --save_graph
```

### Smoothing

```bash
# More smoothing
python run_export.py --input output_dir/video_outputs.json --height 1.69 --smooth 4.0

# Less smoothing
python run_export.py --input output_dir/video_outputs.json --height 1.69 --smooth 8.0

# No smoothing
python run_export.py --input output_dir/video_outputs.json --height 1.69 --smooth 0
```

## Command Reference

### `run_pipeline.py`

One-process convenience wrapper over the same shared Stage 1 + Stage 2 helpers.

It may additionally write:

- `keypoints_raw.json`
- `processing_report.json`

`--visualize` is currently a deprecated compatibility flag and does not generate extra outputs.

### `run_full_pipeline.py`

Canonical two-stage orchestrator:

```bash
python run_full_pipeline.py \
    --input VIDEO_PATH \
    --height HEIGHT \
    --mass MASS \
    --output OUTPUT_DIR \
    --fps FPS \
    --detector DETECTOR \
    --segmentor SEGMENTOR \
    --fov FOV \
    --use-mask \
    --single_person true \
    --support-surface-mode auto \
    --smooth CUTOFF \
    --ground-alignment-mode auto \
    --vertical-translation-mode auto \
    --post-ik-foot-snap off \
    --global-translation \
    --skip-inference \
    --skip-ik \
    --skip-fbx
```

### `run_inference.py`

Stage 1 only:

```bash
python run_inference.py \
    --input VIDEO_PATH \
    --output OUTPUT_DIR \
    --fps FPS \
    --device cuda \
    --detector DETECTOR \
    --segmentor SEGMENTOR \
    --fov FOV \
    --use-mask \
    --single_person true \
    --support-surface-mode auto
```

### `run_export.py`

Stage 2 only:

```bash
python run_export.py \
    --input VIDEO_OUTPUTS_JSON \
    --height HEIGHT \
    --mass MASS \
    --fps FPS \
    --global-translation \
    --smooth CUTOFF \
    --ground-alignment-mode auto \
    --vertical-translation-mode auto \
    --post-ik-foot-snap off \
    --save_graph \
    --skip-ik \
    --skip-fbx
```

## Output Files

Typical output directory:

```text
output_YYYYMMDD_HHMMSS_videoname/
├── frames/                           # Extracted JPG frames
├── video_outputs.json                # SAM3D keypoints + cam_t
├── inference_meta.json               # Video metadata
├── post_ik_contact_meta.json         # Contact/flight metadata used before IK
├── markers_videoname.trc             # OpenSim markers
├── markers_videoname_ik.mot          # Joint angles
├── markers_videoname_ik_raw.mot      # Raw IK output when post-IK correction is enabled
├── markers_videoname_foot_snap_report.json
├── markers_videoname_model.osim      # OpenSim model with runtime markers
├── graphs/coords/*.png               # TRC coordinate plots when --save_graph is enabled
├── graphs/angles/*.png               # MOT angle plots when --save_graph is enabled and IK runs
└── markers_videoname.fbx             # Blender animation
```

## Troubleshooting

### Input not found

Check the input path and confirm the file exists.

### No module named `torch`

Activate the inference environment:

```bash
conda activate sam_3d_body
```

### CUDA out of memory

- lower `--fps`
- use a smaller source video
- use `run_inference.py --device cpu`

### IK errors

- verify the TRC was written successfully
- confirm you are using the `Pose2Sim` environment for IK
- try `--skip-ik` first to isolate export-stage issues

### Blender export not found

The runtime auto-detects Blender under `C:\Program Files\Blender Foundation\Blender *\blender.exe`.

If your install lives elsewhere, set `BLENDER_PATH` or `SAM3D_OPENSIM_BLENDER_PATH`.

### `--visualize` has no visible effect

`run_pipeline.py --visualize` is currently a deprecated no-op.

### `graphs/angles/` is missing

If you used `--skip-ik`, only `graphs/coords/` is expected.

Angle plots are generated only when a MOT file exists.

It is kept only for compatibility and does not generate extra viewer or media outputs.

## Practical Tips

- Use `--fov moge2` when enabling `--global-translation`.
- Prefer the two-stage workflow when tuning export behavior.
- Use `--support-surface-mode manual_roi` for treadmill, box, or elevated contact scenes.
- Use `--ground-alignment-mode contact_aware` for jumps and flight phases.
