# SAM3D-OpenSim Technical Documentation

Technical reference for the SAM3D Body to OpenSim pipeline.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SAM3D-OpenSim Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │  Video  │───▶│ SAM3 Detector│───▶│ SAM3D Body  │───▶│ MHR70 3D       │  │
│  │  Input  │    │ (person bbox)│    │ (3D pose)   │    │ Keypoints+cam_t│  │
│  └─────────┘    └──────────────┘    └─────────────┘    └───────┬────────┘  │
│                        │                                        │           │
│                        ▼                                        │           │
│                 ┌──────────────┐                               │           │
│                 │ MoGe2 FOV    │ (focal length)                │           │
│                 └──────────────┘                               │           │
│                                                                 │           │
│                     ┌───────────────────────────────────────────┘           │
│                     ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Post-Processing Pipeline                         │  │
│  │  1. Butterworth smoothing (6 Hz default)                             │  │
│  │  2. Bone length normalization                                         │  │
│  │  3. Coordinate transformation (Camera → OpenSim)                     │  │
│  │  4. Global translation from cam_t                                    │  │
│  │  5. Per-frame ground alignment                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                     │                                                       │
│         ┌───────────┼───────────┬───────────────┐                          │
│         ▼           ▼           ▼               ▼                          │
│  ┌────────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐                    │
│  │ TRC Export │ │OpenSim │ │ Blender  │ │ Processing │                    │
│  │ (22 marks) │ │ IK     │ │ FBX      │ │ Report     │                    │
│  └─────┬──────┘ └───┬────┘ └────┬─────┘ └────────────┘                    │
│        ▼            ▼           ▼                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                   │
│  │ .trc     │ │ .mot     │ │ .fbx     │                                   │
│  │ markers  │ │ 40 DOF   │ │ skeleton │                                   │
│  └──────────┘ └──────────┘ └──────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Reference

### sam3d_inference.py

SAM3D Body model wrapper with component loading.

```python
from src.sam3d_inference import SAM3DInference

sam3d = SAM3DInference(
    sam3d_root="C:/Sam3dBody/sam-3d-body",
    checkpoint_path="...",
    device="cuda",
    detector_name="sam3",      # vitdet, yolo11, sam3, or None
    fov_name="moge2",          # moge2 or None
)

results = sam3d.process_video(frame_paths)
keypoints_3d, valid_frames = sam3d.extract_keypoints_3d(results["frames"])
```

### coordinate_transform.py

Coordinate system transformation with global translation.

```python
from src.coordinate_transform import CoordinateTransformer

transformer = CoordinateTransformer(subject_height=1.69, units="mm")

keypoints_opensim = transformer.transform(
    keypoints_3d,
    camera_translation=cam_translations,
    apply_global_translation=True,
    align_to_ground=True,
)
```

### post_processing.py

Temporal smoothing and bone normalization.

```python
from src.post_processing import PostProcessor

processor = PostProcessor(
    smooth_filter=True,
    filter_cutoff=6.0,      # Hz
    normalize_bones=True,
)

keypoints_smooth = processor.process(keypoints_3d, fps=30.0, subject_height=1.69)
```

## Coordinate Systems

### SAM3D Body (Camera Space)

```
        Y (down)
        |
        |
        +-----> X (right)
       /
      /
     Z (forward/depth)
```

### OpenSim (World Space)

```
        Y (up)
        |
        |
        +-----> Z (right)
       /
      /
     X (forward)
```

### Transformation Matrix

```python
CAMERA_TO_OPENSIM = np.array([
    [0,  0,  1],  # X_opensim = Z_camera (forward)
    [0, -1,  0],  # Y_opensim = -Y_camera (up)
    [1,  0,  0],  # Z_opensim = X_camera (right)
])
```

## MHR70 Marker Mapping

### IK Markers (22 total)

| Marker | MHR70 Index | Body Segment | IK Weight |
|--------|-------------|--------------|-----------|
| Nose | 0 | head | 0.8 |
| LEye | 1 | head | 0.4 |
| REye | 2 | head | 0.4 |
| LEar | 3 | head | 0.6 |
| REar | 4 | head | 0.6 |
| Neck | 69 | torso | 1.0 |
| LShoulder | 5 | torso | 1.0 |
| RShoulder | 6 | torso | 1.0 |
| LElbow | 7 | humerus_l | 1.0 |
| RElbow | 8 | humerus_r | 1.0 |
| LWrist | 62 | radius_l | 1.0 |
| RWrist | 41 | radius_r | 1.0 |
| LIndex3 | 49 | radius_l | 0.6 |
| RIndex3 | 28 | radius_r | 0.6 |
| LMiddleTip | 50 | radius_l | 0.5 |
| RMiddleTip | 29 | radius_r | 0.5 |
| LHip | 9 | pelvis | 1.0 |
| RHip | 10 | pelvis | 1.0 |
| LKnee | 11 | femur_l | 1.0 |
| RKnee | 12 | femur_r | 1.0 |
| LAnkle | 13 | tibia_l | 1.0 |
| RAnkle | 14 | tibia_r | 1.0 |

### Hand Markers for Arm Rotation

Four hand markers are used to improve forearm pronation/supination tracking:

- **LIndex3/RIndex3**: Index knuckle (weight 0.6)
- **LMiddleTip/RMiddleTip**: Middle fingertip (weight 0.5)

These are attached to the radius (forearm) segment in OpenSim.

## Global Translation

### How It Works

1. **MoGe2 FOV**: Estimates camera focal length from first frame
2. **cam_t Extraction**: SAM3D Body outputs `pred_cam_t = [x, y, z]` per frame
3. **Coordinate Transform**: cam_t transformed from camera to OpenSim space
4. **Smoothing**: 5-frame moving average to reduce jitter
5. **Relative Motion**: Translation relative to first frame (starts at origin)

### Implementation

```python
def _apply_global_translation(self, keypoints, camera_translation, height_scale):
    # Transform cam_t to OpenSim space
    cam_t_opensim = camera_translation @ self.CAMERA_TO_OPENSIM.T
    cam_t_opensim = cam_t_opensim * height_scale

    # Smooth cam_t
    cam_t_smoothed = self._smooth_cam_t(cam_t_opensim)

    # Apply relative translation (X=forward, Z=lateral)
    first_frame_t = cam_t_smoothed[0].copy()
    for i in range(num_frames):
        delta_t = cam_t_smoothed[i] - first_frame_t
        keypoints[i, :, 0] += delta_t[0]  # Forward
        keypoints[i, :, 2] += delta_t[2]  # Lateral
```

## Per-Frame Ground Alignment

Ensures feet always touch the ground (Y=0):

```python
def _align_to_ground(self, keypoints):
    # Foot indices
    left_heel_idx, right_heel_idx = 17, 20
    left_toe_idx, right_toe_idx = 15, 18

    for i in range(keypoints.shape[0]):
        # Find lowest foot point
        foot_heights = [
            keypoints[i, left_heel_idx, 1],
            keypoints[i, right_heel_idx, 1],
            keypoints[i, left_toe_idx, 1],
            keypoints[i, right_toe_idx, 1],
        ]
        min_y = min(foot_heights)

        # Shift frame so lowest point is at Y=0
        keypoints[i, :, 1] -= min_y
```

## Butterworth Smoothing

Low-pass filter to reduce jitter:

- **Default cutoff**: 6 Hz
- **Filter order**: 4
- **Type**: Zero-phase (filtfilt)

Preserves natural movement frequencies:
- Walking: ~2 Hz
- Running: ~3-4 Hz
- Arm swing: ~1-2 Hz

## OpenSim IK Configuration

- **Model**: Pose2Sim Simple Model (40 DOF)
- **Accuracy**: 1e-5
- **Markers**: 22 (COCO17 + Neck + Hand markers)

### Joint Angles Output

40 degrees of freedom including:
- Pelvis: 6 DOF (tx, ty, tz, rx, ry, rz)
- Spine/Torso: 3 DOF
- Neck/Head: 3 DOF
- Shoulders: 6 DOF (3 per side)
- Elbows: 2 DOF (flexion + pronation)
- Hips: 6 DOF (3 per side)
- Knees: 2 DOF
- Ankles: 4 DOF (2 per side)

## Comparison with MotionBERT-OpenSim

| Aspect | MotionBERT-OpenSim | SAM3D-OpenSim |
|--------|-------------------|---------------|
| Input | Video → 2D → 3D lifting | Video → Direct 3D |
| Skeleton | 17 joints (H36M) | 70 keypoints (MHR70) |
| Hands/Feet | No | Yes (detailed) |
| Global Position | No | Yes (cam_t) |
| FOV Estimation | No | Yes (MoGe2) |
| Mesh Output | No | Yes (6890 vertices) |

## FBX Export

The FBX export uses a skeleton template (`Import_OS4_Patreon_Aitor_Skely.blend`) with the metarig_skely armature. Joint angles from the .mot file are applied to the skeleton bones, including forearm pronation/supination from the hand markers.

### Bone Mappings

| OpenSim Joint | Blender Bone | Notes |
|---------------|--------------|-------|
| pelvis | spine | Location + rotation |
| hip | thigh.R/L | Hip flexion/adduction/rotation |
| knee | shin.R/L | Knee angle |
| ankle | foot.R/L | Ankle angle |
| lumbar | spine.001 | L5/S1 angles |
| thorax | spine.002 | Neck/thorax angles |
| shoulder | upper_arm.R/L | Arm flex/add/rotation |
| elbow | forearm.R/L | Elbow flex + pronation/supination |

### Forearm Rotation

The hand markers (LIndex3, RIndex3, LMiddleTip, RMiddleTip) constrain the `pro_sup_r/l` angles in OpenSim IK, which are then applied to the forearm bones in Blender for realistic arm rotation.

## File Formats

### video_outputs.json

SAM3D-compatible format:

```json
[
  {
    "frame": "frame_000001.png",
    "outputs": [
      {
        "bbox": [x1, y1, x2, y2],
        "focal_length": 1234.56,
        "pred_keypoints_3d": [[x, y, z], ...],
        "pred_cam_t": [tx, ty, tz]
      }
    ]
  }
]
```

### TRC Format

OpenSim marker trajectories:

```
PathFileType    4    (X/Y/Z)    markers.trc
DataRate    CameraRate    NumFrames    NumMarkers    Units    ...
30.0    30.0    1136    22    mm    ...
Frame#    Time    Nose    ...    RAnkle
1    0.0    x y z    ...    x y z
```

### MOT Format

OpenSim joint angles:

```
name markers_ik.mot
nRows=1136
nColumns=41
inDegrees=yes
endheader
time    pelvis_tx    pelvis_ty    ...    ankle_angle_r
0.0    0.0    0.0    ...    0.0
```
