<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-10 09:53:31 KST -->

# SAM3D-OpenSim

## Purpose
SAM3D-OpenSim converts monocular video into OpenSim-ready motion data by running SAM3D Body inference, transforming MHR70 keypoints into biomechanical marker trajectories, solving inverse kinematics, and exporting Blender-ready motion. The current pipeline also supports single-person selection and tracking, MoGe-based support-surface grounding, hybrid vertical translation, and post-IK stance-phase foot snapping.

## Key Files
| File | Description |
|------|-------------|
| `run_full_pipeline.py` | Windows-oriented orchestrator for inference, OpenSim IK, post-IK foot snap, and Blender export. |
| `run_inference.py` | Stage 1 CLI that extracts frames and saves SAM3D Body outputs plus selection and scene-ground metadata to `video_outputs.json`. |
| `run_export.py` | Stage 2 CLI that converts saved SAM3D results into TRC, MOT, and FBX outputs with contact-aware translation options. |
| `run_pipeline.py` | Combined pipeline entry point used by the full runner for TRC-centric processing and optional direct IK/export. |
| `test_imports.py` | Environment smoke test for SAM3, MoGe2, and SAM3D Body imports. |
| `server.py` | Simple HTTP server for the local animation viewer. |
| `index.html` | Two-panel viewer for syncing a source video with exported GLB/FBX animation. |
| `Import_OS4_Patreon_Aitor_Skely.blend` | Blender rig template used by the Skely FBX export scripts. |
| `README.md` | Primary user-facing overview, setup summary, and CLI examples. |
| `requirements.txt` | Python package requirements for the repository-side tooling. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `config/` | Pipeline configuration and MHR70-to-OpenSim marker mapping data (see `config/AGENTS.md`). |
| `docs/` | Long-form documentation and technical reference material (see `docs/AGENTS.md`). |
| `models/` | OpenSim marker-set assets checked into the repository (see `models/AGENTS.md`). |
| `scripts/` | Blender-side export scripts for TRC- and MOT-driven animation export (see `scripts/AGENTS.md`). |
| `src/` | Core inference, transformation, conversion, TRC export, and IK helper modules (see `src/AGENTS.md`). |
| `utils/` | Shared file, config, and video-processing utilities used by the CLIs (see `utils/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Keep Windows-specific external tool paths aligned with the actual deployment environment; several entry points assume SAM3D Body, Pose2Sim, and Blender live outside the repo.
- Prefer the two-stage workflow when iterating on export logic: reuse an existing `video_outputs.json` instead of re-running expensive inference.
- Treat coordinate-system and contact changes as cross-cutting: updates usually require matching edits in `src/coordinate_transform.py`, `src/keypoint_converter.py`, `config/marker_mapping.yaml`, runtime OpenSim marker/task helpers, and Blender export scripts.
- Keep viewer tooling (`index.html`, `server.py`) separate from the motion pipeline; it consumes outputs but should not redefine motion semantics.

### Testing Requirements
- Run `python test_imports.py` in the `sam_3d_body` environment after changing inference integration or external dependency paths.
- Run `python run_inference.py --help`, `python run_export.py --help`, and `python run_full_pipeline.py --help` after CLI argument changes.
- For export and post-processing changes, smoke-test `python run_export.py --input <existing-output>/video_outputs.json --height <meters> --skip-ik --skip-fbx`.
- For OpenSim IK or Blender export changes, validate on Windows with the `Pose2Sim` environment and Blender installed; the local Linux sandbox is not a full substitute.

### Common Patterns
- Entry points prepend `PROJECT_ROOT` to `sys.path` and delegate most logic to `src/` and `utils/`.
- Intermediate data is stored in timestamped output directories named `output_YYYYMMDD_HHMMSS_<video>`.
- The SAM3D interchange format is a per-frame JSON list whose `outputs` entries contain `pred_keypoints_3d`, `pred_cam_t`, `focal_length`, optional 2D/shape data, and clip-level scene-ground metadata when MoGe support-surface features are enabled.

## Dependencies

### Internal
- `config/` provides YAML-backed defaults and marker mapping definitions.
- `src/` contains the numerical pipeline and integration wrappers.
- `utils/` handles config loading, JSON serialization, and frame extraction.
- `scripts/` plus `Import_OS4_Patreon_Aitor_Skely.blend` implement Blender export workflows.
- `docs/` mirrors the repository knowledge that should stay consistent with code changes.

### External
- SAM3D Body for 3D pose estimation and component orchestration.
- SAM3 for the `sam3` detector option.
- MoGe2 for monocular focal-length estimation.
- Pose2Sim/OpenSim for inverse kinematics.
- Blender 4.5+ for FBX and GLB export.
- OpenCV, NumPy, SciPy, PyYAML, and tqdm for repository-side processing.

## Environments
| Environment | Purpose |
|-------------|---------|
| `sam_3d_body` | SAM3D Body inference, detector loading, and dependency verification. |
| `Pose2Sim` | OpenSim inverse kinematics, IK task generation, and post-IK MOT correction. |

## External Dependencies
| Component | Location |
|-----------|----------|
| SAM3D Body | `C:\Sam3dBody\sam-3d-body` |
| SAM3 | `C:\Sam3dBody\sam3` repository root containing the nested `sam3` package |
| MoGe2 | Installed in the `sam_3d_body` environment |
| Blender | Auto-detected from `C:\Program Files\Blender Foundation\Blender *\blender.exe` or explicit overrides |
| Pose2Sim | Installed in the `Pose2Sim` environment |

## Key Concepts

### MHR70 Skeleton
70-keypoint skeleton format from SAM3D Body:
- Body: indices 0-14 (COCO17-like)
- Feet: indices 15-20
- Right hand: indices 21-41
- Left hand: indices 42-62
- Extra anatomical: indices 63-69

### cam_t
Camera translation vector from SAM3D Body. Used for global translation tracking when `--global-translation` is enabled.

### Global Translation
Tracks subject movement through 3D space using `cam_t`. Accurate results depend on `--fov moge2`.

### Support Surface
Support surface is the MoGe-derived contact plane used as the scene-ground reference. It can be auto-estimated or selected from a manual ROI before inference continues.

### Ground Alignment
Ground alignment controls how stance and flight frames are anchored to the support surface. `contact_aware` is the main mode for jumps and landing-sensitive clips.

### Vertical Translation
Vertical translation determines whether the export uses legacy X/Z-only global translation or the hybrid support-plane mode that blends `pred_cam_t_y` with support-plane clearance.

### Hand Markers
`LIndex3`, `RIndex3`, `LMiddleTip`, and `RMiddleTip` improve inverse kinematics for forearm pronation and supination.

### Post-IK Foot Snap
Post-IK foot snap is a MOT-level correction that adjusts `pelvis_ty` during stance frames so the IK result lands closer to the support surface without re-running marker fitting.

## Common Tasks

### Run Best-Quality Pipeline
```bash
conda activate sam_3d_body
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation
```

### Run With Manual Support Surface
```bash
python run_full_pipeline.py --input video.mp4 --height 1.69 \
    --detector sam3 --fov moge2 --global-translation \
    --support-surface-mode manual_roi --vertical-translation-mode hybrid_support_plane
```

### Re-export With Contact-Aware Translation
```bash
python run_export.py --input output_dir/video_outputs.json --height 1.69 \
    --global-translation --ground-alignment-mode contact_aware \
    --vertical-translation-mode hybrid_support_plane --post-ik-foot-snap stance_only
```

### Adjust Smoothing
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

### SAM3 Import Error
The repository root should resolve to `C:/Sam3dBody/sam3`; the upstream package still contains a nested `sam3` package inside that repo.

### Missing SAM3 Dependencies
```bash
pip install decord psutil ftfy regex triton-windows
```

### OpenSim Errors
Run IK-related commands in the `Pose2Sim` environment, not `sam_3d_body`.

## Recent Changes (2026-03-10)
1. Added single-person selection and tracking as the default inference mode.
2. Added manual support-surface ROI selection and MoGe scene-ground metadata.
3. Added contact-aware ground alignment with hybrid support-plane vertical translation.
4. Added centralized OpenSim runtime marker and IK task generation, including heel, big-toe, and small-toe markers.
5. Added post-IK stance-phase foot snap that preserves raw MOT alongside corrected MOT output.
6. Added Windows path auto-discovery helpers for Conda environments, Pose2Sim assets, and Blender executables.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
