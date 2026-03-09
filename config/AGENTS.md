<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-08 12:51:08 KST -->

# config

## Purpose
This directory contains YAML configuration that defines repository-local defaults for external tool paths, inference behavior, smoothing and normalization settings, and the mapping from SAM3D Body's MHR70 skeleton to OpenSim marker names and body segments.

## Key Files
| File | Description |
|------|-------------|
| `config.yaml` | Primary pipeline configuration with SAM3D, processing, OpenSim, and output defaults. |
| `marker_mapping.yaml` | Declarative MHR70 keypoint names, OpenSim marker mappings, derived markers, and body-segment assignments. |

## Subdirectories
This directory has no versioned subdirectories.

## For AI Agents

### Working In This Directory
- Keep YAML keys synchronized with the fields consumed by `utils/io_utils.py`, `run_pipeline.py`, and `run_inference.py`.
- Treat marker names as API surface: if a marker is renamed here, update `src/keypoint_converter.py`, `models/Markers_MHR70.xml`, and any IK task generation that references it.
- Preserve Windows path examples unless the repository is explicitly being made cross-platform.

### Testing Requirements
- After changing config schema, verify `load_config()` and `load_marker_mapping()` callers still read the expected keys.
- Re-run `python run_export.py --help` or a small export smoke test if defaults or required settings change.
- Re-run `python test_imports.py` when altering external dependency paths under `sam3d`.

### Common Patterns
- `config.yaml` stores operational defaults and absolute external paths.
- `marker_mapping.yaml` mirrors code-level defaults so changes should normally be reflected in both places.

## Dependencies

### Internal
- `utils/io_utils.py` loads both YAML files.
- `src/keypoint_converter.py` can use mapping data and mirrors the default marker naming.
- `src/opensim_ik.py` and `run_export.py` depend on marker naming consistency.

### External
- PyYAML for parsing configuration.
- OpenSim and Pose2Sim naming conventions for marker and body-segment assignments.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
