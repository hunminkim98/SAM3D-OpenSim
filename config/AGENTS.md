<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-10 09:53:31 KST -->

# config

## Purpose
This directory contains YAML configuration that defines repository-local defaults for external tool paths, inference behavior, single-person selection, support-surface selection, ground and vertical translation modes, post-IK correction, and the mapping from SAM3D Body's MHR70 skeleton to OpenSim marker names, derived markers, runtime IK subset order, weights, and body segments.

## Key Files
| File | Description |
|------|-------------|
| `config.yaml` | Primary pipeline configuration with SAM3D, processing, support-surface, OpenSim, and output defaults. |
| `marker_mapping.yaml` | Declarative MHR70 keypoint names, OpenSim marker mappings, derived markers, body-segment assignments, runtime IK subset order, and runtime marker weights. |

## Subdirectories
This directory has no versioned subdirectories.

## For AI Agents

### Working In This Directory
- Keep YAML keys synchronized with the fields consumed by `utils/io_utils.py`, `run_pipeline.py`, and `run_inference.py`.
- Treat marker names and weights as API surface: if a marker is renamed here, update `src/keypoint_converter.py`, `models/Markers_MHR70.xml`, and any IK task generation that references it.
- Preserve Windows path examples unless the repository is explicitly being made cross-platform.

### Testing Requirements
- After changing config schema, verify `load_config()` and `load_marker_mapping()` callers still read the expected keys.
- Re-run `python run_export.py --help` or a small export smoke test if defaults or required settings change.
- Re-run `python test_imports.py` when altering external dependency paths under `sam3d`.

### Common Patterns
- `config.yaml` stores operational defaults and absolute external paths.
- `config.yaml` is the default source for `single_person`, `ground_alignment_mode`, `vertical_translation_mode`, `support_surface_mode`, and `post_ik_foot_snap_mode`.
- `marker_mapping.yaml` is the default source for converter marker naming, derived markers, runtime IK subset order, and runtime IK weight overrides.

## Dependencies

### Internal
- `utils/io_utils.py` loads both YAML files.
- `src/keypoint_converter.py` can use mapping data and mirrors the default marker naming.
- `src/opensim_marker_spec.py`, `src/opensim_ik.py`, and `run_export.py` depend on marker naming and weight consistency.

### External
- PyYAML for parsing configuration.
- OpenSim and Pose2Sim naming conventions for marker and body-segment assignments.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
