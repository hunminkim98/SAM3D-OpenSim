<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-08 12:51:08 KST -->

# models

## Purpose
This directory contains repository-managed OpenSim asset definitions. In the current tree it primarily versions the marker-set XML that must stay consistent with the code and YAML mapping used during TRC export and inverse kinematics.

## Key Files
| File | Description |
|------|-------------|
| `Markers_MHR70.xml` | OpenSim marker-set definition for the MHR70-derived markers used by the pipeline. |

## Subdirectories
This directory has no versioned subdirectories.

## For AI Agents

### Working In This Directory
- Keep marker names, body assignments, and counts synchronized with `config/marker_mapping.yaml`, `src/keypoint_converter.py`, and any IK setup XML emitted at runtime.
- Be explicit when the repository relies on external OpenSim models from Pose2Sim instead of versioning them here.
- Avoid changing marker semantics casually; downstream IK stability depends on these names and attachments.

### Testing Requirements
- After marker-set changes, export a TRC and verify the marker names match the XML and YAML mapping.
- Re-run an IK smoke test in the `Pose2Sim` environment if body assignments or required markers change.

### Common Patterns
- Marker XML uses OpenSim body names such as `head`, `torso`, `humerus_l`, and `toes_r`.
- The repository currently versions marker definitions, while some `.osim` models are expected from external Pose2Sim installs.

## Dependencies

### Internal
- `config/marker_mapping.yaml` for naming and segment consistency.
- `src/keypoint_converter.py` and `run_export.py` for emitted marker trajectories.
- `src/opensim_ik.py` and runtime-generated IK setup files.

### External
- OpenSim/Pose2Sim tooling that consumes the marker-set XML.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
