<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-08 12:51:08 KST -->

# docs

## Purpose
This directory holds long-form repository documentation. It captures the deeper architecture, usage modes, configuration behavior, and troubleshooting guidance that supplement the shorter root-level guides.

## Key Files
| File | Description |
|------|-------------|
| `FULL_DOCUMENTATION.md` | Comprehensive technical documentation covering architecture, usage, marker mapping, coordinate systems, and troubleshooting. |

## Subdirectories
This directory has no versioned subdirectories.

## For AI Agents

### Working In This Directory
- Keep documentation aligned with the actual CLI flags and defaults exposed in `run_full_pipeline.py`, `run_inference.py`, and `run_export.py`.
- Preserve the Windows-specific setup assumptions unless code changes intentionally remove them.
- Update diagrams and output-file counts when marker sets, exports, or pipeline stages change.

### Testing Requirements
- Cross-check examples against the current help text of the corresponding CLI entry points.
- Ensure any new external dependency mentioned here is also reflected in `README.md`, `SETUP.md`, or root `AGENTS.md` if it affects setup.

### Common Patterns
- Root-level markdown files handle quick-start and setup material.
- `docs/FULL_DOCUMENTATION.md` is the authoritative long-form reference for design decisions and troubleshooting detail.

## Dependencies

### Internal
- Root documentation files: `README.md`, `SETUP.md`, `HOW_TO_RUN.md`, and `DOCUMENTATION.md`.
- CLI and module behavior in `run_*.py` and `src/`.

### External
- None at runtime; content describes SAM3D Body, MoGe2, Pose2Sim/OpenSim, and Blender integrations.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
