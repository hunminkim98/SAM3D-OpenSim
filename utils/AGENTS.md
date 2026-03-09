<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-08 12:51:08 KST | Updated: 2026-03-08 12:51:08 KST -->

# utils

## Purpose
This directory contains shared helper functions for loading configuration, serializing pipeline data, generating output directories, and extracting or reassembling video frames. The CLIs use these utilities to keep file and media handling out of the core pipeline modules.

## Key Files
| File | Description |
|------|-------------|
| `io_utils.py` | YAML, JSON, and pickle helpers plus output-directory naming and config merging utilities. |
| `video_utils.py` | Video metadata inspection, frame extraction, frame streaming, and frame-to-video assembly helpers. |
| `__init__.py` | Package marker for utility imports. |

## Subdirectories
This directory has no versioned subdirectories.

## For AI Agents

### Working In This Directory
- Keep these helpers dependency-light; they are imported by multiple entry points and should remain safe to import before heavyweight model setup.
- Preserve output naming and serialization formats unless the corresponding callers are updated together.
- Avoid introducing repository-global side effects here; these functions are expected to be reusable and predictable.

### Testing Requirements
- Run a syntax check such as `python -m py_compile utils/*.py`.
- Re-run the CLI help commands and a small export or inference smoke test if directory creation, JSON structure, or frame extraction behavior changes.
- Verify new config helpers still accept the repository's existing YAML shape.

### Common Patterns
- Paths are normalized with `pathlib.Path`.
- Serialization helpers include NumPy-aware conversion so pipeline arrays can be written directly to JSON.
- Frame extraction writes deterministic sequential filenames such as `frame_000000.jpg`.

## Dependencies

### Internal
- Called by `run_inference.py`, `run_pipeline.py`, and other root entry points.
- Reads files from `config/`.

### External
- OpenCV for video I/O in `video_utils.py`.
- PyYAML and NumPy in `io_utils.py`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
