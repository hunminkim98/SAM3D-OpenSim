"""Stage-1 mesh visualization and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from tqdm import tqdm

from src.sam3d_mesh_renderer import Renderer as LocalMeshRenderer


LIGHT_GREEN = (0.60784314, 0.89803922, 0.39215686)
MESH_SEQUENCE_FORMAT_CHOICES = ("ply", "obj")


class MeshVisualizationError(RuntimeError):
    """Raised when optional mesh sidecar output generation fails."""


def _resolve_frame_outputs(frame_data: Dict[str, Any], *, single_person: bool) -> List[Dict[str, Any]]:
    """Return the list of mesh-bearing outputs to render/export for one frame."""
    if single_person:
        selected = frame_data.get("output")
        return [selected] if selected is not None else []

    outputs = frame_data.get("outputs", [])
    return [output for output in outputs if output is not None]


def _sort_outputs_by_depth(outputs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort far-to-near so later overlays stay visually in front."""

    def depth_key(output: Dict[str, Any]) -> float:
        cam_t = np.asarray(output.get("pred_cam_t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        return float(cam_t[2]) if cam_t.size >= 3 else 0.0

    return sorted(outputs, key=depth_key, reverse=True)


def _require_local_renderer():
    """Return the repo-local renderer class used for mesh overlays."""
    return LocalMeshRenderer


def _build_export_mesh(
    vertices: np.ndarray,
    camera_translation: np.ndarray,
    faces: np.ndarray,
):
    """Mirror the upstream trimesh export transform without importing pyrender."""
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - dependency failure
        raise MeshVisualizationError(
            "Mesh sequence export requires trimesh to be installed."
        ) from exc

    vertices = np.asarray(vertices, dtype=np.float32)
    camera_translation = np.asarray(camera_translation, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    vertex_colors = np.array([(*LIGHT_GREEN, 1.0)] * vertices.shape[0], dtype=np.float32)
    mesh = trimesh.Trimesh(
        vertices=vertices.copy() + camera_translation.reshape(1, 3),
        faces=faces.copy(),
        vertex_colors=vertex_colors,
        process=False,
    )

    rot = trimesh.transformations.rotation_matrix(np.radians(180.0), [1, 0, 0])
    mesh.apply_transform(rot)
    return mesh


def render_mesh_overlay_frame(
    image_bgr: np.ndarray,
    outputs: Iterable[Dict[str, Any]],
    faces: np.ndarray,
) -> np.ndarray:
    """Render all available meshes onto a single BGR frame."""
    outputs_sorted = _sort_outputs_by_depth(list(outputs))
    if not outputs_sorted:
        return image_bgr.copy()

    renderer_cls = _require_local_renderer()
    rendered = image_bgr.copy()
    faces = np.asarray(faces, dtype=np.int32)

    for person_output in outputs_sorted:
        vertices = person_output.get("pred_vertices")
        cam_t = person_output.get("pred_cam_t")
        focal_length = person_output.get("focal_length")
        if vertices is None or cam_t is None or focal_length is None:
            continue

        renderer = renderer_cls(focal_length=float(focal_length), faces=faces)
        try:
            rendered = (
                renderer(
                    np.asarray(vertices, dtype=np.float32),
                    np.asarray(cam_t, dtype=np.float32),
                    rendered.copy(),
                    mesh_base_color=LIGHT_GREEN,
                    scene_bg_color=(1, 1, 1),
                )
                * 255.0
            ).astype(np.uint8)
        except Exception as exc:
            raise MeshVisualizationError(
                "Mesh overlay rendering failed while rasterizing a frame. "
                "If this is an OpenGL backend issue, try setting "
                "SAM3D_OPENSIM_MESH_RENDER_PLATFORM before running inference."
            ) from exc

    return rendered


def save_mesh_overlay_video(
    *,
    frame_paths: List[str],
    frame_outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    output_path: str | Path,
    fps: float,
    single_person: bool,
    codec: str = "mp4v",
) -> str:
    """Render mesh overlays frame-by-frame and write them as a video."""
    import cv2

    if len(frame_paths) != len(frame_outputs):
        raise ValueError("frame_paths and frame_outputs must have the same length")
    if not frame_paths:
        raise ValueError("No frames available for mesh overlay video")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )

    try:
        for frame_path, frame_data in tqdm(
            zip(frame_paths, frame_outputs),
            total=len(frame_paths),
            desc="Rendering mesh overlay video",
            unit="frames",
        ):
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                raise ValueError(f"Cannot read frame: {frame_path}")

            outputs = _resolve_frame_outputs(frame_data, single_person=single_person)
            if outputs:
                image_bgr = render_mesh_overlay_frame(image_bgr, outputs, faces)

            writer.write(image_bgr)
    finally:
        writer.release()

    return str(output_path)


def save_mesh_sequence(
    *,
    frame_outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    output_dir: str | Path,
    single_person: bool,
    export_format: str,
) -> Dict[str, Any]:
    """Export one mesh file per detected/rendered person per frame."""
    export_format = str(export_format).strip().lower()
    if export_format not in MESH_SEQUENCE_FORMAT_CHOICES:
        raise ValueError(
            f"Unsupported mesh export format: {export_format!r}. "
            f"Expected one of {MESH_SEQUENCE_FORMAT_CHOICES}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: list[str] = []
    faces = np.asarray(faces, dtype=np.int32)

    for frame_idx, frame_data in tqdm(
        enumerate(frame_outputs),
        total=len(frame_outputs),
        desc=f"Exporting mesh sequence ({export_format})",
        unit="frames",
    ):
        outputs = _resolve_frame_outputs(frame_data, single_person=single_person)
        for person_idx, person_output in enumerate(outputs):
            vertices = person_output.get("pred_vertices")
            cam_t = person_output.get("pred_cam_t")
            if vertices is None or cam_t is None:
                continue

            mesh = _build_export_mesh(vertices, cam_t, faces)
            mesh_path = output_dir / f"frame_{frame_idx:06d}_person_{person_idx:03d}.{export_format}"
            mesh.export(mesh_path)
            exported_files.append(str(mesh_path))

    return {
        "directory": str(output_dir),
        "format": export_format,
        "count": len(exported_files),
        "files": exported_files,
    }


def save_mesh_sidecars(
    *,
    output_dir: str | Path,
    frame_paths: List[str],
    frame_outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    fps: float,
    single_person: bool,
    save_mesh_video: bool,
    save_mesh_sequence_files: bool,
    mesh_sequence_format: str,
) -> Dict[str, Any]:
    """Generate requested Stage-1 mesh sidecar outputs."""
    if not save_mesh_video and not save_mesh_sequence_files:
        return {}
    if faces is None:
        raise MeshVisualizationError("SAM3D Body did not expose mesh faces for visualization.")

    output_dir = Path(output_dir)
    saved: Dict[str, Any] = {}

    if save_mesh_video:
        mesh_vis_dir = output_dir / "mesh_vis"
        video_path = mesh_vis_dir / "overlay.mp4"
        try:
            saved["mesh_video"] = save_mesh_overlay_video(
                frame_paths=frame_paths,
                frame_outputs=frame_outputs,
                faces=faces,
                output_path=video_path,
                fps=fps,
                single_person=single_person,
            )
        except Exception as exc:
            saved["mesh_video_error"] = str(exc)

    if save_mesh_sequence_files:
        mesh_export_dir = output_dir / "mesh_export"
        try:
            sequence_info = save_mesh_sequence(
                frame_outputs=frame_outputs,
                faces=faces,
                output_dir=mesh_export_dir,
                single_person=single_person,
                export_format=mesh_sequence_format,
            )
            saved["mesh_sequence_dir"] = sequence_info["directory"]
            saved["mesh_sequence_format"] = sequence_info["format"]
            saved["mesh_sequence_count"] = sequence_info["count"]
        except Exception as exc:
            saved["mesh_sequence_error"] = str(exc)

    return saved
