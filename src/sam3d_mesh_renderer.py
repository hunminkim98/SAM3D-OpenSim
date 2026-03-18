"""Local SAM3D mesh renderer adapted from the upstream official implementation.

This module intentionally vendors the minimum renderer logic we need for Stage 1
mesh overlay video generation. The upstream renderer sets
``PYOPENGL_PLATFORM="egl"`` at import time, which is brittle on Windows. Here
we keep the rendering math and camera conventions while leaving platform
selection under this repository's control.
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np


def _apply_platform_override_from_env() -> None:
    """Optionally override the PyOpenGL platform before importing pyrender."""
    platform_override = os.environ.get("SAM3D_OPENSIM_MESH_RENDER_PLATFORM")
    if platform_override:
        os.environ["PYOPENGL_PLATFORM"] = platform_override


def _require_render_dependencies():
    """Import optional renderer dependencies lazily."""
    _apply_platform_override_from_env()
    try:
        import pyrender
        import trimesh
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "Local SAM3D mesh rendering requires pyrender and trimesh. "
            "If you need to force a backend, set "
            "SAM3D_OPENSIM_MESH_RENDER_PLATFORM before running inference."
        ) from exc
    return pyrender, trimesh


def create_raymond_lights(pyrender_module) -> List[object]:
    """Return Raymond light nodes matching the upstream renderer behavior."""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float32)
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0], dtype=np.float32)
    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp], dtype=np.float32)
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0], dtype=np.float32)
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender_module.Node(
                light=pyrender_module.DirectionalLight(
                    color=np.ones(3, dtype=np.float32),
                    intensity=1.0,
                ),
                matrix=matrix,
            )
        )

    return nodes


class Renderer:
    """Repo-local wrapper around pyrender for SAM3D Body mesh overlays."""

    def __init__(self, focal_length: float, faces=None):
        self.focal_length = float(focal_length)
        self.faces = np.asarray(faces, dtype=np.int32) if faces is not None else None
        self._pyrender, self._trimesh = _require_render_dependencies()

    def __call__(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        image: np.ndarray,
        full_frame: bool = False,
        imgname: Optional[str] = None,
        side_view: bool = False,
        top_view: bool = False,
        rot_angle: float = 90,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        tri_color_lights: bool = False,
        return_rgba: bool = False,
        camera_center=None,
    ) -> np.ndarray:
        """Render a mesh over an input image using the upstream conventions."""
        if self.faces is None:
            raise ValueError("Mesh faces are required for rendering.")

        if full_frame:
            if imgname is None:
                raise ValueError("imgname is required when full_frame=True")
            image = cv2.imread(imgname).astype(np.float32)

        image = image.astype(np.float32) / 255.0
        h, w = image.shape[:2]

        renderer = self._pyrender.OffscreenRenderer(
            viewport_height=h,
            viewport_width=w,
        )

        camera_translation = np.asarray(cam_t, dtype=np.float32).copy()
        camera_translation[0] *= -1.0

        material = self._pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(1.0, 1.0, 1.0, 1.0),
        )

        mesh = self._trimesh.Trimesh(
            np.asarray(vertices, dtype=np.float32).copy(),
            self.faces.copy(),
            process=False,
        )

        if side_view:
            rot = self._trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0]
            )
            mesh.apply_transform(rot)
        elif top_view:
            rot = self._trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [1, 0, 0]
            )
            mesh.apply_transform(rot)

        rot = self._trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        mesh = self._pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = self._pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0],
            ambient_light=(0.3, 0.3, 0.3),
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, 3] = camera_translation
        if camera_center is None:
            camera_center = [image.shape[1] / 2.0, image.shape[0] / 2.0]

        camera = self._pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=float(camera_center[0]),
            cy=float(camera_center[1]),
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights(self._pyrender)
        if tri_color_lights:
            colors = [
                np.array([1, 0.2, 0.3], dtype=np.float32),
                np.array([0.2, 1, 0.2], dtype=np.float32),
                np.array([0.2, 0.2, 1], dtype=np.float32),
            ]
            for light_node, color in zip(light_nodes, colors):
                light_node.light.color = color
                light_node.light.intensity = 2.0

        for node in light_nodes:
            scene.add_node(node)

        color, _render_depth = renderer.render(
            scene,
            flags=self._pyrender.RenderFlags.RGBA,
        )
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        mesh_color_bgr = np.array(
            [mesh_base_color[2], mesh_base_color[1], mesh_base_color[0]],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        shading = np.clip(color[:, :, :3].mean(axis=2, keepdims=True), 0.0, 1.0)
        color_scale = max(float(mesh_color_bgr.mean()), 1e-6)
        tinted_bgr = np.clip(shading * (mesh_color_bgr / color_scale), 0.0, 1.0)

        if return_rgba:
            tinted_rgba = np.concatenate([tinted_bgr[:, :, ::-1], color[:, :, -1:]], axis=2)
            return tinted_rgba.astype(np.float32)

        valid_mask = color[:, :, -1][:, :, np.newaxis]
        output_img = tinted_bgr * valid_mask + (1.0 - valid_mask) * image
        return output_img.astype(np.float32)
