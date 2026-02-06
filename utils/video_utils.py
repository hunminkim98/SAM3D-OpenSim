"""
Video processing utilities for frame extraction.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Generator
import cv2
import numpy as np
from tqdm import tqdm


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info:
            - fps: Frames per second
            - frame_count: Total number of frames
            - width: Frame width in pixels
            - height: Frame height in pixels
            - duration: Duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    image_format: str = "jpg",
    quality: int = 95,
) -> Tuple[list, float]:
    """
    Extract frames from video to image files.

    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        target_fps: Target FPS (None = use original FPS)
        start_frame: First frame to extract (0-indexed)
        end_frame: Last frame to extract (None = all frames)
        image_format: Output image format ('jpg' or 'png')
        quality: JPEG quality (1-100)

    Returns:
        Tuple of (list of frame paths, actual fps)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    # Calculate frame interval for target FPS
    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
        actual_fps = original_fps
    else:
        frame_interval = round(original_fps / target_fps)
        actual_fps = original_fps / frame_interval

    # Set encoding parameters
    if image_format.lower() == "jpg":
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    pbar = tqdm(
        total=(end_frame - start_frame) // frame_interval,
        desc="Extracting frames",
        unit="frames",
    )

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            # Save frame
            frame_filename = f"frame_{saved_idx:06d}{ext}"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, params)
            frame_paths.append(str(frame_path))
            saved_idx += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()

    return frame_paths, actual_fps


def frame_generator(
    video_path: str,
    target_fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields frames from video without saving to disk.

    Args:
        video_path: Path to input video
        target_fps: Target FPS (None = use original FPS)
        start_frame: First frame to process
        end_frame: Last frame to process

    Yields:
        Tuple of (frame_index, frame_array in RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    # Calculate frame interval
    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
    else:
        frame_interval = round(original_fps / target_fps)

    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    output_idx = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield output_idx, frame_rgb
            output_idx += 1

        frame_idx += 1

    cap.release()


def create_video_from_frames(
    frame_paths: list,
    output_path: str,
    fps: float,
    codec: str = "mp4v",
) -> None:
    """
    Create video from frame images.

    Args:
        frame_paths: List of paths to frame images (in order)
        output_path: Path for output video
        fps: Frames per second
        codec: Video codec (default 'mp4v')
    """
    if not frame_paths:
        raise ValueError("No frames provided")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            writer.write(frame)

    writer.release()
