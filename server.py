#!/usr/bin/env python3
"""Simple HTTP server for the animation viewer (GLB/FBX)."""

import http.server
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote

PORT = 8080
PROJECT_ROOT = Path(__file__).parent

# Paths
VIDEO_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output_20260207_120705_motion_2_one_minute"


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        # API: get latest model (prefer GLB over FBX)
        if self.path == "/api/latest-model":
            # Prefer GLB (quaternion-native, no Euler wrapping issues)
            glb_files = sorted(OUTPUT_DIR.glob("*.glb"), key=os.path.getmtime)
            if glb_files:
                latest = glb_files[-1]
                self.send_json({"file": latest.name})
                return
            # Fallback to FBX
            fbx_files = sorted(OUTPUT_DIR.glob("*.fbx"), key=os.path.getmtime)
            if fbx_files:
                latest = fbx_files[-1]
                self.send_json({"file": latest.name})
            else:
                self.send_json({"file": None})
            return

        # API: get video path
        if self.path == "/api/video-path":
            # Try to find video matching the output dir name
            video_file = VIDEO_DIR / "motion_2_one_minute.mp4"
            if not video_file.exists():
                mp4_files = list(VIDEO_DIR.glob("*.mp4"))
                video_file = mp4_files[0] if mp4_files else None
            if video_file and video_file.exists():
                rel = video_file.relative_to(PROJECT_ROOT).as_posix()
                self.send_json({"path": "/" + rel})
            else:
                self.send_json({"path": None})
            return

        # Serve model files (GLB/FBX) from output dir
        if self.path.startswith("/output/"):
            filename = unquote(self.path.split("/")[-1])
            file_path = OUTPUT_DIR / filename
            if file_path.exists():
                # Set appropriate content type
                if filename.endswith('.glb'):
                    content_type = "model/gltf-binary"
                elif filename.endswith('.fbx'):
                    content_type = "application/octet-stream"
                else:
                    content_type = "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_error(404, f"File not found: {filename}")
            return

        # Default file serving
        super().do_GET()

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # Add CORS for all responses
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        # Quieter logging - only log non-200 or API calls
        if args and (str(args[1]) != "200" or "/api/" in str(args[0])):
            super().log_message(format, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    server = http.server.HTTPServer(("", port), ViewerHandler)
    print(f"Viewer server running at http://localhost:{port}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video dir: {VIDEO_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
