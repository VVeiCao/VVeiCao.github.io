#!/usr/bin/env python3
"""Build a self-contained Viser playback HTML for one Redirect4D-Bench track.

The output is a single .html file that embeds the recorded scene and Viser's
client bundle, so it can be served as a static asset and iframed into the
project page (no live server, no extra build directory).

Reads the dataset layout shipped on Hugging Face / produced by the bench:

    tracks/<track>/pointcloud/global_background.ply
    tracks/<track>/pointcloud/<frame>/foreground_5_views_aligned_smooth.ply

and the per-trajectory metadata used by serve_pointcloud_viser.py.

Usage example:
    python webpage/build_viser_playback.py \
        --dataset-root /path/to/redirect4d_bench \
        --track bear_NnAlfavy2us_003_001_seq1 \
        --output webpage/ready-ben/web_assets/viewers/bear.html \
        --bg-subsample 16 --fg-subsample 4 --fps 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import viser

SMOOTH_NAME = "foreground_5_views_aligned_smooth.ply"
TRAJECTORY_PALETTE = (
    (40, 145, 255),
    (255, 132, 40),
    (96, 210, 128),
    (210, 95, 255),
)
ORIGINAL_TRAJECTORY_COLOR = (235, 90, 105)


def load_ply(path: Path, stride: int):
    mesh = trimesh.load(path, process=False)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    if stride > 1 and len(points) > stride:
        points = points[::stride]
        colors = colors[::stride]
    return points, colors


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / max(float(np.linalg.norm(quat)), 1e-8)


def extrinsic_to_camera_pose(extrinsic):
    t_cam_world = np.eye(4, dtype=np.float64)
    t_cam_world[:3, :] = np.asarray(extrinsic, dtype=np.float64)
    t_world_cam = np.linalg.inv(t_cam_world)
    return (
        t_world_cam[:3, 3].astype(np.float32),
        matrix_to_wxyz(t_world_cam[:3, :3]),
    )


def fov_aspect_from_intrinsic(intrinsic, image_size=None,
                               default_fov=0.55, default_aspect=16.0 / 9.0):
    if intrinsic is None:
        return default_fov, default_aspect
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if image_size is None:
        width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
        height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
    else:
        height, width = image_size
    fy = max(float(intrinsic[1, 1]), 1e-8)
    return float(2.0 * np.arctan(float(height) / (2.0 * fy))), float(width) / float(height)


def load_trajectories(track_dir: Path):
    redirected = track_dir / "redirected"
    if not redirected.exists():
        return {}
    out = {}
    for traj_dir in sorted(p for p in redirected.iterdir() if p.is_dir()):
        path = traj_dir / "trajectory.json"
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        camera_path = raw.get("camera_path") or raw.get("keyframes") or []
        positions, wxyzs, fovs, aspects = [], [], [], []
        for item in camera_path:
            if "extrinsic" in item:
                pos, rot = extrinsic_to_camera_pose(item["extrinsic"])
            elif "position" in item:
                pos = np.asarray(item["position"], dtype=np.float32)
                rot = np.asarray(item.get("wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            else:
                continue
            fov, aspect = fov_aspect_from_intrinsic(item.get("intrinsic"))
            positions.append(pos)
            wxyzs.append(rot)
            fovs.append(float(item.get("fov", fov)))
            aspects.append(float(item.get("aspect", aspect)))
        if positions:
            out[traj_dir.name] = {
                "points": np.asarray(positions, dtype=np.float32),
                "wxyz": np.asarray(wxyzs, dtype=np.float32),
                "fov": fovs,
                "aspect": aspects,
            }
    return out


def load_original_camera(track_dir: Path):
    path = track_dir / "camera.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    image_size = raw.get("image_size")
    if isinstance(image_size, list) and len(image_size) >= 2:
        image_size = (int(image_size[0]), int(image_size[1]))
    else:
        image_size = None
    points, wxyzs, fovs, aspects = [], [], [], []
    for k in sorted(k for k in raw.keys() if k.isdigit()):
        item = raw[k]
        if not isinstance(item, dict) or "extrinsic" not in item:
            continue
        pos, rot = extrinsic_to_camera_pose(item["extrinsic"])
        fov, aspect = fov_aspect_from_intrinsic(item.get("intrinsic"), image_size=image_size)
        points.append(pos)
        wxyzs.append(rot)
        fovs.append(fov)
        aspects.append(aspect)
    if not points:
        return None
    return {
        "points": np.asarray(points, dtype=np.float32),
        "wxyz": np.asarray(wxyzs, dtype=np.float32),
        "fov": fovs,
        "aspect": aspects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Folder containing tracks/<track>/...")
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the standalone HTML.")
    parser.add_argument("--bg-subsample", type=int, default=16)
    parser.add_argument("--fg-subsample", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--point-size", type=float, default=0.007)
    parser.add_argument("--fg-point-size", type=float, default=0.004)
    parser.add_argument("--trajectory-camera-scale", type=float, default=0.08)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--port", type=int, default=18091,
                        help="Local viser port (server is started but not exposed).")
    parser.add_argument("--initial-pull-back", type=float, default=1.4,
                        help="Multiplier on the (centroid->camera) ray for the initial view distance.")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark",
                        help="Viser playback theme.")
    parser.add_argument("--dark", action="store_true", default=True,
                        help="(Deprecated; kept for backwards compatibility. Use --theme.)")
    args = parser.parse_args()

    track_dir = args.dataset_root / "tracks" / args.track
    if not track_dir.is_dir():
        sys.exit(f"track folder not found: {track_dir}")

    pc_root = track_dir / "pointcloud"
    if not pc_root.exists():
        sys.exit(f"pointcloud folder not found: {pc_root}")

    bg_path = pc_root / "global_background.ply"
    bg = load_ply(bg_path, args.bg_subsample) if bg_path.exists() else None

    frame_dirs = sorted(p for p in pc_root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.frame_step > 1:
        frame_dirs = frame_dirs[::args.frame_step]
    if args.max_frames:
        frame_dirs = frame_dirs[:args.max_frames]

    frames = []
    for frame_dir in frame_dirs:
        smooth_path = frame_dir / SMOOTH_NAME
        if smooth_path.exists():
            frames.append({"frame": frame_dir.name,
                           "data": load_ply(smooth_path, args.fg_subsample)})

    if bg is None and not frames:
        sys.exit("no point clouds found")

    trajectories = load_trajectories(track_dir)
    original_camera = load_original_camera(track_dir)

    print(f"[recorder] track={args.track} frames={len(frames)} "
          f"trajectories={len(trajectories)} bg_pts={'-' if bg is None else len(bg[0])}",
          flush=True)

    server = viser.ViserServer(host="127.0.0.1", port=args.port, verbose=False)
    server.scene.world_axes.visible = False

    if bg is not None:
        points, colors = bg
        server.scene.add_point_cloud("/background", points=points, colors=colors,
                                     point_size=args.point_size)

    # Compute scene centroid from the first foreground frame (or background as a fallback)
    # so we can frame the view around the actual subject rather than the world origin.
    ref_points = None
    if frames:
        ref_points = frames[0]["data"][0]
    elif bg is not None:
        ref_points = bg[0]
    if ref_points is not None and len(ref_points) > 0:
        scene_center = np.asarray(np.mean(ref_points, axis=0), dtype=np.float64)
    else:
        scene_center = np.zeros(3, dtype=np.float64)

    fg_handles = []
    for idx, item in enumerate(frames):
        points, colors = item["data"]
        h = server.scene.add_point_cloud(
            f"/foreground/{item['frame']}",
            points=points, colors=colors,
            point_size=args.fg_point_size,
            visible=(idx == 0),
        )
        fg_handles.append(h)

    traj_color = TRAJECTORY_PALETTE[0]
    first_traj_name = next(iter(trajectories), None)
    target_camera_handle = None
    if first_traj_name is not None:
        traj = trajectories[first_traj_name]
        if len(traj["points"]) >= 2:
            server.scene.add_spline_catmull_rom(
                f"/target_trajectory/{first_traj_name}",
                points=traj["points"], line_width=3.0, color=traj_color,
            )
        target_camera_handle = server.scene.add_camera_frustum(
            "/target_camera/current",
            fov=traj["fov"][0], aspect=traj["aspect"][0],
            scale=args.trajectory_camera_scale, line_width=2.0,
            color=traj_color, wxyz=traj["wxyz"][0], position=traj["points"][0],
        )

    original_camera_handle = None
    if original_camera is not None and len(original_camera["points"]) >= 2:
        server.scene.add_spline_catmull_rom(
            "/original_camera/path",
            points=original_camera["points"], line_width=4.0,
            color=ORIGINAL_TRAJECTORY_COLOR,
        )
        original_camera_handle = server.scene.add_camera_frustum(
            "/original_camera/current",
            fov=original_camera["fov"][0], aspect=original_camera["aspect"][0],
            scale=args.trajectory_camera_scale * 1.35, line_width=2.0,
            color=ORIGINAL_TRAJECTORY_COLOR,
            wxyz=original_camera["wxyz"][0], position=original_camera["points"][0],
        )

    # Set initial camera pose to match the source camera at frame 0, then pull
    # back along the centroid->camera ray so the subject isn't cropped or rotated.
    def quat_to_R(wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = (float(v) for v in wxyz)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
            [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    pull_back = float(args.initial_pull_back)
    if original_camera is not None and len(original_camera["points"]) > 0:
        cam_pos = np.asarray(original_camera["points"][0], dtype=np.float64)
        view_dir = cam_pos - scene_center
        if np.linalg.norm(view_dir) > 1e-6:
            cam_pos = scene_center + view_dir * pull_back
        # Camera-up in world: in CV convention camera +y is down in image, so
        # world up for the viewer = -column[1] of R_world_cam.
        R_world_cam = quat_to_R(np.asarray(original_camera["wxyz"][0], dtype=np.float64))
        world_up = -R_world_cam[:, 1]
        server.initial_camera.position = tuple(cam_pos.tolist())
        server.initial_camera.look_at = tuple(scene_center.tolist())
        server.initial_camera.up = tuple(world_up.tolist())
    else:
        cam_pos = scene_center + np.array([0.0, -2.0, 1.0], dtype=np.float64) * pull_back
        server.initial_camera.position = tuple(cam_pos.tolist())
        server.initial_camera.look_at = tuple(scene_center.tolist())
        server.initial_camera.up = (0.0, 0.0, 1.0)

    serializer = server.get_scene_serializer()
    dt = 1.0 / max(float(args.fps), 1.0)

    n = max(len(frames), 1)
    for step in range(n):
        if frames:
            for idx, h in enumerate(fg_handles):
                h.visible = (idx == step)
            cur_frame_idx = step
        else:
            cur_frame_idx = 0

        if target_camera_handle is not None and first_traj_name is not None:
            traj = trajectories[first_traj_name]
            j = min(cur_frame_idx, len(traj["points"]) - 1)
            target_camera_handle.position = traj["points"][j]
            target_camera_handle.wxyz = traj["wxyz"][j]
            target_camera_handle.fov = traj["fov"][j]
            target_camera_handle.aspect = traj["aspect"][j]

        if original_camera_handle is not None and original_camera is not None:
            j = min(cur_frame_idx, len(original_camera["points"]) - 1)
            original_camera_handle.position = original_camera["points"][j]
            original_camera_handle.wxyz = original_camera["wxyz"][j]
            original_camera_handle.fov = original_camera["fov"][j]
            original_camera_handle.aspect = original_camera["aspect"][j]

        serializer.insert_sleep(dt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html_text = serializer.as_html(dark_mode=(args.theme == "dark"))
    args.output.write_text(html_text)
    print(f"[recorder] wrote {args.output} ({len(html_text) / 1024 / 1024:.1f} MiB)",
          flush=True)
    server.stop()


if __name__ == "__main__":
    main()
