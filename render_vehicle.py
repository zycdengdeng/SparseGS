"""
Render trained 3DGS scene from vehicle camera viewpoints.

Coordinate transform chain:
  Vehicle Camera --(extrinsics)--> LiDAR --(inv world2lidar)--> World (COLMAP/roadside)

Usage:
  python render_vehicle.py \
    --model_path output/car_road/scene1 \
    --vehicle_calib /path/to/vehicle/calibration/ \
    --transform_json /path/to/world2lidar.json \
    --timestamp 1743583131842 \
    --camera_ids 1 2 3 4 5 6 7
"""

import torch
import numpy as np
import cv2
import json
import yaml
import os
import math
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from scene.cameras import MiniCam

import torchvision

# Vehicle camera definitions
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "front narrow 30°",  "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "front wide 120°",   "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "front-left 120°",   "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "front-right 120°",  "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "rear-left 60°",     "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "rear-right 60°",    "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "rear narrow 60°",   "resolution": (1920, 1080)},
}


def quaternion_to_rotation_matrix(q):
    """Quaternion (x, y, z, w) to rotation matrix."""
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])
    return R


def load_world2lidar(transform_json_path, timestamp_ms):
    """Load world2lidar transform for a given timestamp."""
    with open(transform_json_path, 'r') as f:
        transforms = json.load(f)

    best = None
    best_diff = float('inf')
    for entry in transforms:
        ts = entry['timestamp']
        ts_ms = ts * 1000 if ts < 1e12 else ts
        diff = abs(ts_ms - timestamp_ms)
        if diff < best_diff:
            best_diff = diff
            best = entry

    if best is None:
        raise ValueError(f"No transforms found in {transform_json_path}")

    rotvec = np.array(best['world2lidar']['rotation'])
    R_w2l = Rotation.from_rotvec(rotvec).as_matrix()
    t_w2l = np.array(best['world2lidar']['translation'])

    print(f"[world2lidar] timestamp diff: {best_diff:.0f}ms")
    return R_w2l, t_w2l


def load_vehicle_camera(calib_folder, cam_id):
    """Load vehicle camera intrinsics and extrinsics.

    Returns:
        K (3x3), D (distortion), R_cam2lidar (3x3), t_cam2lidar (3,), resolution (w, h)
    """
    calib_folder = Path(calib_folder)

    cam_subdir = calib_folder / "camera"
    base = cam_subdir if cam_subdir.is_dir() else calib_folder

    intr_path = base / f"camera_{cam_id:02d}_intrinsics.yaml"
    with open(intr_path, 'r') as f:
        intrinsics = yaml.safe_load(f)

    K = np.array(intrinsics['K']).reshape(3, 3)
    D = np.array(intrinsics['D'])

    extr_path = base / f"camera_{cam_id:02d}_extrinsics.yaml"
    with open(extr_path, 'r') as f:
        extrinsics = yaml.safe_load(f)

    transform = extrinsics['transform']
    q = [transform['rotation']['x'], transform['rotation']['y'],
         transform['rotation']['z'], transform['rotation']['w']]
    t = np.array([transform['translation']['x'],
                  transform['translation']['y'],
                  transform['translation']['z']])

    R_cam2lidar = quaternion_to_rotation_matrix(q)
    resolution = VEHICLE_CAMERAS[cam_id]["resolution"]

    return K, D, R_cam2lidar, t, resolution


def compute_undistorted_intrinsics(K, D, cam_id, resolution):
    """Compute new camera matrix after undistortion.

    Fisheye cameras (cam 2, 3, 4) use cv2.fisheye;
    standard cameras use cv2.getOptimalNewCameraMatrix.
    """
    w, h = resolution
    if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D[:4], (w, h), np.eye(3), balance=0.0
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    return new_K


def compute_vehicle_cam_pose(R_w2l, t_w2l, R_cam2lidar, t_cam2lidar):
    """Compute vehicle camera pose in world (COLMAP/roadside) coordinates.

    Returns:
        R_stored (3x3), T_stored (3,) matching DNGaussian convention
        (R_stored = w2c rotation, T_stored = w2c translation)
    """
    T_w2l = np.eye(4)
    T_w2l[:3, :3] = R_w2l
    T_w2l[:3, 3] = t_w2l

    T_c2l = np.eye(4)
    T_c2l[:3, :3] = R_cam2lidar
    T_c2l[:3, 3] = t_cam2lidar

    T_l2w = np.linalg.inv(T_w2l)
    T_c2w = T_l2w @ T_c2l
    T_w2c = np.linalg.inv(T_c2w)

    # DNGaussian/COLMAP convention: R is stored transposed (i.e. R_stored.T = R_w2c)
    R_stored = T_w2c[:3, :3].T
    T_stored = T_w2c[:3, 3]

    return R_stored, T_stored


def create_vehicle_minicam(R_stored, T_stored, fovx, fovy, width, height):
    """Create a MiniCam object for a vehicle camera viewpoint.

    Uses getWorld2View2 and getProjectionMatrix to match how the Camera class
    computes its transforms internally.
    """
    znear = 0.01
    zfar = 100.0

    world_view_transform = torch.tensor(
        getWorld2View2(R_stored, T_stored)
    ).transpose(0, 1).cuda()

    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).cuda()

    full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0)
    ).squeeze(0)

    cam = MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )
    return cam


def find_latest_ply(model_path):
    """Find the latest point_cloud.ply iteration in model_path."""
    pc_dir = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_dir):
        raise FileNotFoundError(f"No point_cloud directory in {model_path}")

    iterations = []
    for d in os.listdir(pc_dir):
        if d.startswith("iteration_"):
            try:
                iterations.append(int(d.split("_")[1]))
            except ValueError:
                continue

    if not iterations:
        raise FileNotFoundError(f"No iteration_* subdirs in {pc_dir}")

    latest = max(iterations)
    ply_path = os.path.join(pc_dir, f"iteration_{latest}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    return ply_path, latest


def render_vehicle_cameras(model_path, vehicle_calib, transform_json, timestamp_ms,
                           camera_ids, pipeline, output_dir, render_scale=1):
    """Main rendering function for vehicle camera viewpoints."""

    # Load trained model from PLY (no optimizer/training_args needed)
    print(f"Loading model from {model_path}")
    gaussians = GaussianModel(3)  # sh_degree=3

    ply_path, iteration = find_latest_ply(model_path)
    print(f"Loading PLY from iteration {iteration}: {ply_path}")
    gaussians.load_ply(ply_path)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Load world2lidar transform
    R_w2l, t_w2l = load_world2lidar(transform_json, timestamp_ms)

    # Create output directory
    vehicle_render_path = os.path.join(output_dir, "vehicle_renders")
    os.makedirs(vehicle_render_path, exist_ok=True)

    print(f"\nRendering {len(camera_ids)} vehicle cameras...")

    for cam_id in tqdm(camera_ids, desc="Vehicle cameras"):
        cam_info = VEHICLE_CAMERAS.get(cam_id)
        if cam_info is None:
            print(f"Unknown camera ID: {cam_id}, skipping")
            continue

        cam_name = cam_info["name"]

        # Load calibration
        K, D, R_cam2lidar, t_cam2lidar, resolution = load_vehicle_camera(
            vehicle_calib, cam_id
        )
        w, h = resolution

        # Compute undistorted intrinsics
        new_K = compute_undistorted_intrinsics(K, D, cam_id, resolution)

        # Apply render scale
        if render_scale != 1:
            w = w // render_scale
            h = h // render_scale
            scale_x = w / resolution[0]
            scale_y = h / resolution[1]
            new_K = new_K.copy()
            new_K[0, :] *= scale_x
            new_K[1, :] *= scale_y

        # Compute FoV from undistorted intrinsics
        fx, fy = new_K[0, 0], new_K[1, 1]
        fovx = focal2fov(fx, w)
        fovy = focal2fov(fy, h)

        # Compute camera pose in world coordinates
        R_stored, T_stored = compute_vehicle_cam_pose(
            R_w2l, t_w2l, R_cam2lidar, t_cam2lidar
        )

        # Create virtual camera (MiniCam - no image needed)
        cam = create_vehicle_minicam(R_stored, T_stored, fovx, fovy, w, h)

        # Render
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipeline, bg_color)

        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        alpha_depth = render_pkg["alpha_depth"]

        # Normalize depth for visualization
        depth_vis = torch.exp(depth - depth.min())
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-5)

        # Normalize alpha_depth
        alpha_vis = (alpha_depth - alpha_depth.min()) / (alpha_depth.max() - alpha_depth.min() + 1e-5)

        # Save outputs
        cam_dir = os.path.join(vehicle_render_path, cam_name)
        os.makedirs(cam_dir, exist_ok=True)

        torchvision.utils.save_image(
            rendering, os.path.join(cam_dir, "render.png")
        )
        torchvision.utils.save_image(
            1 - depth_vis, os.path.join(cam_dir, "depth.png")
        )
        torchvision.utils.save_image(
            alpha_vis, os.path.join(cam_dir, "alpha.png")
        )

        # Save camera info
        cam_meta = {
            "cam_id": cam_id,
            "cam_name": cam_name,
            "resolution": [w, h],
            "fovx_deg": math.degrees(fovx),
            "fovy_deg": math.degrees(fovy),
            "fx": float(fx),
            "fy": float(fy),
            "camera_center": cam.camera_center.cpu().numpy().tolist(),
        }
        with open(os.path.join(cam_dir, "camera_info.json"), 'w') as f:
            json.dump(cam_meta, f, indent=2)

        tqdm.write(f"  {cam_name}: rendered {w}x{h}, "
                   f"fov={math.degrees(fovx):.1f}x{math.degrees(fovy):.1f} deg")

    print(f"\nResults saved to {vehicle_render_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render 3DGS scene from vehicle camera viewpoints")
    model = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--vehicle_calib", type=str, required=True,
                        help="Path to vehicle calibration folder")
    parser.add_argument("--transform_json", type=str, required=True,
                        help="Path to world2lidar transform JSON")
    parser.add_argument("--timestamp", type=int, required=True,
                        help="Timestamp in milliseconds for world2lidar lookup")
    parser.add_argument("--camera_ids", type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7],
                        help="Vehicle camera IDs to render (default: all 7)")
    parser.add_argument("--render_scale", type=int, default=4,
                        help="Downscale factor for rendering resolution (default: 4)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: model_path)")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    pipe = pipeline_params.extract(args)
    output_dir = getattr(args, 'output_dir', None) or args.model_path

    render_vehicle_cameras(
        model_path=args.model_path,
        vehicle_calib=args.vehicle_calib,
        transform_json=args.transform_json,
        timestamp_ms=args.timestamp,
        camera_ids=args.camera_ids,
        pipeline=pipe,
        output_dir=output_dir,
        render_scale=args.render_scale,
    )
