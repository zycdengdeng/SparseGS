#!/usr/bin/env python3
"""
Prepare roadside camera dataset for SparseGS training.

Converts raw Car-Road cooperative dataset to SparseGS COLMAP format:
  - Reads calib.json for camera parameters
  - Reads LiDAR PCD (ASCII format)
  - Undistorts images using OpenCV
  - Colors point cloud from camera projections
  - Generates COLMAP format (cameras.txt, images.txt, points3D.ply)
  - Optionally generates depth maps using Depth Anything

Supports two input formats:
  1. Raw dataset: /mnt/car_road_data_TianJin/{scene}/road/...
  2. Self-extracted: self_Dataset/ with calib.json, *.pcd, img/pinhole*/

Usage:
  # From raw dataset
  python prepare_roadside_for_sparsegs.py \
    --raw_dataset /mnt/car_road_data_TianJin \
    --scene 053 --timestamp 1743583131842 \
    --output data/car_road/scene053

  # From self-extracted dataset
  python prepare_roadside_for_sparsegs.py \
    --self_dataset /path/to/self_Dataset \
    --timestamp 1743583131842 \
    --output data/car_road/scene053
"""

import argparse
import json
import os
import sys
import glob
import struct
import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement


# ============================================================================
# Camera mapping: image folder name -> calib.json camera key
# This mapping is hardcoded based on the dataset convention
# ============================================================================
PINHOLE_TO_CAM = {
    "pinhole0": "3",   # cam3
    "pinhole1": "6",   # cam6
    "pinhole2": "9",   # cam9
    "pinhole3": "0",   # cam0
}

PINHOLE_NAMES = ["pinhole0", "pinhole1", "pinhole2", "pinhole3"]


def rodrigues_to_rotmat(rvec):
    """Convert Rodrigues rotation vector to 3x3 rotation matrix."""
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_pcd_ascii(pcd_path):
    """Read ASCII PCD file. Returns (N,3) points and (N,) intensities."""
    points = []
    intensities = []
    header_done = False
    num_points = 0

    with open(pcd_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not header_done:
                if line.startswith('POINTS'):
                    num_points = int(line.split()[1])
                if line.startswith('DATA'):
                    header_done = True
                continue
            parts = line.split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                intensity = float(parts[3]) if len(parts) > 3 else 0.0
                points.append([x, y, z])
                intensities.append(intensity)

    points = np.array(points, dtype=np.float64)
    intensities = np.array(intensities, dtype=np.float64)
    print(f"  Read {len(points)} points from PCD (expected {num_points})")
    return points, intensities


def parse_calib(calib_path):
    """Parse calib.json and extract pinhole camera parameters.

    Returns dict: pinhole_name -> {
        'K': 3x3 intrinsic matrix,
        'dist': distortion coefficients,
        'R_w2c': 3x3 rotation matrix (world-to-camera),
        't_w2c': 3D translation vector (world-to-camera),
        'K_new': undistorted intrinsic matrix,
        'roi': undistortion ROI
    }
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    img_w, img_h = calib["imgSize"]["notFish"]
    cameras = {}

    for pinhole_name, cam_key in PINHOLE_TO_CAM.items():
        cam_data = calib["camera"][cam_key]
        assert cam_data["isFish"] == 0, f"Camera {cam_key} is fisheye, expected pinhole"

        # Intrinsic matrix (row-major 3x3)
        K = np.array(cam_data["intri"], dtype=np.float64).reshape(3, 3)

        # Distortion coefficients
        dist = np.array(cam_data["distor"], dtype=np.float64)

        # Extrinsic: virtualLidarToCam (world-to-camera)
        rvec = cam_data["virtualLidarToCam"]["rotate"]
        tvec = cam_data["virtualLidarToCam"]["trans"]
        R_w2c = rodrigues_to_rotmat(rvec)
        t_w2c = np.array(tvec, dtype=np.float64)

        # Compute undistorted camera matrix
        K_new, roi = cv2.getOptimalNewCameraMatrix(
            K, dist, (img_w, img_h), alpha=0
        )

        cameras[pinhole_name] = {
            'K': K,
            'dist': dist,
            'R_w2c': R_w2c,
            't_w2c': t_w2c,
            'K_new': K_new,
            'roi': roi,
            'width': img_w,
            'height': img_h,
        }

        # Print camera center
        center = -R_w2c.T @ t_w2c
        fx_new = K_new[0, 0]
        fy_new = K_new[1, 1]
        cx_new = K_new[0, 2]
        cy_new = K_new[1, 2]
        print(f"  {pinhole_name} (cam{cam_key}): center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
              f"fx={fx_new:.1f}, fy={fy_new:.1f}, cx={cx_new:.1f}, cy={cy_new:.1f}")

    return cameras


def undistort_image(image, K, dist, K_new):
    """Undistort image using OpenCV."""
    return cv2.undistort(image, K, dist, None, K_new)


def project_points_to_camera(points, R_w2c, t_w2c, K_new, width, height, min_depth=0.5):
    """Project 3D points to camera image plane.

    Returns:
        uv: (N, 2) pixel coordinates
        depths: (N,) depth values
        mask: (N,) boolean mask for valid projections
    """
    # Transform to camera coordinates
    P_cam = (R_w2c @ points.T).T + t_w2c  # (N, 3)
    depths = P_cam[:, 2]

    # Project to image plane
    fx, fy = K_new[0, 0], K_new[1, 1]
    cx, cy = K_new[0, 2], K_new[1, 2]

    u = fx * P_cam[:, 0] / depths + cx
    v = fy * P_cam[:, 1] / depths + cy

    # Valid mask
    mask = (depths > min_depth) & \
           (u >= 0) & (u < width) & \
           (v >= 0) & (v < height)

    uv = np.stack([u, v], axis=1)
    return uv, depths, mask


def color_pointcloud(points, cameras, images, filter_visible=True):
    """Color point cloud by projecting to cameras.

    For each point, take the color from the camera where the point is closest.
    Returns (N, 3) uint8 RGB colors and boolean mask of colored points.
    """
    N = len(points)
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_depth = np.full(N, np.inf)
    colored = np.zeros(N, dtype=bool)

    for pinhole_name in PINHOLE_NAMES:
        cam = cameras[pinhole_name]
        img = images[pinhole_name]  # (H, W, 3) BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        uv, depths, mask = project_points_to_camera(
            points, cam['R_w2c'], cam['t_w2c'], cam['K_new'],
            cam['width'], cam['height']
        )

        # For valid points, update color if this camera is closer
        valid_idx = np.where(mask)[0]
        for idx in valid_idx:
            if depths[idx] < best_depth[idx]:
                u_int = int(round(uv[idx, 0]))
                v_int = int(round(uv[idx, 1]))
                u_int = min(max(u_int, 0), cam['width'] - 1)
                v_int = min(max(v_int, 0), cam['height'] - 1)
                colors[idx] = img_rgb[v_int, u_int]
                best_depth[idx] = depths[idx]
                colored[idx] = True

    num_colored = np.sum(colored)
    print(f"  Colored {num_colored}/{N} points ({100*num_colored/N:.1f}%)")

    if filter_visible:
        return colors[colored], colored
    return colors, colored


def write_cameras_txt(path, cameras):
    """Write COLMAP cameras.txt file."""
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(PINHOLE_NAMES)}\n")

        for idx, pinhole_name in enumerate(PINHOLE_NAMES):
            cam = cameras[pinhole_name]
            K_new = cam['K_new']
            camera_id = idx + 1
            fx = K_new[0, 0]
            fy = K_new[1, 1]
            cx = K_new[0, 2]
            cy = K_new[1, 2]
            f.write(f"{camera_id} PINHOLE {cam['width']} {cam['height']} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_images_txt(path, cameras, timestamp):
    """Write COLMAP images.txt file."""
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(PINHOLE_NAMES)}\n")

        for idx, pinhole_name in enumerate(PINHOLE_NAMES):
            cam = cameras[pinhole_name]
            image_id = idx + 1
            camera_id = idx + 1

            # Convert rotation matrix to COLMAP quaternion
            qvec = rotmat2qvec(cam['R_w2c'])
            tvec = cam['t_w2c']
            image_name = f"{pinhole_name}_{timestamp}.png"

            f.write(f"{image_id} {qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                    f"{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} "
                    f"{camera_id} {image_name}\n")
            f.write("\n")  # Empty line for 2D points


def write_points3d_ply(path, points, colors):
    """Write colored point cloud as PLY file compatible with SparseGS."""
    N = len(points)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros((N, 3), dtype=np.float32)
    elements = np.empty(N, dtype=dtype)
    attributes = np.concatenate([
        points.astype(np.float32),
        normals,
        colors.astype(np.uint8)
    ], axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)
    print(f"  Wrote {N} points to {path}")


def write_points3d_txt(path):
    """Write empty points3D.txt placeholder."""
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")


def find_scene_folder(raw_dataset_root, scene_num):
    """Find scene folder by scene number (e.g., 053 -> 053_*)."""
    scene_str = f"{int(scene_num):03d}"
    candidates = glob.glob(os.path.join(raw_dataset_root, f"{scene_str}_*"))
    if not candidates:
        # Also try without zero-padding
        candidates = glob.glob(os.path.join(raw_dataset_root, f"{int(scene_num)}_*"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No scene folder found for scene {scene_str} in {raw_dataset_root}")
    if len(candidates) > 1:
        print(f"  Warning: Multiple matches for scene {scene_str}, using {candidates[0]}")
    return candidates[0]


def locate_raw_data(raw_dataset_root, scene_num, timestamp):
    """Locate data files in raw dataset structure.

    Returns dict with paths to: calib_json, pcd, images (dict pinhole_name -> path)
    """
    scene_folder = find_scene_folder(raw_dataset_root, scene_num)
    print(f"  Scene folder: {scene_folder}")

    # Calibration file - may be at scene root, in road/, or shared in support_info/
    calib_candidates = [
        os.path.join(scene_folder, "calib.json"),
        os.path.join(scene_folder, "road", "calib.json"),
        os.path.join(raw_dataset_root, "support_info", "calib.json"),
    ]
    calib_path = None
    for c in calib_candidates:
        if os.path.exists(c):
            calib_path = c
            break
    if calib_path is None:
        raise FileNotFoundError(
            f"calib.json not found in {scene_folder}. "
            f"Use --calib to specify the path manually."
        )

    # PCD file - in road/lidar/merged_pcd/ or scene root
    ts = str(timestamp)
    pcd_candidates = [
        os.path.join(scene_folder, "road", "lidar", "merged_pcd", f"{ts}.pcd"),
        os.path.join(scene_folder, f"{ts}.pcd"),
    ]
    pcd_path = None
    for p in pcd_candidates:
        if os.path.exists(p):
            pcd_path = p
            break
    if pcd_path is None:
        raise FileNotFoundError(f"PCD file not found for timestamp {ts}")

    # Image files
    images = {}
    for pinhole_name in PINHOLE_NAMES:
        cam_key = PINHOLE_TO_CAM[pinhole_name]
        img_candidates = [
            # Raw dataset format: cam{N}_TIMESTAMP.png
            os.path.join(scene_folder, "road", "cameras", pinhole_name, f"cam{cam_key}_{ts}.png"),
            # Self-extracted format: TIMESTAMP.png
            os.path.join(scene_folder, "road", "cameras", pinhole_name, f"{ts}.png"),
            # Alternative: under img/ directory
            os.path.join(scene_folder, "img", pinhole_name, f"{ts}.png"),
        ]
        img_path = None
        for ip in img_candidates:
            if os.path.exists(ip):
                img_path = ip
                break
        if img_path is None:
            raise FileNotFoundError(
                f"Image not found for {pinhole_name} timestamp {ts}. "
                f"Tried: {img_candidates}"
            )
        images[pinhole_name] = img_path

    return {
        'calib_json': calib_path,
        'pcd': pcd_path,
        'images': images,
    }


def locate_self_dataset(self_dataset_root, timestamp):
    """Locate data files in self-extracted dataset structure."""
    ts = str(timestamp)

    calib_path = os.path.join(self_dataset_root, "calib.json")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"calib.json not found in {self_dataset_root}")

    pcd_path = os.path.join(self_dataset_root, f"{ts}.pcd")
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")

    images = {}
    for pinhole_name in PINHOLE_NAMES:
        img_path = os.path.join(self_dataset_root, "img", pinhole_name, f"{ts}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        images[pinhole_name] = img_path

    return {
        'calib_json': calib_path,
        'pcd': pcd_path,
        'images': images,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare roadside data for SparseGS")
    # Input source (choose one)
    parser.add_argument('--raw_dataset', type=str, default=None,
                        help='Root of raw dataset (e.g., /mnt/car_road_data_TianJin)')
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene number (e.g., 053)')
    parser.add_argument('--self_dataset', type=str, default=None,
                        help='Path to self-extracted dataset folder')
    # Required
    parser.add_argument('--timestamp', type=str, required=True,
                        help='Timestamp to process (e.g., 1743583131842)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (e.g., data/car_road/scene053)')
    # Options
    parser.add_argument('--calib', type=str, default=None,
                        help='Override path to calib.json (e.g., /mnt/car_road_data_TianJin/support_info/calib.json)')
    parser.add_argument('--no_filter_visible', action='store_true',
                        help='Keep all points (including invisible ones)')
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help='Minimum depth for point projection (meters)')

    args = parser.parse_args()

    # Validate input
    if args.raw_dataset and args.self_dataset:
        parser.error("Specify either --raw_dataset or --self_dataset, not both")
    if not args.raw_dataset and not args.self_dataset:
        parser.error("Must specify either --raw_dataset or --self_dataset")
    if args.raw_dataset and not args.scene:
        parser.error("--scene is required when using --raw_dataset")

    timestamp = args.timestamp

    # ========================================================================
    # Step 1: Locate data files
    # ========================================================================
    print("\n[Step 1] Locating data files...")
    if args.raw_dataset:
        data_paths = locate_raw_data(args.raw_dataset, args.scene, timestamp)
    else:
        data_paths = locate_self_dataset(args.self_dataset, timestamp)

    # Override calib path if specified
    if args.calib:
        if not os.path.exists(args.calib):
            raise FileNotFoundError(f"Specified calib file not found: {args.calib}")
        data_paths['calib_json'] = args.calib

    print(f"  Calib: {data_paths['calib_json']}")
    print(f"  PCD:   {data_paths['pcd']}")
    for k, v in data_paths['images'].items():
        print(f"  {k}: {v}")

    # ========================================================================
    # Step 2: Parse calibration
    # ========================================================================
    print("\n[Step 2] Parsing calibration...")
    cameras = parse_calib(data_paths['calib_json'])

    # ========================================================================
    # Step 3: Read and undistort images
    # ========================================================================
    print("\n[Step 3] Reading and undistorting images...")
    images_raw = {}
    images_undist = {}

    for pinhole_name in PINHOLE_NAMES:
        img = cv2.imread(data_paths['images'][pinhole_name])
        if img is None:
            raise RuntimeError(f"Failed to read image: {data_paths['images'][pinhole_name]}")
        images_raw[pinhole_name] = img

        cam = cameras[pinhole_name]
        img_undist = undistort_image(img, cam['K'], cam['dist'], cam['K_new'])
        images_undist[pinhole_name] = img_undist
        print(f"  {pinhole_name}: {img.shape[1]}x{img.shape[0]} -> undistorted")

    # ========================================================================
    # Step 4: Read point cloud
    # ========================================================================
    print("\n[Step 4] Reading point cloud...")
    points, intensities = read_pcd_ascii(data_paths['pcd'])
    print(f"  Point cloud range: X=[{points[:,0].min():.1f}, {points[:,0].max():.1f}], "
          f"Y=[{points[:,1].min():.1f}, {points[:,1].max():.1f}], "
          f"Z=[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")

    # ========================================================================
    # Step 5: Color point cloud
    # ========================================================================
    print("\n[Step 5] Coloring point cloud from camera projections...")
    filter_visible = not args.no_filter_visible
    colors, colored_mask = color_pointcloud(
        points, cameras, images_undist, filter_visible=filter_visible
    )

    if filter_visible:
        points_out = points[colored_mask]
        colors_out = colors
    else:
        points_out = points
        colors_out = colors

    print(f"  Output point cloud: {len(points_out)} points")
    print(f"  Z range after filtering: [{points_out[:,2].min():.1f}, {points_out[:,2].max():.1f}]")

    # ========================================================================
    # Step 6: Create output directories
    # ========================================================================
    print("\n[Step 6] Creating output directories...")
    output_dir = args.output
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sparse", "0"), exist_ok=True)

    # ========================================================================
    # Step 7: Write undistorted images
    # ========================================================================
    print("\n[Step 7] Writing undistorted images...")
    for pinhole_name in PINHOLE_NAMES:
        out_name = f"{pinhole_name}_{timestamp}.png"
        out_path = os.path.join(output_dir, "images", out_name)
        cv2.imwrite(out_path, images_undist[pinhole_name])
        print(f"  {out_name}")

    # ========================================================================
    # Step 8: Write COLMAP files
    # ========================================================================
    print("\n[Step 8] Writing COLMAP sparse reconstruction files...")

    cameras_txt_path = os.path.join(output_dir, "sparse", "0", "cameras.txt")
    write_cameras_txt(cameras_txt_path, cameras)
    print(f"  cameras.txt")

    images_txt_path = os.path.join(output_dir, "sparse", "0", "images.txt")
    write_images_txt(images_txt_path, cameras, timestamp)
    print(f"  images.txt")

    ply_path = os.path.join(output_dir, "sparse", "0", "points3D.ply")
    write_points3d_ply(ply_path, points_out, colors_out)

    points3d_txt_path = os.path.join(output_dir, "sparse", "0", "points3D.txt")
    write_points3d_txt(points3d_txt_path)
    print(f"  points3D.txt (placeholder)")

    # ========================================================================
    # Step 9: Print summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  images/          : {len(PINHOLE_NAMES)} undistorted images (1280x720)")
    print(f"  sparse/0/        : COLMAP reconstruction")
    print(f"    cameras.txt    : {len(PINHOLE_NAMES)} PINHOLE cameras")
    print(f"    images.txt     : {len(PINHOLE_NAMES)} image registrations")
    print(f"    points3D.ply   : {len(points_out)} colored points")
    print(f"  depths/          : (empty - run generate_depth.py next)")

    print(f"\nNext steps:")
    print(f"  1. Generate depth maps:")
    print(f"     python scripts/generate_depth.py --input {output_dir}/images --output {output_dir}/depths")
    print(f"  2. Train SparseGS:")
    print(f"     python train.py -s {output_dir} --model_path output/car_road/scene053 \\")
    print(f"       --lambda_pearson 0.05 --lambda_local_pearson 0.15 --beta 5.0 \\")
    print(f"       --iterations 30000 -r 1")


if __name__ == "__main__":
    main()
