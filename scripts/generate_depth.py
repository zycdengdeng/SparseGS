#!/usr/bin/env python3
"""
Generate monocular depth maps for SparseGS using Depth Anything V2.

SparseGS expects depth maps as .npy files in a 'depths/' directory,
one per image. The depth is normalized to [0,1] at load time by SparseGS,
so absolute scale doesn't matter - only relative depth ordering matters
(Pearson correlation loss).

Usage:
  python scripts/generate_depth.py \
    --input data/car_road/scene053/images \
    --output data/car_road/scene053/depths \
    --model_size vits  # vits/vitb/vitl/vitg

If Depth Anything V2 is not available, falls back to:
  1. BoostingMonocularDepth (SparseGS built-in)
  2. LiDAR-projected sparse depth from depth_maps/ directory
"""

import argparse
import os
import sys
import glob
import numpy as np
from pathlib import Path

def generate_with_depth_anything_v2(input_dir, output_dir, model_size='vits'):
    """Generate depth maps using Depth Anything V2."""
    try:
        import torch
        import cv2
        from PIL import Image
    except ImportError:
        print("PyTorch or OpenCV not available")
        return False

    try:
        # Try to import Depth Anything V2
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("Depth Anything V2 not installed. Install with:")
        print("  pip install depth-anything-v2")
        print("  Or clone: git clone https://github.com/DepthAnything/Depth-Anything-V2")
        return False

    # Model configs
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    if model_size not in model_configs:
        print(f"Unknown model size: {model_size}. Available: {list(model_configs.keys())}")
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = DepthAnythingV2(**model_configs[model_size])
    # Try to load pretrained weights
    weight_path = f'checkpoints/depth_anything_v2_{model_size}.pth'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    else:
        print(f"Weights not found at {weight_path}")
        print("Download from: https://github.com/DepthAnything/Depth-Anything-V2#pretrained-models")
        return False

    model = model.to(device).eval()

    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')) +
                         glob.glob(os.path.join(input_dir, '*.jpg')) +
                         glob.glob(os.path.join(input_dir, '*.JPG')))

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img_name = Path(img_path).stem
        print(f"  Processing {img_name}...")

        raw_img = cv2.imread(img_path)
        depth = model.infer_image(raw_img)  # (H, W) float32

        # Save as .npy
        out_path = os.path.join(output_dir, f"{img_name}.npy")
        np.save(out_path, depth)

    print(f"  Generated {len(image_paths)} depth maps")
    return True


def generate_with_boosting_monodepth(input_dir, output_dir):
    """Generate depth maps using BoostingMonocularDepth (SparseGS built-in)."""
    sparsegs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, sparsegs_root)

    try:
        from BoostingMonocularDepth.prepare_depth import prepare_gt_depth
        prepare_gt_depth(input_folder=input_dir, save_folder=output_dir)
        return True
    except Exception as e:
        print(f"BoostingMonocularDepth failed: {e}")
        return False


def convert_lidar_depth_png_to_npy(depth_maps_dir, images_dir, output_dir):
    """Convert LiDAR-projected depth PNGs (DNGaussian format) to .npy for SparseGS.

    DNGaussian depth format:
      - 8-bit grayscale PNG in disparity format (255=near, 0=far)
      - Named: depth_{image_stem}.png

    SparseGS depth format:
      - .npy float array, relative depth (higher = farther typically)
      - Named: {image_stem}.npy
      - Will be normalized to [0,1] at load time

    Since SparseGS uses Pearson correlation (scale-invariant),
    we just need to maintain the correct depth ordering.
    """
    import cv2

    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')) +
                         glob.glob(os.path.join(images_dir, '*.jpg')))

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for img_path in image_paths:
        img_stem = Path(img_path).stem
        # DNGaussian naming: depth_{image_stem}.png
        depth_png_path = os.path.join(depth_maps_dir, f"depth_{img_stem}.png")

        if not os.path.exists(depth_png_path):
            print(f"  Warning: depth map not found: {depth_png_path}")
            continue

        # Read 8-bit depth PNG
        depth_png = cv2.imread(depth_png_path, cv2.IMREAD_GRAYSCALE)
        if depth_png is None:
            print(f"  Warning: failed to read {depth_png_path}")
            continue

        # Convert disparity (255=near, 0=far) to depth (0=near, 255=far)
        # SparseGS Pearson loss just needs correct relative ordering
        depth_float = 255.0 - depth_png.astype(np.float32)

        # Save as .npy
        out_path = os.path.join(output_dir, f"{img_stem}.npy")
        np.save(out_path, depth_float)
        count += 1
        print(f"  {img_stem}: {depth_png.shape} PNG -> .npy")

    print(f"  Converted {count} depth maps")
    return count > 0


def generate_simple_depth_from_distance(cameras_info_path, images_dir, output_dir):
    """Generate simple depth maps by computing distance-based depth proxy.

    This is a fallback when no depth estimation model is available.
    For each pixel, we compute a rough depth based on the camera geometry.
    This won't be very accurate but gives SparseGS something to work with.
    """
    print("  Generating placeholder depth (constant depth maps)...")
    print("  Warning: This is a poor substitute. Use Depth Anything for better results.")

    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')) +
                         glob.glob(os.path.join(images_dir, '*.jpg')))

    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img_stem = Path(img_path).stem
        # Create a simple gradient depth map as placeholder
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Simple vertical gradient (top=far, bottom=near) as very rough depth prior
        depth = np.linspace(1.0, 0.0, h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)

        out_path = os.path.join(output_dir, f"{img_stem}.npy")
        np.save(out_path, depth)

    print(f"  Generated {len(image_paths)} placeholder depth maps")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for SparseGS")
    parser.add_argument('--input', type=str, required=True,
                        help='Input images directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output depths directory')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'depth_anything', 'boosting', 'lidar_convert', 'placeholder'],
                        help='Depth estimation method')
    parser.add_argument('--model_size', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Depth Anything model size')
    parser.add_argument('--depth_maps_dir', type=str, default=None,
                        help='LiDAR depth PNGs directory (for lidar_convert method)')

    args = parser.parse_args()

    print(f"\nGenerating depth maps:")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Method: {args.method}")

    success = False

    if args.method == 'auto':
        # Try methods in order of preference
        print("\n[Trying] Depth Anything V2...")
        success = generate_with_depth_anything_v2(args.input, args.output, args.model_size)

        if not success:
            print("\n[Trying] BoostingMonocularDepth...")
            success = generate_with_boosting_monodepth(args.input, args.output)

        if not success:
            # Check if LiDAR depth maps exist
            base_dir = os.path.dirname(args.input)
            lidar_dir = os.path.join(base_dir, 'depth_maps')
            if os.path.exists(lidar_dir):
                print(f"\n[Trying] Converting LiDAR depth from {lidar_dir}...")
                success = convert_lidar_depth_png_to_npy(lidar_dir, args.input, args.output)

        if not success:
            print("\n[Fallback] Generating placeholder depth maps...")
            success = generate_simple_depth_from_distance(None, args.input, args.output)

    elif args.method == 'depth_anything':
        success = generate_with_depth_anything_v2(args.input, args.output, args.model_size)

    elif args.method == 'boosting':
        success = generate_with_boosting_monodepth(args.input, args.output)

    elif args.method == 'lidar_convert':
        if args.depth_maps_dir is None:
            base_dir = os.path.dirname(args.input)
            args.depth_maps_dir = os.path.join(base_dir, 'depth_maps')
        success = convert_lidar_depth_png_to_npy(args.depth_maps_dir, args.input, args.output)

    elif args.method == 'placeholder':
        success = generate_simple_depth_from_distance(None, args.input, args.output)

    if success:
        print("\nDepth generation complete!")
        # Verify output
        npy_files = glob.glob(os.path.join(args.output, '*.npy'))
        print(f"  Generated {len(npy_files)} .npy files in {args.output}")
    else:
        print("\nDepth generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
