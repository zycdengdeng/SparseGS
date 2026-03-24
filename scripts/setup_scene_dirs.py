#!/usr/bin/env python3
"""
Create all 18 × 3 = 54 scene directories under data/car_road/.

Naming: scene{NNN}_{near|middle|far}
  e.g., scene088_near, scene088_middle, scene088_far

Each directory will contain the subdirectories expected by SparseGS:
  images/  depths/  sparse/0/

Usage:
  python scripts/setup_scene_dirs.py
  python scripts/setup_scene_dirs.py --data_root /mnt/zyc_wzh/SparseGS/data/car_road
  python scripts/setup_scene_dirs.py --dry_run
"""

import argparse
import os

# 18 clips (scene numbers extracted from clip names)
CLIPS = [
    # ---- 王子晗 (5 clips) ----
    {"clip": "088_car0402_road0402_t70", "scene": "088"},
    {"clip": "086_car0402_road0402_t68", "scene": "086"},
    {"clip": "085_car0402_road0402_t67", "scene": "085"},
    {"clip": "082_car0402_road0402_t64", "scene": "082"},
    {"clip": "076_car0402_road0402_t54", "scene": "076"},
    # ---- 周路豪 (6 clips) ----
    {"clip": "003_car0325_road0327_t3",  "scene": "003"},
    {"clip": "004_car0325_road0327_t4",  "scene": "004"},
    {"clip": "009_car0325_road0327_t10", "scene": "009"},
    {"clip": "015_car0325_road0327_t18", "scene": "015"},
    {"clip": "020_car0325_road0327_t25", "scene": "020"},
    {"clip": "031_car0402_road0402_t9",  "scene": "031"},
    # ---- 毛雨农 (7 clips) ----
    {"clip": "035_car0402_road0402_t13", "scene": "035"},
    {"clip": "039_car0402_road0402_t17", "scene": "039"},
    {"clip": "050_car0402_road0402_t28", "scene": "050"},
    {"clip": "055_car0402_road0402_t33", "scene": "055"},
    {"clip": "056_car0402_road0402_t34", "scene": "056"},
    {"clip": "063_car0402_road0402_t41", "scene": "063"},
    {"clip": "059_car0402_road0402_t37", "scene": "059"},
]

POSITIONS = ["near", "middle", "far"]


def main():
    parser = argparse.ArgumentParser(description="Create 54 scene directories for roadside 3DGS")
    parser.add_argument("--data_root", type=str, default="data/car_road",
                        help="Root directory for scene data")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print directories without creating them")
    args = parser.parse_args()

    total = len(CLIPS) * len(POSITIONS)
    print(f"Creating {total} scene directories under {args.data_root}/\n")

    created = 0
    existed = 0

    for clip in CLIPS:
        scene_num = clip["scene"]
        for pos in POSITIONS:
            dir_name = f"scene{scene_num}_{pos}"
            scene_path = os.path.join(args.data_root, dir_name)

            if args.dry_run:
                print(f"  [DRY] {dir_name}/")
                print(f"         ├── images/")
                print(f"         ├── depths/")
                print(f"         └── sparse/0/")
                continue

            if os.path.exists(scene_path):
                existed += 1
                print(f"  [EXISTS] {dir_name}/")
            else:
                os.makedirs(scene_path, exist_ok=True)
                created += 1
                print(f"  [CREATE] {dir_name}/")

            # Create subdirectories
            for sub in ["images", "depths", os.path.join("sparse", "0")]:
                os.makedirs(os.path.join(scene_path, sub), exist_ok=True)

    if not args.dry_run:
        print(f"\nDone: {created} created, {existed} already existed, {total} total")
    else:
        print(f"\n[DRY RUN] Would create {total} directories")

    # Print summary table
    print(f"\n{'Scene':<12} {'Clip':<35} {'Directories'}")
    print("-" * 80)
    for clip in CLIPS:
        s = clip["scene"]
        dirs = ", ".join(f"scene{s}_{p}" for p in POSITIONS)
        print(f"  {s:<10} {clip['clip']:<35} {dirs}")


if __name__ == "__main__":
    main()
