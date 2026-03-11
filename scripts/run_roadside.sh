#!/bin/bash
# =============================================================================
# SparseGS Training Script for Roadside Dataset (Car-Road Cooperative)
#
# Usage:
#   # Step 1: Prepare data (from raw dataset)
#   python scripts/prepare_roadside_for_sparsegs.py \
#     --raw_dataset /mnt/car_road_data_TianJin \
#     --scene 053 --timestamp 1743583131842 \
#     --output data/car_road/scene053
#
#   # Step 1 (alternative): Prepare data (from self-extracted)
#   python scripts/prepare_roadside_for_sparsegs.py \
#     --self_dataset /path/to/self_Dataset \
#     --timestamp 1743583131842 \
#     --output data/car_road/scene053
#
#   # Step 2: Generate depth maps
#   python scripts/generate_depth.py \
#     --input data/car_road/scene053/images \
#     --output data/car_road/scene053/depths \
#     --method auto
#
#   # Step 3: Train
#   bash scripts/run_roadside.sh
# =============================================================================

set -e

# ======== Configuration ========
SCENE_DIR="data/car_road/scene053"
OUTPUT_DIR="output/car_road/scene053_sparsegs"
ITERATIONS=30000

# ======== SparseGS Parameters ========
# These are tuned for a roadside scene with:
#   - 4 cameras, 1 frame (extreme few-shot)
#   - Large scene scale (~130m camera baseline)
#   - LiDAR point cloud initialization

python3 train.py \
  --source_path ${SCENE_DIR} \
  --model_path ${OUTPUT_DIR} \
  -r 1 \
  --iterations ${ITERATIONS} \
  --lambda_dssim 0.2 \
  --beta 5.0 \
  --lambda_pearson 0.05 \
  --lambda_local_pearson 0.15 \
  --box_p 128 \
  --p_corr 0.5 \
  --lambda_reg 0.0 \
  --lambda_diffusion 0.0 \
  --prune_sched 20000 \
  --save_iterations 7000 ${ITERATIONS} \
  --test_iterations 7000 ${ITERATIONS} \
  --densify_from_iter 500 \
  --densify_until_iter 18000 \
  --densify_grad_threshold 0.0002 \
  --opacity_reset_interval 3000 \
  --percent_dense 0.01

echo ""
echo "Training complete!"
echo "Results: ${OUTPUT_DIR}"
echo "Point cloud: ${OUTPUT_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
