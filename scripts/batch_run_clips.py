#!/usr/bin/env python3
"""
Batch pipeline for roadside 3DGS reconstruction and ego-vehicle rendering.

Runs the full SparseGS pipeline for 18 clips × 3 timestamps (近路/路中/路远):
  1. Data preparation (prepare_roadside_for_sparsegs.py)
  2. Depth estimation (generate_depth.py)
  3. Training (train.py)
  4. Ego-vehicle rendering (render from vehicle position using road labels)

Usage:
  # Run all clips
  python scripts/batch_run_clips.py --raw_dataset /mnt/car_road_data_TianJin

  # Run specific clip(s)
  python scripts/batch_run_clips.py --raw_dataset /mnt/car_road_data_TianJin \
    --clip_filter 088_car0402_road0402_t70

  # Skip training (only prepare + depth)
  python scripts/batch_run_clips.py --raw_dataset /mnt/car_road_data_TianJin \
    --stages prepare depth

  # Parallel training: 6 jobs concurrently on one GPU (~80GB VRAM)
  python scripts/batch_run_clips.py --raw_dataset /mnt/car_road_data_TianJin \
    --stages train --parallel 6

  # Dry run (print commands without executing)
  python scripts/batch_run_clips.py --raw_dataset /mnt/car_road_data_TianJin --dry_run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ==============================================================================
# Clip definitions: 18 clips × 3 timestamps (近路, 路中, 路远)
#
# Format: cam{camera_id}_{timestamp}
# The camera_id indicates which roadside camera best views the vehicle.
# The timestamp is the roadside frame timestamp.
# ==============================================================================

CLIPS = [
    # ---- 王子晗 (5 clips) ----
    {
        "clip": "088_car0402_road0402_t70",
        "near": "cam3_1743652835642",
        "mid":  "cam3_1743652838934",
        "far":  "cam3_1743652843091",
    },
    {
        "clip": "086_car0402_road0402_t68",
        "near": "cam3_1743652311985",
        "mid":  "cam3_1743652315384",
        "far":  "cam9_1743652319536",
    },
    {
        "clip": "085_car0402_road0402_t67",
        "near": "cam9_1743651916518",
        "mid":  "cam9_1743651920099",
        "far":  "cam3_1743651923810",
    },
    {
        "clip": "082_car0402_road0402_t64",
        "near": "cam3_1743650233549",
        "mid":  "cam3_1743650237382",
        "far":  "cam9_1743650240522",
    },
    {
        "clip": "076_car0402_road0402_t54",
        "near": "cam6_1743646656417",
        "mid":  "cam6_1743646658638",
        "far":  "cam6_1743646661123",
    },

    # ---- 周路豪 (6 clips) ----
    {
        "clip": "003_car0325_road0327_t3",
        "near": "cam3_1742877822545",
        "mid":  "cam3_1742877826207",
        "far":  "cam3_1742877830287",
    },
    {
        "clip": "004_car0325_road0327_t4",
        "near": "cam9_1742878212980",
        "mid":  "cam9_1742878217631",
        "far":  "cam9_1742878220772",
    },
    {
        "clip": "009_car0325_road0327_t10",
        "near": "cam9_1742880304953",
        "mid":  "cam9_1742880308674",
        "far":  "cam9_1742880311421",
    },
    {
        "clip": "015_car0325_road0327_t18",
        "near": "cam6_1742884382100",
        "mid":  "cam6_1742884384678",
        "far":  "cam6_1742884386580",
    },
    {
        "clip": "020_car0325_road0327_t25",
        "near": "cam0_1742887514589",
        "mid":  "cam0_1742887517308",
        "far":  "cam0_1742887519629",
    },
    {
        "clip": "031_car0402_road0402_t9",
        "near": "cam6_1743572672542",
        "mid":  "cam6_1743572675860",
        "far":  "cam6_1743572679422",
    },

    # ---- 毛雨农 (7 clips) ----
    {
        "clip": "035_car0402_road0402_t13",
        "near": "cam9_1743574755822",
        "mid":  "cam9_1743574759909",
        "far":  "cam9_1743574763720",
    },
    {
        "clip": "039_car0402_road0402_t17",
        "near": "cam9_1743576182579",
        "mid":  "cam9_1743576186959",
        "far":  "cam9_1743576190790",
    },
    {
        "clip": "050_car0402_road0402_t28",
        "near": "cam6_1743582087079",
        "mid":  "cam6_1743582090530",
        "far":  "cam6_1743582093290",
    },
    {
        "clip": "055_car0402_road0402_t33",
        "near": "cam0_1743583783950",
        "mid":  "cam0_1743583787671",
        "far":  "cam0_1743583791002",
    },
    {
        "clip": "056_car0402_road0402_t34",
        "near": "cam6_1743584168546",
        "mid":  "cam6_1743584171142",
        "far":  "cam6_1743584173593",
    },
    {
        "clip": "063_car0402_road0402_t41",
        "near": "cam0_1743587038293",
        "mid":  "cam0_1743587040355",
        "far":  "cam0_1743587042705",
    },
    {
        "clip": "059_car0402_road0402_t37",
        "near": "cam0_1743584970206",
        "mid":  "cam0_1743584972552",
        "far":  "cam0_1743584975161",
    },
]

# Ego vehicle ID for each clip (from carid_match_report.txt)
# Format: clip_name -> nearsetCarID (label + id in road_labels JSON)
CLIP_CAR_IDS = {
    "088_car0402_road0402_t70": "Suv19",
    "086_car0402_road0402_t68": "Suv23",
    "085_car0402_road0402_t67": "Suv47",
    "082_car0402_road0402_t64": "Suv31",
    "076_car0402_road0402_t54": "Suv12",
    "003_car0325_road0327_t3":  "Suv45",
    "004_car0325_road0327_t4":  "Suv17",
    "009_car0325_road0327_t10": "Suv35",
    "015_car0325_road0327_t18": "Suv55",
    "020_car0325_road0327_t25": "Suv77",
    "031_car0402_road0402_t9":  "Suv41",
    "035_car0402_road0402_t13": "Suv86",
    "039_car0402_road0402_t17": "Suv81",
    "050_car0402_road0402_t28": "Suv78",
    "055_car0402_road0402_t33": "Suv13",
    "056_car0402_road0402_t34": "Suv25",
    "063_car0402_road0402_t41": "Suv75",
    "059_car0402_road0402_t37": "Suv29",
}

# SparseGS training parameters (from run_roadside.sh, tuned for roadside)
TRAIN_PARAMS = {
    "iterations": 30000,
    "lambda_dssim": 0.2,
    "beta": 5.0,
    "lambda_pearson": 0.05,
    "lambda_local_pearson": 0.15,
    "box_p": 128,
    "p_corr": 0.5,
    "lambda_reg": 0.0,
    "lambda_diffusion": 0.0,
    "prune_sched": 20000,
    "save_iterations": "7000 30000",
    "test_iterations": "7000 30000",
    "densify_from_iter": 500,
    "densify_until_iter": 18000,
    "densify_grad_threshold": 0.0002,
    "opacity_reset_interval": 3000,
    "percent_dense": 0.01,
}


def parse_cam_timestamp(cam_ts_str):
    """Parse 'cam3_1743652835642' -> (camera_id='3', timestamp='1743652835642')."""
    match = re.match(r"cam(\d+)_(\d+)", cam_ts_str)
    if not match:
        raise ValueError(f"Invalid cam_timestamp format: {cam_ts_str}")
    return match.group(1), match.group(2)


def get_scene_number(clip_name):
    """Extract scene number from clip name: '088_car0402_road0402_t70' -> '088'."""
    return clip_name.split("_")[0]


def parse_car_id(car_id_str):
    """Parse 'Suv19' -> (label='Suv', id=19)."""
    match = re.match(r"([A-Za-z_]+)(\d+)", car_id_str)
    if not match:
        raise ValueError(f"Cannot parse car ID: {car_id_str}")
    return match.group(1), int(match.group(2))


def find_ego_vehicle_in_labels(label_json_path, car_id_str):
    """Find ego vehicle object in road_labels JSON by nearsetCarID.

    Args:
        label_json_path: Path to interpolation_labels/{timestamp}.json
        car_id_str: e.g., 'Suv19' meaning label='Suv', id=19

    Returns:
        dict with keys: x, y, z, yaw, length, width, height (in road/LiDAR frame)
        None if not found
    """
    label_name, label_id = parse_car_id(car_id_str)

    with open(label_json_path, 'r') as f:
        data = json.load(f)

    for obj in data.get("object", []):
        if obj["id"] == label_id and obj["label"] == label_name:
            return {
                "x": obj["x"],
                "y": obj["y"],
                "z": obj["z"],
                "yaw": obj["yaw"],
                "length": obj["length"],
                "width": obj["width"],
                "height": obj["height"],
            }

    return None


def find_closest_label_json(labels_dir, target_timestamp):
    """Find the label JSON with closest timestamp to target.

    Args:
        labels_dir: Path to road_labels/interpolation_labels/
        target_timestamp: Target timestamp string (e.g., '1743652835642')

    Returns:
        Path to closest JSON file, or None
    """
    target_ts = int(target_timestamp)
    best_path = None
    best_diff = float('inf')

    if not os.path.isdir(labels_dir):
        return None

    for fname in os.listdir(labels_dir):
        if not fname.endswith('.json'):
            continue
        try:
            ts = int(fname.replace('.json', ''))
        except ValueError:
            continue
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_path = os.path.join(labels_dir, fname)

    if best_path:
        print(f"    Label JSON: {os.path.basename(best_path)} (diff: {best_diff}ms)")
    return best_path


def run_cmd(cmd, dry_run=False, cwd=None, live_output=True):
    """Run a shell command, print it, and check return code."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n  $ {cmd_str}")
    if dry_run:
        return True

    if live_output:
        # Stream output directly so user can see tqdm progress bars
        result = subprocess.run(
            cmd_str, shell=True, cwd=cwd,
        )
    else:
        result = subprocess.run(
            cmd_str, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    if result.returncode != 0:
        print(f"  [FAILED] Return code: {result.returncode}")
        if not live_output and result.stdout:
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return False
    return True


# ---------------------------------------------------------------------------
# Parallel training support
# ---------------------------------------------------------------------------

def _parse_iteration_from_log(log_path):
    """Parse latest iteration from a training log file.

    Looks for patterns like '[ITER 1234]' or 'Training progress:  42%|' or
    tqdm-style output.
    """
    if not os.path.exists(log_path):
        return 0, 0  # current_iter, total_iter

    try:
        # Read last 4KB of log (sufficient for recent progress)
        with open(log_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode('utf-8', errors='replace')
    except OSError:
        return 0, 0

    current = 0
    total = 30000  # default

    # Match "[ITER 1234]"
    for m in re.finditer(r'\[ITER\s+(\d+)\]', tail):
        current = max(current, int(m.group(1)))

    # Match tqdm-style "  42%|" or iteration count in postfix
    # tqdm prints like: " 42%|████      | 12600/30000 [..."
    for m in re.finditer(r'(\d+)/(\d+)\s*\[', tail):
        iter_cur, iter_tot = int(m.group(1)), int(m.group(2))
        if iter_cur > current:
            current = iter_cur
            total = iter_tot

    # Match percentage " 42%|"
    for m in re.finditer(r'\s+(\d+)%\|', tail):
        pct = int(m.group(1))
        estimated = int(total * pct / 100)
        if estimated > current:
            current = estimated

    return current, total


def _make_progress_bar(current, total, width=30):
    """Render a text progress bar."""
    if total <= 0:
        return "[waiting...]"
    frac = min(current / total, 1.0)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    return f"|{bar}| {current}/{total} ({pct:.0f}%)"


def _print_parallel_dashboard(jobs, log_dir):
    """Print a live dashboard of parallel training jobs."""
    lines = []
    for task_id, info in jobs.items():
        status = info.get("status", "pending")
        log_path = os.path.join(log_dir, f"{task_id.replace('/', '_')}.log")

        if status == "done":
            lines.append(f"  ✓ {task_id:<45s} DONE")
        elif status == "failed":
            lines.append(f"  ✗ {task_id:<45s} FAILED")
        elif status == "running":
            cur, tot = _parse_iteration_from_log(log_path)
            bar = _make_progress_bar(cur, tot)
            lines.append(f"  ▶ {task_id:<45s} {bar}")
        else:
            lines.append(f"  · {task_id:<45s} queued")

    # Count stats
    n_done = sum(1 for v in jobs.values() if v["status"] == "done")
    n_fail = sum(1 for v in jobs.values() if v["status"] == "failed")
    n_run = sum(1 for v in jobs.values() if v["status"] == "running")
    n_total = len(jobs)

    header = f"  Parallel Training: {n_done} done, {n_fail} failed, {n_run} running, {n_total} total"

    # Use carriage return / ANSI escape to overwrite
    # Move cursor up by len(lines)+2 lines
    n_lines = len(lines) + 2
    sys.stdout.write(f"\033[{n_lines}A\033[J")  # move up and clear
    print(header)
    print("  " + "-" * 68)
    for line in lines:
        print(line)
    sys.stdout.flush()


def _run_train_job(task_id, cmd_str, log_path, jobs_dict, lock):
    """Run a single training command, writing output to log file."""
    with lock:
        jobs_dict[task_id]["status"] = "running"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(
            cmd_str, shell=True,
            stdout=logf, stderr=subprocess.STDOUT,
        )
        proc.wait()

    with lock:
        if proc.returncode == 0:
            jobs_dict[task_id]["status"] = "done"
        else:
            jobs_dict[task_id]["status"] = "failed"

    return proc.returncode == 0


def run_parallel_training(task_list, parallel, gpu_id, dry_run=False):
    """Run multiple training tasks in parallel with a live dashboard.

    Args:
        task_list: list of dicts with keys: task_id, source_dir, model_dir
        parallel: max concurrent jobs
        gpu_id: GPU device ID
        dry_run: if True, only print commands

    Returns:
        (success_list, failed_list)
    """
    p = TRAIN_PARAMS
    log_dir = "logs/parallel_train"
    os.makedirs(log_dir, exist_ok=True)

    # Build commands
    jobs = OrderedDict()
    cmds = {}
    for task in task_list:
        tid = task["task_id"]
        jobs[tid] = {"status": "pending"}
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python3 train.py"
            f" --source_path {task['source_dir']}"
            f" --model_path {task['model_dir']}"
            f" -r 1"
            f" --iterations {p['iterations']}"
            f" --lambda_dssim {p['lambda_dssim']}"
            f" --beta {p['beta']}"
            f" --lambda_pearson {p['lambda_pearson']}"
            f" --lambda_local_pearson {p['lambda_local_pearson']}"
            f" --box_p {p['box_p']}"
            f" --p_corr {p['p_corr']}"
            f" --lambda_reg {p['lambda_reg']}"
            f" --lambda_diffusion {p['lambda_diffusion']}"
            f" --prune_sched {p['prune_sched']}"
            f" --save_iterations {p['save_iterations']}"
            f" --test_iterations {p['test_iterations']}"
            f" --densify_from_iter {p['densify_from_iter']}"
            f" --densify_until_iter {p['densify_until_iter']}"
            f" --densify_grad_threshold {p['densify_grad_threshold']}"
            f" --opacity_reset_interval {p['opacity_reset_interval']}"
            f" --percent_dense {p['percent_dense']}"
        )
        cmds[tid] = cmd

    if dry_run:
        print(f"\n  [DRY RUN] Would run {len(task_list)} jobs, {parallel} at a time:")
        for tid, cmd in cmds.items():
            print(f"\n  $ {cmd}")
            log_path = os.path.join(log_dir, f"{tid.replace('/', '_')}.log")
            print(f"    -> log: {log_path}")
        return list(cmds.keys()), []

    # Print initial dashboard placeholder
    print(f"\n  Parallel training: {len(task_list)} jobs, {parallel} concurrent on GPU {gpu_id}")
    print(f"  Logs: {log_dir}/")
    print()
    # Print placeholder lines for dashboard
    for _ in range(len(jobs) + 2):
        print()

    lock = threading.Lock()
    success_list = []
    failed_list = []

    def _worker(tid):
        log_path = os.path.join(log_dir, f"{tid.replace('/', '_')}.log")
        return tid, _run_train_job(tid, cmds[tid], log_path, jobs, lock)

    # Dashboard updater thread
    stop_event = threading.Event()

    def _dashboard_loop():
        while not stop_event.is_set():
            try:
                _print_parallel_dashboard(jobs, log_dir)
            except Exception:
                pass
            stop_event.wait(timeout=5)

    dash_thread = threading.Thread(target=_dashboard_loop, daemon=True)
    dash_thread.start()

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(_worker, tid): tid for tid in cmds}
        for future in as_completed(futures):
            tid, ok = future.result()
            if ok:
                success_list.append(tid)
            else:
                failed_list.append(tid)

    stop_event.set()
    dash_thread.join(timeout=2)

    # Final dashboard
    _print_parallel_dashboard(jobs, log_dir)
    print()

    return success_list, failed_list


def stage_prepare(clip_name, timestamp, raw_dataset, output_dir, dry_run=False):
    """Stage 1: Prepare roadside data for SparseGS."""
    scene_num = get_scene_number(clip_name)
    cmd = (
        f"python3 scripts/prepare_roadside_for_sparsegs.py"
        f" --raw_dataset {raw_dataset}"
        f" --scene {scene_num}"
        f" --timestamp {timestamp}"
        f" --output {output_dir}"
    )
    return run_cmd(cmd, dry_run=dry_run)


def stage_depth(output_dir, depth_method="auto", dry_run=False):
    """Stage 2: Generate monocular depth maps."""
    images_dir = os.path.join(output_dir, "images")
    depths_dir = os.path.join(output_dir, "depths")
    cmd = (
        f"python3 scripts/generate_depth.py"
        f" --input {images_dir}"
        f" --output {depths_dir}"
        f" --method {depth_method}"
    )
    return run_cmd(cmd, dry_run=dry_run)


def stage_train(source_dir, model_dir, gpu_id=0, dry_run=False):
    """Stage 3: Train SparseGS model."""
    p = TRAIN_PARAMS
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python3 train.py"
        f" --source_path {source_dir}"
        f" --model_path {model_dir}"
        f" -r 1"
        f" --iterations {p['iterations']}"
        f" --lambda_dssim {p['lambda_dssim']}"
        f" --beta {p['beta']}"
        f" --lambda_pearson {p['lambda_pearson']}"
        f" --lambda_local_pearson {p['lambda_local_pearson']}"
        f" --box_p {p['box_p']}"
        f" --p_corr {p['p_corr']}"
        f" --lambda_reg {p['lambda_reg']}"
        f" --lambda_diffusion {p['lambda_diffusion']}"
        f" --prune_sched {p['prune_sched']}"
        f" --save_iterations {p['save_iterations']}"
        f" --test_iterations {p['test_iterations']}"
        f" --densify_from_iter {p['densify_from_iter']}"
        f" --densify_until_iter {p['densify_until_iter']}"
        f" --densify_grad_threshold {p['densify_grad_threshold']}"
        f" --opacity_reset_interval {p['opacity_reset_interval']}"
        f" --percent_dense {p['percent_dense']}"
    )
    return run_cmd(cmd, dry_run=dry_run)


def stage_render(model_dir, clip_name, timestamp, position, raw_dataset,
                 render_output_dir, gpu_id=0, dry_run=False):
    """Stage 4: Render ego-vehicle viewpoint.

    Locates the ego vehicle in road_labels JSON and renders from its position.
    The vehicle's (x, y, z, yaw) in road/LiDAR coords is used to construct
    a virtual camera.
    """
    car_id = CLIP_CAR_IDS.get(clip_name)
    if not car_id:
        print(f"    [SKIP] No car ID mapping for {clip_name}")
        return False

    # Find label JSON
    labels_dir = os.path.join(raw_dataset, clip_name, "road_labels", "interpolation_labels")
    label_json = find_closest_label_json(labels_dir, timestamp)
    if label_json is None:
        print(f"    [SKIP] No label JSON found in {labels_dir}")
        return False

    if dry_run:
        print(f"    Would render: clip={clip_name}, ts={timestamp}, pos={position}, car={car_id}")
        print(f"    Label JSON: {label_json}")
        return True

    # Find ego vehicle in labels
    vehicle = find_ego_vehicle_in_labels(label_json, car_id)
    if vehicle is None:
        print(f"    [SKIP] Car {car_id} not found in {label_json}")
        return False

    print(f"    Ego vehicle: {car_id} at ({vehicle['x']:.1f}, {vehicle['y']:.1f}, {vehicle['z']:.1f}), yaw={vehicle['yaw']:.2f}")

    # Save vehicle info for downstream rendering
    os.makedirs(render_output_dir, exist_ok=True)
    vehicle_info = {
        "clip": clip_name,
        "position": position,
        "timestamp": timestamp,
        "car_id": car_id,
        "vehicle": vehicle,
        "model_path": model_dir,
        "label_json": label_json,
    }
    info_path = os.path.join(render_output_dir, "vehicle_info.json")
    with open(info_path, 'w') as f:
        json.dump(vehicle_info, f, indent=2)
    print(f"    Saved vehicle info: {info_path}")

    # Render using render_vehicle.py if vehicle_calib and transform_json are available
    # Otherwise, use the vehicle position from labels to create virtual cameras
    #
    # TODO: Integrate with render_vehicle.py once vehicle calibration paths are confirmed.
    #       For now, we save the vehicle pose info for manual/custom rendering.
    #
    # To render with render_vehicle.py, uncomment:
    # cmd = (
    #     f"CUDA_VISIBLE_DEVICES={gpu_id} python3 render_vehicle.py"
    #     f" --model_path {model_dir}"
    #     f" --vehicle_calib {vehicle_calib_path}"
    #     f" --transform_json {transform_json_path}"
    #     f" --timestamp {timestamp}"
    #     f" --camera_ids 1 2 3 4 5 6 7"
    #     f" --render_scale 4"
    #     f" --output_dir {render_output_dir}"
    # )
    # return run_cmd(cmd, dry_run=dry_run)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch SparseGS pipeline for roadside clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--raw_dataset", type=str, default="/mnt/car_road_data_TianJin",
                        help="Raw dataset root path")
    parser.add_argument("--data_root", type=str, default="data/car_road",
                        help="Output data root for prepared scenes")
    parser.add_argument("--output_root", type=str, default="output/car_road",
                        help="Output root for trained models")
    parser.add_argument("--stages", type=str, nargs="+",
                        default=["prepare", "depth", "train", "render"],
                        choices=["prepare", "depth", "train", "render"],
                        help="Pipeline stages to run")
    parser.add_argument("--clip_filter", type=str, nargs="+", default=None,
                        help="Only run these clip names (e.g., 088_car0402_road0402_t70)")
    parser.add_argument("--position_filter", type=str, nargs="+", default=None,
                        choices=["near", "mid", "far", "middle"],
                        help="Only run these positions (default: all three). 'middle' is alias for 'mid'.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--depth_method", type=str, default="auto",
                        choices=["auto", "depth_anything", "boosting", "placeholder"],
                        help="Depth estimation method")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue with next task if current one fails")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of training jobs to run concurrently (default: 1). "
                             "Set to 6 to run 2 scenes x 3 positions in parallel on a large GPU.")

    args = parser.parse_args()

    # Normalize "middle" -> "mid" for internal use (CLIPS dict uses "mid")
    raw_positions = args.position_filter or ["near", "mid", "far"]
    positions = ["mid" if p == "middle" else p for p in raw_positions]
    position_labels = {"near": "近路", "mid": "路中", "far": "路远"}
    # Directory name mapping: mid -> middle for clarity
    position_dir_names = {"near": "near", "mid": "middle", "far": "far"}

    # Filter clips if requested
    clips = CLIPS
    if args.clip_filter:
        clips = [c for c in CLIPS if c["clip"] in args.clip_filter]
        if not clips:
            print(f"No clips matched filter: {args.clip_filter}")
            sys.exit(1)

    # Summary
    total_tasks = len(clips) * len(positions)
    print("=" * 70)
    print(f"SparseGS Batch Pipeline")
    print(f"=" * 70)
    print(f"  Clips:      {len(clips)}")
    print(f"  Positions:  {positions}")
    print(f"  Total runs: {total_tasks}")
    print(f"  Stages:     {args.stages}")
    print(f"  Raw data:   {args.raw_dataset}")
    print(f"  Data root:  {args.data_root}")
    print(f"  Output root:{args.output_root}")
    print(f"  GPU:        {args.gpu_id}")
    if args.parallel > 1:
        print(f"  Parallel:   {args.parallel} concurrent training jobs")
    if args.dry_run:
        print(f"  *** DRY RUN - no commands will be executed ***")
    print("=" * 70)

    results = {"success": 0, "failed": 0, "skipped": 0}
    failed_tasks = []
    start_time = time.time()

    # Build the full list of (clip_info, pos, task_id, scene_dir, model_dir, ...)
    all_tasks = []
    for clip_idx, clip_info in enumerate(clips):
        clip_name = clip_info["clip"]
        scene_num = get_scene_number(clip_name)
        for pos in positions:
            cam_ts = clip_info[pos]
            cam_id, timestamp = parse_cam_timestamp(cam_ts)
            pos_dir = position_dir_names[pos]
            task_id = f"{clip_name}/{pos}"
            scene_dir = os.path.join(args.data_root, f"scene{scene_num}_{pos_dir}")
            model_dir = os.path.join(args.output_root, f"scene{scene_num}_{pos_dir}")
            render_dir = os.path.join(model_dir, f"vehicle_render_{pos_dir}")
            all_tasks.append({
                "clip_info": clip_info,
                "clip_name": clip_name,
                "scene_num": scene_num,
                "pos": pos,
                "cam_id": cam_id,
                "timestamp": timestamp,
                "task_id": task_id,
                "scene_dir": scene_dir,
                "model_dir": model_dir,
                "render_dir": render_dir,
            })

    use_parallel_train = args.parallel > 1 and "train" in args.stages

    # ---- Sequential stages: prepare, depth (always sequential) ----
    sequential_stages = [s for s in args.stages if s not in ("train",)] if use_parallel_train else args.stages
    train_candidates = []  # tasks that pass prepare+depth and need training

    for task_num, task in enumerate(all_tasks, 1):
        task_id = task["task_id"]

        print(f"\n{'=' * 70}")
        print(f"[{task_num}/{total_tasks}] {task_id}")
        print(f"  Clip:      {task['clip_name']} (scene {task['scene_num']})")
        print(f"  Position:  {position_labels[task['pos']]} ({task['pos']})")
        print(f"  Camera:    cam{task['cam_id']}")
        print(f"  Timestamp: {task['timestamp']}")
        print(f"{'=' * 70}")

        task_ok = True

        # Stage 1: Prepare
        if "prepare" in sequential_stages:
            print(f"\n  [Stage 1/4] Data Preparation")
            ok = stage_prepare(task["clip_name"], task["timestamp"], args.raw_dataset,
                               task["scene_dir"], dry_run=args.dry_run)
            if not ok:
                task_ok = False
                if not args.continue_on_error:
                    print(f"  [ABORT] Prepare failed for {task_id}")
                    failed_tasks.append(task_id)
                    results["failed"] += 1
                    continue

        # Stage 2: Depth
        if "depth" in sequential_stages and task_ok:
            print(f"\n  [Stage 2/4] Depth Estimation")
            ok = stage_depth(task["scene_dir"], args.depth_method, dry_run=args.dry_run)
            if not ok:
                task_ok = False
                if not args.continue_on_error:
                    print(f"  [ABORT] Depth failed for {task_id}")
                    failed_tasks.append(task_id)
                    results["failed"] += 1
                    continue

        # Stage 3: Train (sequential mode)
        if "train" in sequential_stages and task_ok:
            print(f"\n  [Stage 3/4] Training SparseGS")
            ok = stage_train(task["scene_dir"], task["model_dir"], args.gpu_id,
                             dry_run=args.dry_run)
            if not ok:
                task_ok = False
                if not args.continue_on_error:
                    print(f"  [ABORT] Training failed for {task_id}")
                    failed_tasks.append(task_id)
                    results["failed"] += 1
                    continue

        # Stage 4: Render (sequential mode)
        if "render" in sequential_stages and task_ok:
            print(f"\n  [Stage 4/4] Ego-Vehicle Rendering")
            ok = stage_render(task["model_dir"], task["clip_name"], task["timestamp"],
                              task["pos"], args.raw_dataset, task["render_dir"],
                              args.gpu_id, dry_run=args.dry_run)
            if not ok:
                task_ok = False

        if use_parallel_train and task_ok:
            # Collect for parallel training later
            train_candidates.append(task)
        elif not use_parallel_train:
            if task_ok:
                results["success"] += 1
                print(f"\n  [OK] {task_id} completed")
            else:
                results["failed"] += 1
                failed_tasks.append(task_id)
                print(f"\n  [FAILED] {task_id}")

    # ---- Parallel training stage ----
    if use_parallel_train and train_candidates:
        print(f"\n{'=' * 70}")
        print(f"PARALLEL TRAINING: {len(train_candidates)} jobs, {args.parallel} concurrent")
        print(f"{'=' * 70}")

        train_task_list = [
            {"task_id": t["task_id"], "source_dir": t["scene_dir"], "model_dir": t["model_dir"]}
            for t in train_candidates
        ]

        success_ids, failed_ids = run_parallel_training(
            train_task_list, args.parallel, args.gpu_id, dry_run=args.dry_run
        )

        # Now run render for successful training jobs (sequentially)
        if "render" in args.stages:
            print(f"\n{'=' * 70}")
            print(f"RENDERING: {len(success_ids)} trained scenes")
            print(f"{'=' * 70}")
            for task in train_candidates:
                if task["task_id"] in success_ids:
                    print(f"\n  [Render] {task['task_id']}")
                    ok = stage_render(task["model_dir"], task["clip_name"],
                                      task["timestamp"], task["pos"],
                                      args.raw_dataset, task["render_dir"],
                                      args.gpu_id, dry_run=args.dry_run)
                    if ok:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                        failed_tasks.append(task["task_id"] + "/render")
                else:
                    results["success"] += 1
        else:
            results["success"] += len(success_ids)

        results["failed"] += len(failed_ids)
        failed_tasks.extend(failed_ids)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"BATCH COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"  Success: {results['success']}/{total_tasks}")
    print(f"  Failed:  {results['failed']}/{total_tasks}")
    if failed_tasks:
        print(f"\n  Failed tasks:")
        for t in failed_tasks:
            print(f"    - {t}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
