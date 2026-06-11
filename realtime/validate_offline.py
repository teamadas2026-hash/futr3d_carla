"""
Offline validation of the real-time rig BEFORE wiring CARLA.

Takes real .pkl entries, runs each through the SAME pipeline + model + lidar2img
that the live loop uses, and overlays predicted 3D boxes on the sample's own
camera images. If boxes land on the right vehicles, then model, pipeline,
lidar2img, and camera order are all proven correct — and anything that breaks
later is purely a live-data-format issue.

Run from the futr3d root:
    python realtime/validate_offline.py
"""

import os
import sys

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate

# --- make project + realtime importable when run from futr3d root ------------
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("realtime"))

# your working runtime builder (adjust module name if your file differs)
from realtime_detect import build_runtime

# your overlay function (adjust path to wherever visualize_results.py lives)
try:
    from tools.misc.visualize_results import project_boxes_to_image
except ImportError:
    sys.path.insert(0, os.path.abspath("tools"))
    from visualize_results import project_boxes_to_image

from mmdet3d.core.bbox import get_box_type

def add_pipeline_fields(input_dict, box_type_3d="LiDAR"):
    """Replicate Custom3DDataset.pre_pipeline so the test-time transforms
    (GlobalRotScaleTrans, RandomFlip3D, DefaultFormatBundle3D) find their keys."""
    box_type, box_mode = get_box_type(box_type_3d)
    input_dict["box_type_3d"] = box_type
    input_dict["box_mode_3d"] = box_mode
    for k in ("img_fields", "bbox3d_fields", "pts_mask_fields",
              "pts_seg_fields", "bbox_fields", "mask_fields", "seg_fields"):
        input_dict[k] = []
    return input_dict
# =============================================================================
# CONFIG — point these at your files
# =============================================================================
CONFIG     = "plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py"
CHECKPOINT = "work_dirs/lidar_0075v_cam_vov/epoch_12.pth"
PKL        = "data/nuscenes/rgb2/nuscenes_infos_val.pkl"   # use the VAL pkl
SAMPLE_IDS = [0, 1, 2, 3, 4]      # which samples to render
SCORE_THR  = 0.2                  # lower this if you see nothing
OUT_DIR    = "realtime/validation_out"


# =============================================================================
# Build an input_dict from a REAL pkl entry (its own paths + own calibration)
# =============================================================================
def infer_from_info(rt, info):
    cam_order = list(info["cams"].keys())
    image_paths, lidar2img = [], []
    for cam in cam_order:
        ci = info["cams"][cam]
        image_paths.append(ci["data_path"])
        # exact lidar2img formula from your visualize_results.py
        l2c_r = np.linalg.inv(ci["sensor2lidar_rotation"])
        l2c_t = ci["sensor2lidar_translation"] @ l2c_r.T
        l2c_rt = np.eye(4)
        l2c_rt[:3, :3] = l2c_r.T
        l2c_rt[3, :3] = -l2c_t
        viewpad = np.eye(4)
        viewpad[:3, :3] = ci["cam_intrinsic"]
        lidar2img.append(viewpad @ l2c_rt.T)

    input_dict = dict(
        sample_idx=info["token"],
        pts_filename=info["lidar_path"],
        sweeps=info.get("sweeps", []),
        timestamp=info["timestamp"] / 1e6,
        img_filename=image_paths,
        lidar2img=lidar2img,
    )
    add_pipeline_fields(input_dict, rt["cfg"].data.test.get("box_type_3d", "LiDAR"))

    data = rt["pipeline"](input_dict)
    data = collate([data], samples_per_gpu=1)
    with torch.no_grad():
        result = rt["model"](return_loss=False, rescale=True, **data)
    res = result[0].get("pts_bbox", result[0])
    return (res["boxes_3d"], res["scores_3d"], res["labels_3d"],
            lidar2img, image_paths, cam_order)


def tile_2x3(imgs, h=320):
    resized = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]
    w = max(im.shape[1] for im in resized)
    padded = [np.pad(im, ((0, 0), (0, w - im.shape[1]), (0, 0))) for im in resized]
    while len(padded) < 6:
        padded.append(np.zeros_like(padded[0]))
    return np.concatenate([np.concatenate(padded[0:3], axis=1),
                           np.concatenate(padded[3:6], axis=1)], axis=0)


def main():
    rt = build_runtime(CONFIG, CHECKPOINT, PKL)
    class_names = list(rt["classes"])
    data_infos = mmcv.load(PKL)
    data_infos = data_infos["infos"] if isinstance(data_infos, dict) else data_infos
    os.makedirs(OUT_DIR, exist_ok=True)

    for i in SAMPLE_IDS:
        info = data_infos[i]
        boxes, scores, labels, lidar2img, image_paths, cam_order = infer_from_info(rt, info)
        n_kept = int((scores >= SCORE_THR).sum())
        print(f"sample {i}: {len(scores)} raw dets, {n_kept} above thr={SCORE_THR}")

        tiles = []
        for j, cam in enumerate(cam_order):
            img = cv2.imread(image_paths[j])
            if img is None:
                print(f"  !! could not read {image_paths[j]}")
                continue
            vis = project_boxes_to_image(boxes, scores, labels, lidar2img[j],
                                         img, class_names, score_thr=SCORE_THR)
            cv2.putText(vis, cam, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(vis)

        if tiles:
            out = os.path.join(OUT_DIR, f"sample_{i:03d}_pred.png")
            cv2.imwrite(out, tile_2x3(tiles))
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
