# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib
import json
import os
import sys

import mmcv
import numpy as np
import torch
from mmcv import Config

from mmdet3d.datasets import build_dataset


# ── NMS radius per class (metres) — matches nuScenes eval protocol ────────────
CLASS_NMS_RADIUS = {
    'car':                   2.0,
    'truck':                 3.0,
    'bus':                   3.0,
    'trailer':               4.0,
    'construction_vehicle':  4.0,
    'motorcycle':            0.5,
    'bicycle':               0.5,
    'pedestrian':            0.5,
    'traffic_cone':          0.5,
    'barrier':               1.0,
}
DEFAULT_NMS_RADIUS = 2.0


def circle_nms(boxes_3d, scores_3d, labels_3d, class_names,
               nms_radius_map=None, max_num=83):
    """
    Per-class circular NMS in the BEV (bird's eye view) plane.

    Boxes whose BEV centres are within `radius` metres of a higher-scoring
    box of the same class are suppressed.

    Args:
        boxes_3d  : LiDARInstance3DBoxes  (N,)
        scores_3d : torch.Tensor           (N,)
        labels_3d : torch.Tensor           (N,) long
        class_names : list[str]
        nms_radius_map : dict[str, float]  class → radius in metres
        max_num : int  maximum boxes to keep overall

    Returns:
        kept boxes_3d, scores_3d, labels_3d  (all filtered)
    """
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    if nms_radius_map is None:
        nms_radius_map = CLASS_NMS_RADIUS

    if len(boxes_3d) == 0:
        return boxes_3d, scores_3d, labels_3d

    centers = boxes_3d.gravity_center[:, :2].numpy()   # (N, 2)  x,y only
    scores_np = scores_3d.numpy()
    labels_np = labels_3d.numpy()

    # sort all boxes by score descending
    order = np.argsort(-scores_np)
    centers  = centers[order]
    scores_np = scores_np[order]
    labels_np = labels_np[order]
    orig_idx  = order                                   # track original indices

    kept = []
    suppressed = np.zeros(len(order), dtype=bool)

    for i in range(len(order)):
        if suppressed[i]:
            continue
        kept.append(i)
        if len(kept) >= max_num:
            break

        cls_name = (class_names[labels_np[i]]
                    if labels_np[i] < len(class_names) else '')
        radius = nms_radius_map.get(cls_name, DEFAULT_NMS_RADIUS)

        # suppress all later boxes of the SAME class within radius
        same_cls = labels_np[i + 1:] == labels_np[i]
        dist = np.linalg.norm(centers[i + 1:] - centers[i], axis=1)
        suppressed[i + 1:] |= (same_cls & (dist < radius))

    kept_orig = orig_idx[kept]

    # rebuild tensors
    kept_t = torch.tensor(kept_orig, dtype=torch.long)
    # boxes_3d.tensor is already in internal bottom-center convention
    # (origin (0.5, 0.5, 0)); rewrapping it as gravity-center (0.5, 0.5, 0.5)
    # would sink every box by h/2. Keep the existing convention.
    boxes_kept = LiDARInstance3DBoxes(
        boxes_3d.tensor[kept_t], box_dim=boxes_3d.tensor.shape[-1],
        origin=(0.5, 0.5, 0))
    scores_kept = scores_3d[kept_t]
    labels_kept = labels_3d[kept_t]

    return boxes_kept, scores_kept, labels_kept


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle or json format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    parser.add_argument(
        '--score-thr', type=float, default=0.09,
        help='bbox score threshold (default: 0.09)')
    parser.add_argument(
        '--no-nms', action='store_true',
        help='disable post-hoc circular NMS (not recommended)')
    parser.add_argument(
        '--max-num', type=int, default=83,
        help='max detections per sample after NMS (default: 83, matches test_cfg)')
    args = parser.parse_args()
    return args


def nusc_json_to_show_format(json_data, dataset):
    """
    Convert NuScenes JSON result format → show() format.
    JSON boxes are in GLOBAL coordinates; convert back to LIDAR frame.
    """
    import pyquaternion
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    class_names = list(dataset.CLASSES)
    results = []

    for i in range(len(dataset)):
        sample_token = dataset.data_infos[i]['token']
        detections = json_data.get('results', {}).get(sample_token, [])
        info = dataset.data_infos[i]

        if not detections:
            results.append(dict(pts_bbox=dict(
                boxes_3d=LiDARInstance3DBoxes(
                    torch.zeros((0, 9)), box_dim=9, origin=(0.5, 0.5, 0.5)),
                scores_3d=torch.zeros(0),
                labels_3d=torch.zeros(0, dtype=torch.long)
            )))
            continue

        boxes, scores, labels = [], [], []
        for det in detections:
            # global → ego
            center_global = np.array(det['translation'])
            q_e2g = pyquaternion.Quaternion(info['ego2global_rotation'])
            t_e2g = np.array(info['ego2global_translation'])
            center_ego = q_e2g.inverse.rotate(center_global - t_e2g)

            # ego → lidar
            q_l2e = pyquaternion.Quaternion(info['lidar2ego_rotation'])
            t_l2e = np.array(info['lidar2ego_translation'])
            center_lidar = q_l2e.inverse.rotate(center_ego - t_l2e)

            # orientation
            q_global = pyquaternion.Quaternion(det['rotation'])
            q_lidar = q_l2e.inverse * q_e2g.inverse * q_global
            yaw = q_lidar.yaw_pitch_roll[0]

            # NuScenes size [w,l,h] → LiDAR box [l,w,h]
            w, l, h = det['size']
            v = det.get('velocity', [0, 0])

            box = [center_lidar[0], center_lidar[1], center_lidar[2],
                   l, w, h, yaw, v[0], v[1]]
            boxes.append(box)
            scores.append(det['detection_score'])

            cls = det['detection_name']
            label = class_names.index(cls) if cls in class_names else 0
            labels.append(label)

        results.append(dict(pts_bbox=dict(
            boxes_3d=LiDARInstance3DBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                box_dim=9, origin=(0.5, 0.5, 0.5)),
            scores_3d=torch.tensor(scores, dtype=torch.float32),
            labels_3d=torch.tensor(labels, dtype=torch.long)
        )))

    return results


def project_boxes_to_image(boxes_3d, scores_3d, labels_3d,
                            lidar2img, img, class_names,
                            score_thr=0.09, thickness=2):
    """Project 3D LiDAR boxes onto a camera image and draw wireframes."""
    import cv2

    palette = [
        (0,   255,   0),   # car
        (0,   200, 255),   # truck
        (0,   128, 255),   # trailer
        (0,     0, 255),   # bus
        (255, 128,   0),   # construction_vehicle
        (255,   0, 255),   # bicycle
        (128,   0, 255),   # motorcycle
        (255, 255,   0),   # pedestrian
        (0,   255, 255),   # traffic_cone
        (128, 255, 128),   # barrier
    ]

    img_out = img.copy()
    h, w = img_out.shape[:2]

    keep = scores_3d > score_thr
    if keep.sum() == 0:
        return img_out

    boxes_3d  = boxes_3d[keep]
    scores_3d = scores_3d[keep]
    labels_3d = labels_3d[keep]

    corners = boxes_3d.corners.numpy()       # (N, 8, 3)
    N = corners.shape[0]
    ones = np.ones((N, 8, 1))
    corners_h = np.concatenate([corners, ones], axis=-1)  # (N,8,4)

    pts_2d_h = (lidar2img @ corners_h.reshape(-1, 4).T).T  # (N*8, 4)
    depths    = pts_2d_h[:, 2].reshape(N, 8)
    pts_2d_h[:, :2] /= pts_2d_h[:, 2:3] + 1e-6
    pts_2d = pts_2d_h[:, :2].reshape(N, 8, 2)

    edges = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # verticals
    ]

    for n in range(N):
        if depths[n].min() < 0.1:
            continue
        pts = pts_2d[n]
        if ((pts[:,0]<0).all() or (pts[:,0]>w).all() or
                (pts[:,1]<0).all() or (pts[:,1]>h).all()):
            continue

        label = int(labels_3d[n].item())
        score = float(scores_3d[n].item())
        color = palette[label % len(palette)]
        cls_name = class_names[label] if label < len(class_names) else str(label)

        for e0, e1 in edges:
            p0 = (int(pts[e0,0]), int(pts[e0,1]))
            p1 = (int(pts[e1,0]), int(pts[e1,1]))
            cv2.line(img_out, p0, p1, color, thickness, cv2.LINE_AA)

        tx, ty = int(pts[4,0]), int(pts[4,1])
        cv2.putText(img_out, f'{cls_name} {score:.2f}', (tx, ty-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img_out


def visualize_images(results, dataset, out_dir, score_thr=0.09):
    """Project predicted boxes onto camera images and save composites."""
    import cv2

    mmcv.mkdir_or_exist(out_dir)
    class_names = list(dataset.CLASSES)

    for i, result in enumerate(mmcv.track_iter_progress(results)):
        data_info = dataset.data_infos[i]
        if 'cams' not in data_info:
            continue

        det = result.get('pts_bbox', result)
        boxes_3d  = det['boxes_3d']
        scores_3d = det['scores_3d']
        labels_3d = det['labels_3d']

        cam_imgs = []
        for cam_type, cam_info in data_info['cams'].items():
            img_path = cam_info['data_path']
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3]  = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img = viewpad @ lidar2cam_rt.T

            img_vis = project_boxes_to_image(
                boxes_3d, scores_3d, labels_3d,
                lidar2img, img, class_names, score_thr=score_thr)

            cv2.putText(img_vis, cam_type, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cam_imgs.append(img_vis)

        if not cam_imgs:
            continue

        target_h = 320
        resized = [cv2.resize(im, (int(im.shape[1]*target_h/im.shape[0]),
                                   target_h))
                   for im in cam_imgs]

        row_size = 3
        rows = []
        for r in range(0, len(resized), row_size):
            row_imgs = resized[r:r+row_size]
            max_w = max(im.shape[1] for im in row_imgs)
            padded = [np.pad(im, ((0,0),(0,max_w-im.shape[1]),(0,0)))
                      for im in row_imgs]
            while len(padded) < row_size:
                padded.append(np.zeros_like(padded[0]))
            rows.append(np.concatenate(padded, axis=1))

        composite = np.concatenate(rows, axis=0)
        file_name = os.path.splitext(
            os.path.basename(data_info['lidar_path']))[0]
        cv2.imwrite(os.path.join(out_dir, f'{file_name}_pred.png'), composite)


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle', '.json')):
        raise ValueError('The results file must be a pkl or json file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # ── Load custom plugin ────────────────────────────────────────────────
    if hasattr(cfg, 'plugin') and cfg.plugin:
        sys.path.insert(0, os.path.abspath('.'))
        module_path = cfg.plugin.strip('/').replace('/', '.')
        importlib.import_module(module_path)
    # ─────────────────────────────────────────────────────────────────────

    dataset = build_dataset(cfg.data.test)

    # Load & convert results
    if args.result.endswith('.json'):
        with open(args.result, 'r') as f:
            json_data = json.load(f)
        results = nusc_json_to_show_format(json_data, dataset)
    else:
        results = mmcv.load(args.result)

    # ── Post-hoc circular NMS ─────────────────────────────────────────────
    if not args.no_nms:
        print('\nApplying post-hoc circular NMS...')
        class_names = list(dataset.CLASSES)
        for i, result in enumerate(results):
            det = result.get('pts_bbox', result)
            # score filter first, then NMS
            keep = det['scores_3d'] >= args.score_thr
            b = det['boxes_3d'][keep] if keep.any() else det['boxes_3d'][keep]
            s = det['scores_3d'][keep]
            l = det['labels_3d'][keep]
            b, s, l = circle_nms(b, s, l, class_names,
                                  nms_radius_map=CLASS_NMS_RADIUS,
                                  max_num=args.max_num)
            det['boxes_3d']  = b
            det['scores_3d'] = s
            det['labels_3d'] = l
        print(f'NMS done. max_num={args.max_num}, score_thr={args.score_thr}')
    # ─────────────────────────────────────────────────────────────────────

    mmcv.mkdir_or_exist(args.show_dir)

    # 1. Point cloud .obj files
    if getattr(dataset, 'show', None) is not None:
        pts_dir = os.path.join(args.show_dir, 'pts')
        mmcv.mkdir_or_exist(pts_dir)
        print('\n[1/2] Saving point cloud visualizations...')
        eval_pipeline = cfg.get('eval_pipeline', {})
        if eval_pipeline:
            dataset.show(results, pts_dir, pipeline=eval_pipeline)
        else:
            dataset.show(results, pts_dir)

    # 2. Camera image overlays
    img_dir = os.path.join(args.show_dir, 'imgs')
    print(f'\n[2/2] Saving camera image overlays to {img_dir} ...')
    visualize_images(results, dataset, img_dir, score_thr=0.0)  # NMS already applied above

    print(f'\nDone. Results saved to {args.show_dir}')
    print(f'  Point clouds : {os.path.join(args.show_dir, "pts")}')
    print(f'  Camera images: {img_dir}')


if __name__ == '__main__':
    main()