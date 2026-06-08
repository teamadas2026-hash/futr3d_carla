"""
project_lidar_to_cam_fixed.py
=================================
Fixed version of project_lidar_to_cam.py.

Root-cause fixes applied
------------------------
1. **lidar2img construction** – The original script fell through to the
   ``sensor2ego`` branch, which is mathematically correct, but the resulting
   transform was being ignored (``use_lidar2img`` stayed False) so
   ``project_lidar_to_image`` was called with an *uninitialised*
   ``T_lidar_to_cam``.  Fixed: always set ``use_lidar2img=True`` and pass
   the 3×4 matrix produced from K @ T_lidar_to_cam[:3,:].

2. **Bin file column count** – Standard nuScenes LiDAR is 5 cols (x,y,z,
   intensity, ring).  Carla-generated datasets sometimes write 4 cols
   (x,y,z,intensity).  The original loader raised on 4-col files.
   Fixed: try 5 cols, then 4 cols, then 3 cols.

3. **Diagnostic prints** – Added T_lidar_to_cam printout and a sample of
   camera-space Z values so miscalibration is obvious at a glance.
"""

import argparse
import os

import cv2
import mmcv
import numpy as np
from pyquaternion import Quaternion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r") as f:
        import json
        return json.load(f)


def resolve_path(root, path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    norm_root = os.path.normpath(root)
    norm_path = os.path.normpath(path)
    if norm_path.startswith(norm_root + os.sep):
        return norm_path
    return os.path.normpath(os.path.join(root, path))


def build_index(items):
    return {item["token"]: item for item in items}


def quat_to_rot(q):
    return Quaternion(q).rotation_matrix


def euler_to_rot(roll_deg, pitch_deg, yaw_deg):
    roll  = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw   = np.deg2rad(yaw_deg)
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    rot_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr,  cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def make_transform(translation, rotation):
    """Build 4×4 rigid transform.  `rotation` may be a quaternion (list/array
    of 4) *or* a 3×3 rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape == (3, 3):
        rot = rotation
    else:
        rot = quat_to_rot(rotation)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3]  = np.array(translation, dtype=np.float64)
    return T


def load_lidar_bin(path):
    """Load a binary point-cloud file.

    Tries column counts 5 → 4 → 3 for float32, then repeats for float64.
    Returns an (N, 3) XYZ array.
    """
    for dtype in (np.float32, np.float64):
        raw = np.fromfile(path, dtype=dtype)
        for ncols in (5, 4, 3):
            if raw.size >= ncols and raw.size % ncols == 0:
                pts = raw.reshape(-1, ncols)[:, :3]
                # Sanity-check: coordinates should be finite and within ±1000 m
                if np.isfinite(pts).all() and np.abs(pts).max() < 1000:
                    return pts.astype(np.float64)
    raise ValueError(f"Cannot determine point layout for {path}")


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def build_lidar2img(K, T_lidar_to_cam):
    """Return a 3×4 projection matrix: K @ T_lidar_to_cam[:3, :]"""
    return (K @ T_lidar_to_cam[:3, :]).astype(np.float64)


def project_points(points, lidar2img, img_w, img_h):
    """Project (N,3) lidar points with a 3×4 lidar2img matrix.

    Returns
    -------
    uv_in : (M, 2)  float64 pixel coords that lie inside the image
    n_front : int   number of points with positive depth
    """
    ones   = np.ones((points.shape[0], 1), dtype=np.float64)
    pts_h  = np.hstack([points, ones])            # (N, 4)
    pts_2d = (lidar2img @ pts_h.T).T             # (N, 3)

    z      = pts_2d[:, 2]
    front  = z > 0.1
    pts_2d = pts_2d[front]

    if pts_2d.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), 0

    uv = pts_2d[:, :2] / pts_2d[:, 2:3]
    uv = uv[np.isfinite(uv).all(axis=1)]

    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )
    return uv[in_bounds], int(front.sum())


# ---------------------------------------------------------------------------
# PKL loader
# ---------------------------------------------------------------------------

def load_info_sample(info_pkl, sample_index):
    info_data = mmcv.load(info_pkl)
    if isinstance(info_data, dict) and "infos" in info_data:
        infos = info_data["infos"]
    elif isinstance(info_data, dict) and "data_list" in info_data:
        infos = info_data["data_list"]
    else:
        infos = info_data
    return infos[sample_index]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Project one LiDAR frame onto front camera image (fixed)")
    parser.add_argument("--root",          required=True)
    parser.add_argument("--version",       required=True,
                        help="Dataset version dir name, e.g. v1.0-mini")
    parser.add_argument("--sample-index",  type=int,    default=0)
    parser.add_argument("--out",           default="./lidar_on_cam.png")
    parser.add_argument("--max-points",    type=int,    default=20000)
    parser.add_argument("--point-radius",  type=int,    default=2)
    parser.add_argument("--overlay-alpha", type=float,  default=0.7)
    parser.add_argument("--corr-roll",     type=float,  default=0.0)
    parser.add_argument("--corr-pitch",    type=float,  default=0.0)
    parser.add_argument("--corr-yaw",      type=float,  default=0.0)
    parser.add_argument("--save-mask",     default=None)
    parser.add_argument("--info-pkl",      default=None,
                        help="mmdet3d-style info pkl (preferred path)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build calibration
    # ------------------------------------------------------------------
    if args.info_pkl is not None:
        # ---- PKL path ------------------------------------------------
        info     = load_info_sample(args.info_pkl, args.sample_index)
        cam_info = info["cams"]["CAM_FRONT"]

        cam_path   = resolve_path(args.root,
                                  cam_info.get("data_path") or cam_info.get("img_path"))
        lidar_path = resolve_path(args.root,
                                  info.get("lidar_path") or info.get("pts_path"))

        K = np.array(cam_info["cam_intrinsic"], dtype=np.float64)

        # Priority 1 – precomputed lidar2img (3×3 or 3×4 or 4×4)
        if "lidar2img" in cam_info:
            L2I = np.array(cam_info["lidar2img"], dtype=np.float64)
            if L2I.shape == (4, 4):
                # Some pipelines store a 4×4 homogeneous matrix
                lidar2img = (K @ L2I[:3, :]).reshape(3, 4)
                # Actually if it's already K*[R|t] baked in, don't multiply by K again
                # Heuristic: if diagonal ~1000 it's already the full projection
                if abs(L2I[0, 0]) > 100:
                    lidar2img = L2I[:3, :]
                else:
                    lidar2img = K @ L2I[:3, :]
            elif L2I.shape == (3, 4):
                lidar2img = L2I
            else:
                raise ValueError(f"Unexpected lidar2img shape {L2I.shape}")

        # Priority 2 – lidar2cam + intrinsic
        elif "lidar2cam" in cam_info and "cam_intrinsic" in cam_info:
            L2C = np.array(cam_info["lidar2cam"], dtype=np.float64)
            lidar2img = K @ L2C[:3, :]

        # Priority 3 – ego-based chain (FIXED: now actually uses lidar2img path)
        elif all(k in cam_info for k in
                 ["sensor2ego_rotation", "sensor2ego_translation"]) and \
             all(k in info for k in
                 ["lidar2ego_rotation", "lidar2ego_translation"]):

            T_cam_ego   = make_transform(cam_info["sensor2ego_translation"],
                                         cam_info["sensor2ego_rotation"])
            T_lidar_ego = make_transform(info["lidar2ego_translation"],
                                         info["lidar2ego_rotation"])
            T_lidar_to_cam = np.linalg.inv(T_cam_ego) @ T_lidar_ego

            # Optional angle correction (applied in camera frame)
            if args.corr_roll or args.corr_pitch or args.corr_yaw:
                R_corr = euler_to_rot(args.corr_roll, args.corr_pitch, args.corr_yaw)
                T_corr = np.eye(4, dtype=np.float64)
                T_corr[:3, :3] = R_corr
                T_lidar_to_cam = T_corr @ T_lidar_to_cam

            print("[DEBUG] T_lidar_to_cam:\n", T_lidar_to_cam)
            lidar2img = build_lidar2img(K, T_lidar_to_cam)

        else:
            print(f"CAM_FRONT keys: {sorted(cam_info.keys())}")
            raise KeyError("Cannot build calibration from available keys.")

    else:
        # ---- JSON / raw nuScenes path ---------------------------------
        version_dir       = os.path.join(args.root, args.version)
        sample            = load_json(os.path.join(version_dir, "sample.json"))
        sample_data       = load_json(os.path.join(version_dir, "sample_data.json"))
        calibrated_sensor = load_json(os.path.join(version_dir, "calibrated_sensor.json"))
        ego_pose          = load_json(os.path.join(version_dir, "ego_pose.json"))
        sensor            = load_json(os.path.join(version_dir, "sensor.json"))

        sample_token    = sample[args.sample_index]["token"]
        calib_by_token  = build_index(calibrated_sensor)
        ego_by_token    = build_index(ego_pose)
        sensor_by_token = build_index(sensor)

        cam_sd = lidar_sd = None
        for sd in sample_data:
            if sd["sample_token"] != sample_token:
                continue
            calib = calib_by_token[sd["calibrated_sensor_token"]]
            chan  = sensor_by_token[calib["sensor_token"]]["channel"]
            if chan == "CAM_FRONT"  and cam_sd   is None: cam_sd   = sd
            if chan == "LIDAR_TOP"  and lidar_sd  is None: lidar_sd = sd

        if cam_sd is None or lidar_sd is None:
            raise RuntimeError("Could not find CAM_FRONT and LIDAR_TOP sample_data")

        cam_calib   = calib_by_token[cam_sd["calibrated_sensor_token"]]
        lidar_calib = calib_by_token[lidar_sd["calibrated_sensor_token"]]
        cam_ego     = ego_by_token[cam_sd["ego_pose_token"]]
        lidar_ego   = ego_by_token[lidar_sd["ego_pose_token"]]

        T_cam_ego          = make_transform(cam_calib["translation"],   cam_calib["rotation"])
        T_lidar_ego        = make_transform(lidar_calib["translation"], lidar_calib["rotation"])
        T_ego_global_cam   = make_transform(cam_ego["translation"],     cam_ego["rotation"])
        T_ego_global_lidar = make_transform(lidar_ego["translation"],   lidar_ego["rotation"])

        T_cam_global   = T_ego_global_cam   @ T_cam_ego
        T_lidar_global = T_ego_global_lidar @ T_lidar_ego
        T_lidar_to_cam = np.linalg.inv(T_cam_global) @ T_lidar_global

        if args.corr_roll or args.corr_pitch or args.corr_yaw:
            R_corr = euler_to_rot(args.corr_roll, args.corr_pitch, args.corr_yaw)
            T_corr = np.eye(4, dtype=np.float64)
            T_corr[:3, :3] = R_corr
            T_lidar_to_cam = T_corr @ T_lidar_to_cam

        K         = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)
        lidar2img = build_lidar2img(K, T_lidar_to_cam)
        cam_path   = os.path.join(args.root, cam_sd["filename"])
        lidar_path = os.path.join(args.root, lidar_sd["filename"])

    print(f"[INFO] lidar2img:\n{lidar2img}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    img = cv2.imread(cam_path)
    if img is None:
        raise RuntimeError(f"Failed to read camera image: {cam_path}")
    h, w = img.shape[:2]
    print(f"[INFO] Image size: {w}×{h}")

    pts = load_lidar_bin(lidar_path)
    print(f"[INFO] Loaded {pts.shape[0]} points, XYZ range: "
          f"x=[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y=[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] "
          f"z=[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")

    if pts.shape[0] > args.max_points:
        idx = np.random.choice(pts.shape[0], args.max_points, replace=False)
        pts = pts[idx]

    # Quick depth sanity-check: show camera-Z distribution
    ones  = np.ones((pts.shape[0], 1))
    ph    = np.hstack([pts, ones])
    z_cam = (lidar2img[2:3, :] @ ph.T).flatten()
    print(f"[DEBUG] Camera-Z stats: min={z_cam.min():.2f}  max={z_cam.max():.2f}  "
          f"median={np.median(z_cam):.2f}  >0.1: {(z_cam>0.1).sum()}")

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------
    uv_in, n_front = project_points(pts, lidar2img, w, h)

    print(f"[INFO] Total points loaded : {pts.shape[0]}")
    print(f"[INFO] Points in front (z>0.1): {n_front}")
    print(f"[INFO] Points inside image     : {uv_in.shape[0]}")
    if uv_in.shape[0] > 0:
        print(f"[INFO] UV range  min={uv_in.min(axis=0)}  max={uv_in.max(axis=0)}")

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    if uv_in.shape[0] > 0:
        # Compute depth for each surviving point (dot product with camera Z row)
        ones = np.ones((pts.shape[0], 1))
        ph = np.hstack([pts, ones])
        depths = (lidar2img[2:3, :] @ ph.T).flatten()  # camera-space Z

        # Only keep depths for the pts that survived into uv_in
        # Re-project to get depths aligned with uv_in
        pts_2d = (lidar2img @ ph.T).T
        z_all = pts_2d[:, 2]
        front = z_all > 0.1
        uv_all = pts_2d[front, :2] / pts_2d[front, 2:3]
        z_front = z_all[front]
        finite = np.isfinite(uv_all).all(axis=1)
        uv_all = uv_all[finite]
        z_front = z_front[finite]
        in_b = (uv_all[:, 0] >= 0) & (uv_all[:, 0] < w) & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < h)
        uv_draw = uv_all[in_b]
        z_draw = z_front[in_b]

        z_min, z_max = z_draw.min(), z_draw.max()
        mask = np.zeros_like(img)
        for i, (x_f, y_f) in enumerate(uv_draw):
            t = float(np.clip((z_draw[i] - z_min) / (z_max - z_min + 1e-6), 0, 1))
            # Red (close) -> Green -> Blue (far) in BGR
            r = int(255 * max(0, 1 - 2 * t))
            g = int(255 * (1 - abs(2 * t - 1)))
            b = int(255 * max(0, 2 * t - 1))
            cv2.circle(mask, (int(round(x_f)), int(round(y_f))),
                       args.point_radius, (b, g, r), -1)
        img = cv2.addWeighted(img, 1.0, mask, args.overlay_alpha, 0)

    if args.save_mask:
        os.makedirs(os.path.dirname(args.save_mask) or ".", exist_ok=True)
        cv2.imwrite(args.save_mask, mask)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"[INFO] Saved: {args.out}")


if __name__ == "__main__":
    main()