"""
verify_lidar_detection.py
=========================
Overlays depth-colored LiDAR points AND projected 3D ground-truth bounding
boxes onto the front camera image.  Run it on any sample from the info pkl
to verify that the calibration is correct and the detector is seeing the
right objects.

Usage
-----
python verify_lidar_detection.py \\
    --root  /path/to/nuscenes \\
    --info-pkl  nuscenes_infos_val.pkl \\
    --sample-index 0 \\
    --out  verify_out.png

Optional flags
    --no-points   skip lidar point overlay (show boxes only)
    --no-boxes    skip box overlay         (show points only)
    --max-points  20000
    --point-radius 2
    --depth-min / --depth-max   manual depth clamp (metres) for colour scale
"""

import argparse, os
import cv2
import mmcv
import numpy as np
from pyquaternion import Quaternion


# ───────────────────────── helpers ─────────────────────────────────────────

def make_T(translation, rotation):
    """4×4 rigid transform.  rotation = quaternion list or 3×3 ndarray."""
    rotation = np.asarray(rotation, dtype=np.float64)
    rot = rotation if rotation.shape == (3, 3) else Quaternion(rotation).rotation_matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3]  = np.asarray(translation, dtype=np.float64)
    return T


def load_lidar_bin(path):
    """Load .pcd.bin with automatic column-count detection (5 / 4 / 3)."""
    for dtype in (np.float32, np.float64):
        raw = np.fromfile(path, dtype=dtype)
        for ncols in (5, 4, 3):
            if raw.size >= ncols and raw.size % ncols == 0:
                pts = raw.reshape(-1, ncols)[:, :3].astype(np.float64)
                if np.isfinite(pts).all() and np.abs(pts).max() < 1000:
                    return pts
    raise ValueError(f"Cannot read point cloud: {path}")


def load_sample(pkl_path, idx):
    d = mmcv.load(pkl_path)
    infos = d["infos"] if "infos" in d else (d["data_list"] if "data_list" in d else d)
    return infos[idx]


def build_lidar2img(info, cam_info):
    """Build 3×4 projection matrix from ego-based calibration."""
    K  = np.array(cam_info["cam_intrinsic"], dtype=np.float64)
    Tl = make_T(info["lidar2ego_translation"],         info["lidar2ego_rotation"])
    Tc = make_T(cam_info["sensor2ego_translation"],    cam_info["sensor2ego_rotation"])
    T_lidar_to_cam = np.linalg.inv(Tc) @ Tl
    return K @ T_lidar_to_cam[:3, :], T_lidar_to_cam


def project(pts_lidar, lidar2img):
    """(N,3) → (N,2) pixel + (N,) depth.  Returns only z>0 points."""
    ones  = np.ones((len(pts_lidar), 1))
    ph    = np.hstack([pts_lidar, ones])
    p2d   = (lidar2img @ ph.T).T
    z     = p2d[:, 2]
    mask  = z > 0.1
    p2d, z = p2d[mask], z[mask]
    uv    = p2d[:, :2] / z[:, None]
    fin   = np.isfinite(uv).all(axis=1)
    return uv[fin], z[fin]


def depth_to_bgr(z, z_min, z_max):
    """Scalar depth → BGR tuple.  Red = near, yellow = mid, green = far."""
    t = float(np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1))
    r = int(255 * max(0.0, 1.0 - 2*t))
    g = int(255 * (1.0 - abs(2*t - 1.0)))
    b = int(255 * max(0.0, 2*t - 1.0))
    return (b, g, r)


# ───────────────────────── 3-D box drawing ─────────────────────────────────

CLASS_COLORS = {          # BGR
    "car":          (0,   200, 255),
    "truck":        (0,   140, 255),
    "bus":          (0,    80, 255),
    "pedestrian":   (0,   255, 100),
    "cyclist":      (255, 200,   0),
    "motorcycle":   (255, 100,   0),
    "bicycle":      (200, 255,   0),
    "traffic_cone": (255,   0, 200),
    "barrier":      (180, 180, 180),
}
DEFAULT_COLOR = (200, 200, 200)

# Edge pairs for a box with corners ordered:
#   0-3  top face (z = +dz/2)   4-7  bottom face (z = -dz/2)
EDGES = [
    (0,1),(1,2),(2,3),(3,0),   # top
    (4,5),(5,6),(6,7),(7,4),   # bottom
    (0,4),(1,5),(2,6),(3,7),   # verticals
]
# Front-face corners (indices where x = +dx/2 in local box frame) → highlight
FRONT_EDGES = [(0,1),(1,5),(5,4),(4,0)]


def box_corners_lidar(box):
    """Return (8,3) corners in lidar frame for a 7-DoF box [x,y,z,dx,dy,dz,yaw]."""
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    # local corners: ±dx/2, ±dy/2, ±dz/2
    lx, ly, lz = dx/2, dy/2, dz/2
    local = np.array([
        [ lx,  ly,  lz],  # 0 top-front-right
        [ lx, -ly,  lz],  # 1 top-front-left
        [-lx, -ly,  lz],  # 2 top-rear-left
        [-lx,  ly,  lz],  # 3 top-rear-right
        [ lx,  ly, -lz],  # 4 bot-front-right
        [ lx, -ly, -lz],  # 5 bot-front-left
        [-lx, -ly, -lz],  # 6 bot-rear-left
        [-lx,  ly, -lz],  # 7 bot-rear-right
    ], dtype=np.float64)
    return (R @ local.T).T + np.array([cx, cy, cz])


def draw_box_3d(img, corners_lidar, lidar2img, color, label=None):
    """Project 3-D box corners and draw edges onto img (in-place)."""
    ones = np.ones((8, 1))
    ph   = np.hstack([corners_lidar, ones])
    p2d  = (lidar2img @ ph.T).T
    z    = p2d[:, 2]

    # skip box if all corners behind camera
    if (z <= 0).all():
        return

    # project only positive-depth corners; mark the rest as invalid
    valid = z > 0
    uv = np.full((8, 2), np.nan)
    uv[valid] = (p2d[valid, :2] / z[valid, None])

    h, w = img.shape[:2]

    def to_pt(idx):
        u, v = uv[idx]
        if not (np.isfinite(u) and np.isfinite(v)): return None
        return (int(round(float(u))), int(round(float(v))))

    # Draw all 12 edges (thin, semi-transparent look via alpha)
    overlay = img.copy()
    for i, j in EDGES:
        p1, p2 = to_pt(i), to_pt(j)
        if p1 and p2:
            cv2.line(overlay, p1, p2, color, 1, cv2.LINE_AA)

    # Highlight front face (thicker, brighter)
    for i, j in FRONT_EDGES:
        p1, p2 = to_pt(i), to_pt(j)
        if p1 and p2:
            cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Label at projected center (bottom of box)
    if label:
        bottom_mid_uv = uv[[4,5,6,7]]
        valid_bm = bottom_mid_uv[np.isfinite(bottom_mid_uv).all(axis=1)]
        if len(valid_bm):
            cx_lbl = int(valid_bm[:, 0].mean())
            cy_lbl = int(valid_bm[:, 1].mean())
            if 0 <= cx_lbl < w and 0 <= cy_lbl < h:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img,
                              (cx_lbl - 2, cy_lbl - th - 4),
                              (cx_lbl + tw + 2, cy_lbl + 2),
                              (0, 0, 0), -1)
                cv2.putText(img, label,
                            (cx_lbl, cy_lbl - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 1, cv2.LINE_AA)


# ───────────────────────── main ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",         required=True)
    ap.add_argument("--info-pkl",     required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--out",          default="./verify_out.png")
    ap.add_argument("--max-points",   type=int, default=20000)
    ap.add_argument("--point-radius", type=int, default=2)
    ap.add_argument("--depth-min",    type=float, default=None)
    ap.add_argument("--depth-max",    type=float, default=None)
    ap.add_argument("--no-points",    action="store_true")
    ap.add_argument("--no-boxes",     action="store_true")
    args = ap.parse_args()

    info     = load_sample(args.info_pkl, args.sample_index)
    cam_info = info["cams"]["CAM_FRONT"]

    # Resolve paths
    def rp(p):
        if p is None: return None
        if os.path.isabs(p): return p
        return os.path.normpath(os.path.join(args.root, p))

    cam_path   = rp(cam_info.get("data_path") or cam_info.get("img_path"))
    lidar_path = rp(info.get("lidar_path") or info.get("pts_path"))

    img = cv2.imread(cam_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {cam_path}")
    h, w = img.shape[:2]

    lidar2img, _ = build_lidar2img(info, cam_info)
    print(f"[INFO] Image: {w}×{h}")
    print(f"[INFO] lidar2img:\n{lidar2img}")

    # ── Point cloud overlay ──────────────────────────────────────────────
    if not args.no_points:
        pts = load_lidar_bin(lidar_path)
        print(f"[INFO] Loaded {len(pts)} points")
        if len(pts) > args.max_points:
            idx = np.random.choice(len(pts), args.max_points, replace=False)
            pts = pts[idx]

        uv, depths = project(pts, lidar2img)
        in_b = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        uv, depths = uv[in_b], depths[in_b]
        print(f"[INFO] Points in image: {len(uv)}")

        z_min = args.depth_min if args.depth_min is not None else float(depths.min())
        z_max = args.depth_max if args.depth_max is not None else float(depths.max())
        print(f"[INFO] Depth range for colour scale: {z_min:.1f}m – {z_max:.1f}m")

        mask = np.zeros_like(img)
        for (xf, yf), z in zip(uv, depths):
            bgr = depth_to_bgr(z, z_min, z_max)
            cv2.circle(mask, (int(round(xf)), int(round(yf))),
                       args.point_radius, bgr, -1)
        img = cv2.addWeighted(img, 1.0, mask, 0.8, 0)

    # ── 3-D box overlay ──────────────────────────────────────────────────
    if not args.no_boxes:
        boxes = np.array(info["gt_boxes"]) if len(info["gt_boxes"]) else np.zeros((0,7))
        names = list(info["gt_names"])
        drawn, skipped = 0, 0
        for box, name in zip(boxes, names):
            cx,cy,cz = float(box[0]),float(box[1]),float(box[2])
            uv_h = lidar2img @ np.array([cx,cy,cz,1.0])
            depth = float(uv_h[2])
            if depth <= 0:
                skipped += 1
                continue
            u, v = float(uv_h[0])/depth, float(uv_h[1])/depth
            # only draw if center or at least part of box could be visible
            if not (-w*0.5 <= u <= w*1.5 and -h*0.5 <= v <= h*1.5):
                skipped += 1
                continue
            color   = CLASS_COLORS.get(name.lower(), DEFAULT_COLOR)
            corners = box_corners_lidar(box)
            draw_box_3d(img, corners, lidar2img, color, label=name)
            drawn += 1

        print(f"[INFO] Boxes drawn: {drawn}  |  skipped (behind/off-image): {skipped}")

    # ── Legend ───────────────────────────────────────────────────────────
    # Colour-by-depth bar (bottom-left)
    if not args.no_points:
        bar_w, bar_h, bar_x, bar_y = 200, 14, 20, h - 40
        for i in range(bar_w):
            t = i / bar_w
            z = z_min + t * (z_max - z_min)
            bgr = depth_to_bgr(z, z_min, z_max)
            cv2.rectangle(img, (bar_x+i, bar_y), (bar_x+i+1, bar_y+bar_h), bgr, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 1)
        cv2.putText(img, f"{z_min:.0f}m", (bar_x, bar_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(img, f"{z_max:.0f}m", (bar_x+bar_w-28, bar_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # Class legend (bottom-right)
    if not args.no_boxes:
        legend_x = w - 150
        legend_y = h - 20 - len(CLASS_COLORS)*18
        for cls, bgr in CLASS_COLORS.items():
            cv2.rectangle(img, (legend_x, legend_y), (legend_x+12, legend_y+10), bgr, -1)
            cv2.putText(img, cls, (legend_x+16, legend_y+9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1)
            legend_y += 16

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"[INFO] Saved: {args.out}")


if __name__ == "__main__":
    main()
