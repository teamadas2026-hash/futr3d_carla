"""
nuScenes LiDAR + 3D Bounding Box Visualizer
Uses Plotly for a fully interactive 3-D viewer in your browser.
Works on WSL2 / headless / any environment — no GPU or display server needed.

Points are coloured by intensity (plasma colorscale).
Boxes are per-category coloured wireframe cuboids.

Usage:
    python visualize_nuscenes_lidar.py \
        --dataroot ../data/nuscenes/rgb2 \
        --version v1.0-mini \
        --sample-index 0

Output:
    An HTML file is saved to --out-dir and opened in your default browser.

Controls (in browser):
    Left-drag   : rotate
    Right-drag  : pan
    Scroll      : zoom
    Double-click: reset view
"""

import argparse
import os
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box


# ── Per-category colours (hex) ────────────────────────────────────────────────
CATEGORY_COLORS = {
    "human.pedestrian.adult":                "#FF6B6B",
    "human.pedestrian.child":                "#FF8E53",
    "human.pedestrian.wheelchair":           "#FFC300",
    "human.pedestrian.stroller":             "#FFD700",
    "human.pedestrian.personal_mobility":    "#ADFF2F",
    "human.pedestrian.police_officer":       "#00FA9A",
    "human.pedestrian.construction_worker":  "#00CED1",
    "vehicle.car":                           "#1E90FF",
    "vehicle.motorcycle":                    "#9370DB",
    "vehicle.bicycle":                       "#FF69B4",
    "vehicle.bus.bendy":                     "#FF4500",
    "vehicle.bus.rigid":                     "#FF6347",
    "vehicle.truck":                         "#20B2AA",
    "vehicle.construction":                  "#8B4513",
    "vehicle.emergency.ambulance":           "#DC143C",
    "vehicle.emergency.police":              "#00008B",
    "vehicle.trailer":                       "#8FBC8F",
    "movable_object.barrier":               "#A0A0A0",
    "movable_object.trafficcone":            "#FFA500",
    "movable_object.pushable_pullable":      "#DDA0DD",
    "movable_object.debris":                "#A0522D",
    "static_object.bicycle_rack":            "#2E8B57",
}
DEFAULT_COLOR = "#FFFFFF"

CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # front face
    (4, 5), (5, 6), (6, 7), (7, 4),   # back face
    (0, 4), (1, 5), (2, 6), (3, 7),   # lateral edges
]


def get_color(category_name: str) -> str:
    return CATEGORY_COLORS.get(category_name, DEFAULT_COLOR)


def box_to_lines(box: Box):
    """Return (xs, ys, zs) lists with None gaps suitable for a single Scatter3d line trace."""
    corners = box.corners().T  # (8, 3)
    xs, ys, zs = [], [], []
    for s, e in CUBOID_EDGES:
        xs += [corners[s, 0], corners[e, 0], None]
        ys += [corners[s, 1], corners[e, 1], None]
        zs += [corners[s, 2], corners[e, 2], None]
    return xs, ys, zs


def render_sample(
    nusc: NuScenes,
    sample_token: str,
    out_dir: str,
    point_ratio: float = 1.0,
) -> str:
    sample = nusc.get("sample", sample_token)
    scene  = nusc.get("scene", sample["scene_token"])
    print(f"\nScene : {scene['name']}")
    print(f"Sample: {sample_token}")

    # ── 1. Load LiDAR ─────────────────────────────────────────────────────────
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data  = nusc.get("sample_data", lidar_token)
    lidar_path  = os.path.join(nusc.dataroot, lidar_data["filename"])

    pc        = LidarPointCloud.from_file(lidar_path)
    pts_xyz   = pc.points[:3].T   # (N, 3)
    intensity = pc.points[3]      # (N,)

    # Optionally downsample for faster rendering
    if point_ratio < 1.0:
        idx     = np.random.choice(len(pts_xyz), int(len(pts_xyz) * point_ratio), replace=False)
        pts_xyz   = pts_xyz[idx]
        intensity = intensity[idx]

    print(f"Points: {len(pts_xyz):,}")

    # ── 2. Build point cloud trace ─────────────────────────────────────────────
    pcd_trace = go.Scatter3d(
        x=pts_xyz[:, 0],
        y=pts_xyz[:, 1],
        z=pts_xyz[:, 2],
        mode="markers",
        marker=dict(
            size=1.2,
            color=intensity,
            colorscale="Plasma",
            cmin=0,
            cmax=255,
            colorbar=dict(
                title=dict(text="Intensity", font=dict(color="#cccccc")),
                tickfont=dict(color="#cccccc"),
                thickness=12,
                len=0.5,
                x=1.01,
            ),
            opacity=0.85,
        ),
        name="LiDAR points",
        hoverinfo="skip",
    )

    # ── 3. Load and draw boxes ─────────────────────────────────────────────────
    _, boxes, _ = nusc.get_sample_data(lidar_token)

    # Group boxes by category so each gets one legend entry
    cat_traces: dict[str, dict] = {}   # category → {xs, ys, zs}
    for box in boxes:
        cat = box.name
        if cat not in cat_traces:
            cat_traces[cat] = {"xs": [], "ys": [], "zs": []}
        xs, ys, zs = box_to_lines(box)
        cat_traces[cat]["xs"].extend(xs)
        cat_traces[cat]["ys"].extend(ys)
        cat_traces[cat]["zs"].extend(zs)

    box_traces = []
    for cat, data in cat_traces.items():
        label = cat.split(".")[-1].replace("_", " ").title()
        box_traces.append(go.Scatter3d(
            x=data["xs"], y=data["ys"], z=data["zs"],
            mode="lines",
            line=dict(color=get_color(cat), width=3),
            name=label,
            legendgroup=label,
            hoverinfo="name",
        ))

    print(f"Boxes : {len(boxes)}  ({len(cat_traces)} categories)")

    # ── 4. Heading arrows (front-face centre dots) ────────────────────────────
    head_x, head_y, head_z, head_c = [], [], [], []
    for box in boxes:
        c = box.corners().T  # (8,3)
        front_centre = c[:4].mean(axis=0)
        head_x.append(front_centre[0])
        head_y.append(front_centre[1])
        head_z.append(front_centre[2])
        head_c.append(get_color(box.name))

    heading_trace = go.Scatter3d(
        x=head_x, y=head_y, z=head_z,
        mode="markers",
        marker=dict(size=4, color=head_c, symbol="circle", opacity=1.0),
        name="Heading (front face)",
        hoverinfo="skip",
        showlegend=True,
    )

    # ── 5. Assemble figure ─────────────────────────────────────────────────────
    all_traces = [pcd_trace] + box_traces + [heading_trace]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        title=dict(
            text=f"<b>nuScenes LiDAR</b> — {scene['name']}  |  {sample_token[:8]}…",
            font=dict(color="#E0E0E0", size=14),
            x=0.01,
        ),
        scene=dict(
            xaxis=dict(title="X (m)", gridcolor="#1f2937", backgroundcolor="#0D1117",
                       showbackground=True, color="#888"),
            yaxis=dict(title="Y (m)", gridcolor="#1f2937", backgroundcolor="#0D1117",
                       showbackground=True, color="#888"),
            zaxis=dict(title="Z (m)", gridcolor="#1f2937", backgroundcolor="#0D1117",
                       showbackground=True, color="#888"),
            bgcolor="#0D1117",
            # Start with a BEV-like camera
            camera=dict(
                eye=dict(x=0, y=0, z=2.2),
                up=dict(x=1, y=0, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="data",
        ),
        legend=dict(
            font=dict(color="#cccccc", size=10),
            bgcolor="rgba(20,25,35,0.8)",
            bordercolor="#444",
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=850,
    )

    # ── 6. Save & open ─────────────────────────────────────────────────────────
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"lidar_{sample_token[:8]}.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\n[✓] Saved → {out_path}")

    # Try to open in browser (works on WSL2 if a Windows browser is in PATH)
    try:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")
        print("[✓] Opened in browser")
    except Exception:
        print("[!] Could not auto-open browser — open the file manually.")

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize nuScenes LiDAR + 3D bboxes in an interactive browser viewer."
    )
    p.add_argument("--dataroot",      type=str, required=True)
    p.add_argument("--version",       type=str, default="v1.0-mini",
                   choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"])
    p.add_argument("--sample-index",  type=int, default=0)
    p.add_argument("--sample-token",  type=str, default=None,
                   help="Specific sample token — overrides --sample-index")
    p.add_argument("--out-dir",       type=str, default="./nuscenes_output")
    p.add_argument("--point-ratio",   type=float, default=1.0,
                   help="Fraction of points to render, 0–1 (default: 1.0). "
                        "Use 0.5 to halve point count for faster load.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading nuScenes {args.version} from {args.dataroot} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    token = args.sample_token or nusc.sample[args.sample_index]["token"]
    render_sample(nusc, token, args.out_dir, args.point_ratio)


if __name__ == "__main__":
    main()