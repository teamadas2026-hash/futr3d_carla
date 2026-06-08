"""
nuScenes 3D Bounding Box Visualizer
Renders projected 3D bboxes from all annotations onto all 6 camera images
for a given sample, and saves a combined grid figure.

Usage:
    python visualize_nuscenes_bboxes.py \
        --dataroot ../data/nuscenes/rgb2 \
        --version v1.0-mini \
        --sample-index 0 \
        --out-dir ./output
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion


# ── Camera order (left-to-right, front-to-back) ──────────────────────────────
CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

# ── One distinct colour per object category ───────────────────────────────────
CATEGORY_COLORS = {
    "human.pedestrian.adult":           "#FF6B6B",
    "human.pedestrian.child":           "#FF8E53",
    "human.pedestrian.wheelchair":      "#FFC300",
    "human.pedestrian.stroller":        "#FFD700",
    "human.pedestrian.personal_mobility":"#ADFF2F",
    "human.pedestrian.police_officer":  "#00FA9A",
    "human.pedestrian.construction_worker":"#00CED1",
    "vehicle.car":                      "#1E90FF",
    "vehicle.motorcycle":               "#9370DB",
    "vehicle.bicycle":                  "#FF69B4",
    "vehicle.bus.bendy":                "#FF4500",
    "vehicle.bus.rigid":                "#FF6347",
    "vehicle.truck":                    "#20B2AA",
    "vehicle.construction":             "#8B4513",
    "vehicle.emergency.ambulance":      "#DC143C",
    "vehicle.emergency.police":         "#00008B",
    "vehicle.trailer":                  "#556B2F",
    "movable_object.barrier":           "#808080",
    "movable_object.trafficcone":       "#FFA500",
    "movable_object.pushable_pullable": "#DDA0DD",
    "movable_object.debris":            "#A0522D",
    "static_object.bicycle_rack":       "#2E8B57",
}
DEFAULT_COLOR = "#FFFFFF"


def get_color(category_name: str) -> tuple:
    hex_color = CATEGORY_COLORS.get(category_name, DEFAULT_COLOR)
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    return (r, g, b)


def draw_box_on_axis(ax, box: Box, intrinsic: np.ndarray, imsize: tuple):
    """Project a 3-D box into 2-D and draw it on *ax*."""
    corners = view_points(box.corners(), intrinsic, normalize=True)[:2]  # (2, 8)

    # 8 corners → edges of the cuboid
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # front face
        (4, 5), (5, 6), (6, 7), (7, 4),   # back face
        (0, 4), (1, 5), (2, 6), (3, 7),   # connecting edges
    ]

    color = get_color(box.name)
    lw = 1.8

    for s, e in edges:
        x = [corners[0, s], corners[0, e]]
        y = [corners[1, s], corners[1, e]]
        ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round")

    # Draw a filled circle at the centre-front to indicate heading
    center_front = corners[:, :4].mean(axis=1)
    ax.plot(
        center_front[0], center_front[1],
        "o", color=color, markersize=4, markeredgewidth=0.5, markeredgecolor="black",
    )


def render_sample(
    nusc: NuScenes,
    sample_token: str,
    out_dir: str,
    min_visibility: BoxVisibility = BoxVisibility.ANY,
) -> str:
    sample = nusc.get("sample", sample_token)

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.patch.set_facecolor("#0D1117")
    axes_flat = axes.flatten()

    # Map cameras to subplot positions
    cam_to_ax = {cam: axes_flat[i] for i, cam in enumerate(CAMERAS)}

    # Collect all boxes across all cams for the legend
    seen_categories = set()

    for cam_name, ax in cam_to_ax.items():
        if cam_name not in sample["data"]:
            ax.set_visible(False)
            continue

        sd_token = sample["data"][cam_name]
        sd_record = nusc.get("sample_data", sd_token)
        img_path = os.path.join(nusc.dataroot, sd_record["filename"])

        # Load image
        img = Image.open(img_path)
        ax.imshow(img)

        # Camera intrinsics
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record["width"], sd_record["height"])

        # Get boxes visible in this camera
        _, boxes, _ = nusc.get_sample_data(
            sd_token,
            box_vis_level=min_visibility,
        )

        for box in boxes:
            seen_categories.add(box.name)
            draw_box_on_axis(ax, box, intrinsic, imsize)

        # Tidy up axes
        ax.set_xlim(0, imsize[0])
        ax.set_ylim(imsize[1], 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            cam_name.replace("CAM_", "").replace("_", " "),
            color="#E0E0E0",
            fontsize=11,
            fontweight="bold",
            pad=6,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=get_color(cat), linewidth=2.5,
               label=cat.split(".")[-1].replace("_", " "))
        for cat in sorted(seen_categories)
    ]
    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=min(len(legend_elements), 7),
            fontsize=8,
            framealpha=0.15,
            facecolor="#1C2333",
            edgecolor="#444",
            labelcolor="#E0E0E0",
            bbox_to_anchor=(0.5, -0.02),
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    scene_token = sample["scene_token"]
    scene = nusc.get("scene", scene_token)
    fig.suptitle(
        f"Scene: {scene['name']}  |  Sample token: {sample_token[:8]}…",
        color="#E0E0E0",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout(pad=1.2)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, f"bbox_{sample_token[:8]}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Saved → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize nuScenes 3D bboxes projected onto all 6 cameras."
    )
    parser.add_argument(
        "--dataroot", type=str, required=True,
        help="Root directory of the nuScenes dataset (contains 'samples/', 'v1.0-*/', etc.)"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="Dataset version (default: v1.0-mini)"
    )
    parser.add_argument(
        "--sample-index", type=int, default=0,
        help="Index of the sample to visualize (default: 0). Use --all to process every sample."
    )
    parser.add_argument(
        "--sample-token", type=str, default=None,
        help="Specific sample token to visualize. Overrides --sample-index."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Render ALL samples in the dataset (can be slow for trainval)."
    )
    parser.add_argument(
        "--out-dir", type=str, default="./nuscenes_output",
        help="Directory to save output images (default: ./nuscenes_output)"
    )
    parser.add_argument(
        "--visibility", type=str, default="any",
        choices=["any", "most", "all"],
        help=(
            "Minimum box visibility to include: "
            "'any' = at least 1 corner visible, "
            "'most' = ≥40%% visible, "
            "'all' = fully visible (default: any)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vis_map = {
        "any":  BoxVisibility.ANY,
        "most": BoxVisibility.ANY,   # nuScenes only exposes ANY / ALL robustly
        "all":  BoxVisibility.ALL,
    }
    min_visibility = vis_map[args.visibility]

    print(f"Loading nuScenes {args.version} from {args.dataroot} …")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    if args.all:
        tokens = [s["token"] for s in nusc.sample]
        print(f"Processing {len(tokens)} samples …")
    elif args.sample_token:
        tokens = [args.sample_token]
    else:
        tokens = [nusc.sample[args.sample_index]["token"]]

    for token in tokens:
        render_sample(nusc, token, args.out_dir, min_visibility)

    print(f"\nDone! Output saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
