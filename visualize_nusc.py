"""
NuScenes Inference Results Visualizer
--------------------------------------
Supports:
  1. BEV (Bird's Eye View) for a single sample token
  2. BEV grid showing all samples in a scene
  3. Animated BEV GIF across all samples in a scene
  4. Per-class score histogram

Usage:
    python visualize_nusc.py                         # interactive CLI
    python visualize_nusc.py --mode bev              # single BEV (first sample)
    python visualize_nusc.py --mode scene_grid       # grid of all samples
    python visualize_nusc.py --mode animate          # animated GIF
    python visualize_nusc.py --mode stats            # class distribution + score hist
    python visualize_nusc.py --sample <token>        # specify sample token
    python visualize_nusc.py --scene  <token>        # specify scene token
    python visualize_nusc.py --score_thresh 0.3      # score filter (default 0.25)
    python visualize_nusc.py --range 60              # BEV range in metres (default 50)
    python visualize_nusc.py --out output.png        # custom output path
"""

import json, os, argparse, math, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

# ─────────────────────────── data paths ────────────────────────────
#BASE = "/mnt/user-data/uploads"
OUT  = "/mnt/d/teamcarla/futr3d/outputs"
INFO_PKL = "/mnt/d/teamcarla/futr3d/data/nuscenes/nuscenes_infos_val.pkl"
os.makedirs(OUT, exist_ok=True)

def load(name):
    with open(f"{name}") as f:
        return json.load(f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ─────────────────────────── colour map ────────────────────────────
CLASS_COLORS = {
    "car":                  "#4FC3F7",   # light blue
    "truck":                "#FF8A65",   # orange
    "bus":                  "#FFD54F",   # amber
    "trailer":              "#CE93D8",   # purple
    "motorcycle":           "#80CBC4",   # teal
    "bicycle":              "#AED581",   # lime
    "pedestrian":           "#F06292",   # pink
    "traffic_cone":         "#FF7043",   # deep orange
    "barrier":              "#A1887F",   # brown
    "construction_vehicle": "#90A4AE",   # blue-grey
}
DEFAULT_COLOR = "#BDBDBD"

# ─────────────────────────── geometry helpers ──────────────────────

def quat_to_yaw(q):
    """Quaternion [w,x,y,z] → yaw (rotation around Z axis, radians)."""
    w, x, y, z = q
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))


def box_corners_bev(translation, size, rotation_quat):
    """Return 4 corners of the 2-D footprint (x,y) of a 3-D box."""
    cx, cy = translation[0], translation[1]
    l, w   = size[1], size[0]           # length (front/back), width (left/right)
    yaw    = quat_to_yaw(rotation_quat)

    corners_local = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2],
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners_world = corners_local @ rot.T + np.array([cx, cy])
    return corners_world


def ego_to_local(translation, ego_translation, ego_rotation_quat):
    """Transform a world-frame point into the ego vehicle's local frame."""
    dx = translation[0] - ego_translation[0]
    dy = translation[1] - ego_translation[1]
    yaw = quat_to_yaw(ego_rotation_quat)
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx =  c*dx - s*dy
    ly =  s*dx + c*dy
    return lx, ly


def corners_ego_frame(translation, size, rotation_quat, ego_translation, ego_rotation_quat):
    """Box corners in ego-centric frame (ego car at origin, heading = +x)."""
    corners_w = box_corners_bev(translation, size, rotation_quat)
    yaw = quat_to_yaw(ego_rotation_quat)
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    ego_xy = np.array(ego_translation[:2])
    corners_local = (corners_w - ego_xy) @ rot.T
    return corners_local

# ─────────────────────────── BEV drawing ───────────────────────────

def draw_ego_vehicle(ax, size=4.5):
    """Draw a simple car silhouette for the ego vehicle."""
    l, w = size, size * 0.45
    rect = plt.Polygon(
        [[-l/2, -w/2], [l/2, -w/2], [l/2*1.05, 0], [l/2, w/2], [-l/2, w/2]],
        closed=True, facecolor="#FFEB3B", edgecolor="black", linewidth=1.5, zorder=10
    )
    ax.add_patch(rect)
    # arrow showing heading (+x)
    ax.annotate("", xy=(l/2+1.5, 0), xytext=(l/2, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=11)


def draw_bev(ax, boxes, ego_translation, ego_rotation_quat, bev_range,
             score_thresh=0.25, title=""):
    """Core BEV render function."""
    ax.set_facecolor("#1C1C2E")
    ax.set_aspect("equal")
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)

    # concentric range circles
    for r in range(10, bev_range+1, 10):
        circle = plt.Circle((0, 0), r, color="#2E2E4E", fill=False, linewidth=0.8, linestyle="--")
        ax.add_patch(circle)
        ax.text(0, r+0.5, f"{r}m", color="#555577", fontsize=6, ha="center", va="bottom")

    # axes
    ax.axhline(0, color="#2E2E4E", linewidth=0.7)
    ax.axvline(0, color="#2E2E4E", linewidth=0.7)

    draw_ego_vehicle(ax)

    drawn = 0
    for box in boxes:
        if box["detection_score"] < score_thresh:
            continue
        cls   = box["detection_name"]
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        alpha = 0.4 + 0.6 * box["detection_score"]   # fade low-confidence boxes

        corners = corners_ego_frame(
            box["translation"], box["size"], box["rotation"],
            ego_translation, ego_rotation_quat
        )

        # skip if completely out of range
        if np.max(np.abs(corners)) > bev_range * 1.1:
            continue

        poly = plt.Polygon(corners, closed=True,
                           facecolor=color, edgecolor=color,
                           alpha=alpha, linewidth=1.2, zorder=5)
        ax.add_patch(poly)

        # heading arrow on the front edge
        front = (corners[0] + corners[1]) / 2
        cx_loc = corners.mean(axis=0)
        ax.annotate("", xy=front, xytext=cx_loc,
                    arrowprops=dict(arrowstyle="->", color="white",
                                    lw=0.8, mutation_scale=8), zorder=6)

        # velocity arrow (if significant)
        vx, vy = box.get("velocity", [0, 0])
        speed = math.sqrt(vx**2 + vy**2)
        if speed > 0.5:
            yaw_ego = quat_to_yaw(ego_rotation_quat)
            c, s = math.cos(-yaw_ego), math.sin(-yaw_ego)
            vx_l =  c*vx - s*vy
            vy_l =  s*vx + c*vy
            scale = min(speed * 0.8, 5)
            ax.annotate("", xy=(cx_loc[0]+vx_l*scale/speed,
                                cy_loc := cx_loc[1]+vy_l*scale/speed),
                        xytext=(cx_loc[0], cx_loc[1]),
                        arrowprops=dict(arrowstyle="->", color="#FF4081",
                                        lw=1.2, mutation_scale=7), zorder=7)
        drawn += 1

    if title:
        ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    return drawn


def make_legend(fig, ax):
    handles = [mpatches.Patch(color=c, label=cls.replace("_", " ").title())
               for cls, c in CLASS_COLORS.items()]
    # ego vehicle
    handles.append(mpatches.Patch(color="#FFEB3B", label="Ego Vehicle"))
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.25, labelcolor="white",
              facecolor="#1C1C2E", edgecolor="#444444",
              ncol=2, handlelength=1.2)

# ─────────────────────────── build index ───────────────────────────

def build_index(score_thresh, info_pkl_path):
    results = load("/mnt/d/teamcarla/futr3d/carla_results_lidar_cam/pts_bbox/results_nusc.json")["results"]
    info_data = load_pkl(info_pkl_path)
    infos = info_data.get("infos", [])
    info_by_token = {info["token"]: info for info in infos}
    ordered_tokens = [
        info["token"] for info in sorted(infos, key=lambda e: e["timestamp"])
        if info.get("token")
    ]

    # Auto-adapt threshold: if user default (0.25) is above all scores, lower it
    all_scores = [b["detection_score"]
                  for boxes in results.values() for b in boxes]
    if all_scores:
        max_score = max(all_scores)
        if score_thresh > max_score:
            # Use the top-30th percentile as the threshold instead
            import numpy as np
            score_thresh = float(np.percentile(all_scores, 70))
            print(f"[INFO] Score threshold auto-adjusted to {score_thresh:.4f} "
                  f"(max score in data is {max_score:.4f})")

    return dict(results=results, info_by_token=info_by_token,
                ordered_tokens=ordered_tokens, score_thresh=score_thresh)


def get_ego_for_sample(idx, sample_token):
    """Get ego pose from the info pkl entry for the sample token."""
    info = idx["info_by_token"].get(sample_token)
    if not info:
        return [0, 0, 0], [1, 0, 0, 0]
    return info["ego2global_translation"], info["ego2global_rotation"]

# ─────────────────────────── MODE: single BEV ──────────────────────

def mode_bev(idx, sample_token, bev_range, out_path):
    boxes = idx["results"].get(sample_token, [])
    ego_t, ego_r = get_ego_for_sample(idx, sample_token)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0D0D1A")
    n = draw_bev(ax, boxes, ego_t, ego_r, bev_range,
                 idx["score_thresh"],
                 title=f"BEV  ·  sample: {sample_token[:12]}…  "
                       f"(score≥{idx['score_thresh']:.2f})")
    make_legend(fig, ax)
    ax.set_xlabel("← Right   |   Ego   |   Left →", color="#888888", fontsize=8)
    ax.set_ylabel("← Behind   |   Ego   |   Ahead →", color="#888888", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[BEV] drew {n} boxes → {out_path}")

# ─────────────────────────── MODE: scene grid ──────────────────────

def mode_scene_grid(idx, ordered_tokens, bev_range, out_path):
    print(f"[DEBUG] Ordered samples: {len(ordered_tokens)}")
    n = len(ordered_tokens)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(cols*5, rows*5), facecolor="#0D0D1A")
    gs  = GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.05)

    for i, tok in enumerate(ordered_tokens):
        r, c = divmod(i, cols)
        ax   = fig.add_subplot(gs[r, c])
        boxes = idx["results"].get(tok, [])
        ego_t, ego_r = get_ego_for_sample(idx, tok)
        info = idx["info_by_token"].get(tok)
        ts_ms = (info["timestamp"] // 1000) if info else 0
        draw_bev(ax, boxes, ego_t, ego_r, bev_range,
             idx["score_thresh"],
             title=f"t={ts_ms}ms")

    # legend on the last subplot
    make_legend(fig, fig.axes[-1])
    fig.suptitle("All Samples (ordered by timestamp)",
                 color="white", fontsize=13, y=1.01)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[SCENE GRID] {n} samples → {out_path}")

# ─────────────────────────── MODE: animated GIF ────────────────────

def mode_animate(idx, ordered_tokens, bev_range, out_path):
    print(f"[DEBUG] Animation frames: {len(ordered_tokens)}")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D0D1A")

    def update(i):
        ax.clear()
        tok   = ordered_tokens[i]
        boxes = idx["results"].get(tok, [])
        ego_t, ego_r = get_ego_for_sample(idx, tok)
        draw_bev(ax, boxes, ego_t, ego_r, bev_range,
                 idx["score_thresh"],
                 title=f"frame {i+1}/{len(ordered_tokens)}")
        make_legend(fig, ax)

    ani = animation.FuncAnimation(fig, update, frames=len(ordered_tokens),
                                  interval=700, repeat=True)
    ani.save(out_path, writer="pillow", dpi=100)
    plt.close(fig)
    print(f"[ANIMATE] {len(ordered_tokens)} frames → {out_path}")

# ─────────────────────────── MODE: stats ───────────────────────────

def mode_stats(idx, out_path):
    results = idx["results"]
    all_boxes = [b for boxes in results.values() for b in boxes]

    class_counts = {}
    class_scores = {}
    for b in all_boxes:
        cls = b["detection_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        class_scores.setdefault(cls, []).append(b["detection_score"])

    classes = sorted(class_counts, key=lambda c: -class_counts[c])
    colors  = [CLASS_COLORS.get(c, DEFAULT_COLOR) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0D0D1A")
    fig.suptitle("Detection Statistics", color="white", fontsize=14)

    # bar chart
    ax = axes[0]
    ax.set_facecolor("#1C1C2E")
    bars = ax.bar(range(len(classes)), [class_counts[c] for c in classes],
                  color=colors, edgecolor="none")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_"," ") for c in classes],
                       rotation=35, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Count", color="white")
    ax.set_title("Detections per Class  (all samples)", color="white")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    for bar, cls in zip(bars, classes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                str(class_counts[cls]), ha="center", va="bottom",
                color="white", fontsize=8)

    # score distribution violin
    ax2 = axes[1]
    ax2.set_facecolor("#1C1C2E")
    data   = [class_scores[c] for c in classes]
    vparts = ax2.violinplot(data, positions=range(len(classes)),
                            showmedians=True, showextrema=False)
    for i, (pc, cls) in enumerate(zip(vparts["bodies"], classes)):
        pc.set_facecolor(CLASS_COLORS.get(cls, DEFAULT_COLOR))
        pc.set_alpha(0.75)
    vparts["cmedians"].set_color("white")
    vparts["cmedians"].set_linewidth(1.5)
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([c.replace("_"," ") for c in classes],
                        rotation=35, ha="right", color="white", fontsize=9)
    ax2.set_ylabel("Detection Score", color="white")
    ax2.set_title("Score Distribution per Class", color="white")
    ax2.tick_params(colors="#888888")
    ax2.set_ylim(0, 1)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444444")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[STATS] → {out_path}")

def list_available_scenes_and_samples(idx):
    """Print available samples for debugging."""
    print("\n=== Available Samples (first 10) ===")
    for sample_token in list(idx["results"].keys())[:10]:
        boxes = idx["results"][sample_token]
        max_score = max((b["detection_score"] for b in boxes), default=0)
        print(f"  {sample_token}: {len(boxes)} boxes, max_score={max_score:.3f}")

    if len(idx["results"]) > 10:
        print(f"  ... and {len(idx['results']) - 10} more samples")
    print()

# ─────────────────────────── CLI ───────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="bev",
                        choices=["bev","scene_grid","animate","stats","all","list"])
    parser.add_argument("--sample", default=None)
    parser.add_argument("--scene",  default=None)
    parser.add_argument("--score_thresh", type=float, default=0.09)
    parser.add_argument("--range",  type=int,   default=50)
    parser.add_argument("--out",    default=None)
    parser.add_argument("--info",   default=INFO_PKL)
    args = parser.parse_args()

    if not os.path.exists(args.info):
        raise FileNotFoundError(f"Info pkl not found: {args.info}")

    idx = build_index(args.score_thresh, args.info)
    if args.scene:
        print("[INFO] Scene tokens are not available in info pkl; using all samples.")

    # Special mode: list available scenes and samples
    if args.mode == "list":
        list_available_scenes_and_samples(idx)
        return

    # pick defaults – use the sample with highest-scoring detection for best visuals
    all_sample_tokens = list(idx["results"].keys())
    if args.sample:
        if args.sample in idx["results"]:
            first_sample = args.sample
            print(f"[INFO] Using provided sample token: {args.sample}")
        else:
            print(f"[WARNING] Sample token '{args.sample}' not found in results")
            print(f"[INFO] Available samples: {all_sample_tokens[:5]}...")
            first_sample = max(
                all_sample_tokens,
                key=lambda t: max((b["detection_score"] for b in idx["results"][t]), default=0)
            )
            print(f"[INFO] Falling back to best sample: {first_sample}")
    else:
        first_sample = max(
            all_sample_tokens,
            key=lambda t: max((b["detection_score"] for b in idx["results"][t]), default=0)
        )
        print(f"[INFO] Using best sample: {first_sample}")
    ordered_tokens = [t for t in idx["ordered_tokens"] if t in idx["results"]]
    if not ordered_tokens:
        ordered_tokens = list(idx["results"].keys())

    modes = ["bev","scene_grid","animate","stats"] if args.mode == "all" else [args.mode]

    for mode in modes:
        out = args.out
        if mode == "bev":
            out = out or os.path.join(OUT, "bev_single.png")
            mode_bev(idx, first_sample, args.range, out)
        elif mode == "scene_grid":
            out = out or os.path.join(OUT, "scene_grid.png")
            mode_scene_grid(idx, ordered_tokens, args.range, out)
        elif mode == "animate":
            out = out or os.path.join(OUT, "scene_animation.gif")
            mode_animate(idx, ordered_tokens, args.range, out)
        elif mode == "stats":
            out = out or os.path.join(OUT, "detection_stats.png")
            mode_stats(idx, out)


if __name__ == "__main__":
    main()
