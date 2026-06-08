"""
Full-Fusion NuScenes Result Visualizer

Improved visualization utility for FUTR3D full-fusion outputs.

Highlights:
- Configurable input/output paths (no hardcoded paths required)
- Better validation and informative errors
- Reusable index builder for results + nuScenes tables
- Multiple modes: single BEV, scene grid, animated GIF, and class stats
- Auto-threshold adaptation if requested threshold is above all scores

Example usage:
    python visualize_full_fusion.py
    python visualize_full_fusion.py --mode bev
    python visualize_full_fusion.py --mode scene_grid --scene <scene_token>
    python visualize_full_fusion.py --mode animate --scene <scene_token>
    python visualize_full_fusion.py --mode stats
    python visualize_full_fusion.py --mode all --score-thresh 0.1
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


CLASS_COLORS = {
    "car": "#4FC3F7",
    "truck": "#FF8A65",
    "bus": "#FFD54F",
    "trailer": "#CE93D8",
    "motorcycle": "#80CBC4",
    "bicycle": "#AED581",
    "pedestrian": "#F06292",
    "traffic_cone": "#FF7043",
    "barrier": "#A1887F",
    "construction_vehicle": "#90A4AE",
}
DEFAULT_COLOR = "#BDBDBD"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quat_to_yaw(quat_wxyz: List[float]) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def box_corners_bev(translation: List[float], size: List[float], rotation_quat: List[float]) -> np.ndarray:
    cx, cy = translation[0], translation[1]
    length, width = size[1], size[0]
    yaw = quat_to_yaw(rotation_quat)

    corners_local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return corners_local @ rot.T + np.array([cx, cy])


def corners_ego_frame(
    translation: List[float],
    size: List[float],
    rotation_quat: List[float],
    ego_translation: List[float],
    ego_rotation_quat: List[float],
) -> np.ndarray:
    corners_world = box_corners_bev(translation, size, rotation_quat)
    yaw = quat_to_yaw(ego_rotation_quat)
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    ego_xy = np.array(ego_translation[:2])
    return (corners_world - ego_xy) @ rot.T


def draw_ego_vehicle(ax, size: float = 4.5) -> None:
    length, width = size, size * 0.45
    body = plt.Polygon(
        [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length * 0.525, 0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        closed=True,
        facecolor="#FFEB3B",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )
    ax.add_patch(body)
    ax.annotate(
        "",
        xy=(length / 2.0 + 1.5, 0),
        xytext=(length / 2.0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        zorder=11,
    )


def draw_bev(ax, boxes, ego_translation, ego_rotation_quat, bev_range, score_thresh=0.05, title="") -> int:
    ax.set_facecolor("#1C1C2E")
    ax.set_aspect("equal")
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)

    for r in range(10, bev_range + 1, 10):
        circle = plt.Circle((0, 0), r, color="#2E2E4E", fill=False, linewidth=0.8, linestyle="--")
        ax.add_patch(circle)
        ax.text(0, r + 0.5, f"{r}m", color="#555577", fontsize=6, ha="center", va="bottom")

    ax.axhline(0, color="#2E2E4E", linewidth=0.7)
    ax.axvline(0, color="#2E2E4E", linewidth=0.7)
    draw_ego_vehicle(ax)

    drawn = 0
    for box in boxes:
        if box.get("detection_score", 0.0) < score_thresh:
            continue

        cls = box.get("detection_name", "unknown")
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        alpha = 0.4 + 0.6 * float(box.get("detection_score", 0.0))

        corners = corners_ego_frame(
            box["translation"],
            box["size"],
            box["rotation"],
            ego_translation,
            ego_rotation_quat,
        )

        if np.max(np.abs(corners)) > bev_range * 1.1:
            continue

        poly = plt.Polygon(
            corners,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=1.2,
            zorder=5,
        )
        ax.add_patch(poly)

        front = (corners[0] + corners[1]) / 2.0
        center = corners.mean(axis=0)
        ax.annotate(
            "",
            xy=front,
            xytext=center,
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8, mutation_scale=8),
            zorder=6,
        )

        velocity = box.get("velocity", [0.0, 0.0])
        vx, vy = float(velocity[0]), float(velocity[1])
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 0.5:
            yaw_ego = quat_to_yaw(ego_rotation_quat)
            c, s = math.cos(-yaw_ego), math.sin(-yaw_ego)
            vx_local = c * vx - s * vy
            vy_local = s * vx + c * vy
            scale = min(speed * 0.8, 5.0)
            ax.annotate(
                "",
                xy=(center[0] + vx_local * scale / speed, center[1] + vy_local * scale / speed),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color="#FF4081", lw=1.2, mutation_scale=7),
                zorder=7,
            )
        drawn += 1

    if title:
        ax.set_title(title, color="white", fontsize=9, pad=4)

    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    return drawn


def make_legend(ax) -> None:
    handles = [mpatches.Patch(color=color, label=cls.replace("_", " ").title()) for cls, color in CLASS_COLORS.items()]
    handles.append(mpatches.Patch(color="#FFEB3B", label="Ego Vehicle"))
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        framealpha=0.25,
        labelcolor="white",
        facecolor="#1C1C2E",
        edgecolor="#444444",
        ncol=2,
        handlelength=1.2,
    )


def build_index(results_path: Path, nusc_root: Path, score_thresh: float) -> Dict:
    raw = load_json(results_path)
    if "results" not in raw:
        raise ValueError(f"Invalid results file format (missing 'results'): {results_path}")

    results = raw["results"]
    if not isinstance(results, dict) or len(results) == 0:
        raise ValueError(f"No detections found in: {results_path}")

    ego_poses = {e["token"]: e for e in load_json(nusc_root / "ego_pose.json")}
    sample_data = {}
    for sd in load_json(nusc_root / "sample_data.json"):
        sample_data.setdefault(sd["sample_token"], []).append(sd)

    scenes = {s["token"]: s for s in load_json(nusc_root / "scene.json")}
    samples = {s["token"]: s for s in load_json(nusc_root / "sample.json")}

    all_scores = [b.get("detection_score", 0.0) for boxes in results.values() for b in boxes]
    if all_scores:
        max_score = max(all_scores)
        if score_thresh > max_score:
            score_thresh = float(np.percentile(all_scores, 70.0))
            print(
                f"[INFO] Score threshold adjusted to {score_thresh:.4f} "
                f"(max in results is {max_score:.4f})"
            )

    return {
        "results": results,
        "ego_poses": ego_poses,
        "sample_data": sample_data,
        "scenes": scenes,
        "samples": samples,
        "score_thresh": score_thresh,
    }


def get_ego_for_sample(idx: Dict, sample_token: str) -> Tuple[List[float], List[float]]:
    sample_data_entries = idx["sample_data"].get(sample_token, [])

    lidar_sd = next((sd for sd in sample_data_entries if sd.get("channel") == "LIDAR_TOP"), None)
    fallback_sd = sample_data_entries[0] if sample_data_entries else None
    sd = lidar_sd if lidar_sd is not None else fallback_sd

    if sd is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    pose = idx["ego_poses"].get(sd["ego_pose_token"])
    if pose is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]

    return pose["translation"], pose["rotation"]


def ordered_scene_samples(idx: Dict, scene_token: str) -> List[str]:
    scene = idx["scenes"].get(scene_token)
    if scene is None:
        raise KeyError(f"Unknown scene token: {scene_token}")

    ordered = []
    cur = scene["first_sample_token"]
    while cur:
        ordered.append(cur)
        cur = idx["samples"][cur].get("next", "")
    return ordered


def resolve_scene_token(idx: Dict, scene_arg: str, fallback_sample: str) -> str:
    if scene_arg:
        if scene_arg in idx["scenes"]:
            return scene_arg
        by_name = [tok for tok, sc in idx["scenes"].items() if sc.get("name") == scene_arg]
        if by_name:
            return by_name[0]
        raise KeyError(f"Scene not found as token or name: {scene_arg}")

    sample = idx["samples"].get(fallback_sample)
    if sample:
        return sample["scene_token"]

    return max(idx["scenes"], key=lambda t: idx["scenes"][t].get("nbr_samples", 0))


def mode_bev(idx: Dict, sample_token: str, bev_range: int, out_path: Path) -> None:
    boxes = idx["results"].get(sample_token, [])
    ego_t, ego_r = get_ego_for_sample(idx, sample_token)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0D0D1A")
    num_drawn = draw_bev(
        ax,
        boxes,
        ego_t,
        ego_r,
        bev_range,
        score_thresh=idx["score_thresh"],
        title=f"Full Fusion BEV | sample {sample_token[:12]} | score >= {idx['score_thresh']:.2f}",
    )
    make_legend(ax)
    ax.set_xlabel("X (m)", color="#888888", fontsize=8)
    ax.set_ylabel("Y (m)", color="#888888", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[BEV] drew {num_drawn} boxes -> {out_path}")


def mode_scene_grid(idx: Dict, scene_token: str, bev_range: int, out_path: Path, max_samples: int) -> None:
    scene = idx["scenes"][scene_token]
    ordered = ordered_scene_samples(idx, scene_token)
    if max_samples > 0:
        ordered = ordered[:max_samples]

    n = len(ordered)
    cols = min(n, 4)
    rows = int(math.ceil(n / cols))

    fig = plt.figure(figsize=(cols * 5, rows * 5), facecolor="#0D0D1A")

    for i, tok in enumerate(ordered):
        ax = fig.add_subplot(rows, cols, i + 1)
        boxes = idx["results"].get(tok, [])
        ego_t, ego_r = get_ego_for_sample(idx, tok)
        ts_ms = idx["samples"][tok]["timestamp"] // 1000
        draw_bev(ax, boxes, ego_t, ego_r, bev_range, idx["score_thresh"], title=f"t={ts_ms} ms")

    make_legend(fig.axes[-1])
    fig.suptitle(
        f"Full Fusion Scene Grid: {scene.get('name', scene_token)} | {scene.get('description', '')}",
        color="white",
        fontsize=13,
        y=1.01,
    )
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[SCENE GRID] {n} samples -> {out_path}")


def mode_animate(idx: Dict, scene_token: str, bev_range: int, out_path: Path, interval_ms: int, max_samples: int) -> None:
    scene = idx["scenes"][scene_token]
    ordered = ordered_scene_samples(idx, scene_token)
    if max_samples > 0:
        ordered = ordered[:max_samples]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0D0D1A")

    def update(i):
        ax.clear()
        tok = ordered[i]
        boxes = idx["results"].get(tok, [])
        ego_t, ego_r = get_ego_for_sample(idx, tok)
        draw_bev(
            ax,
            boxes,
            ego_t,
            ego_r,
            bev_range,
            idx["score_thresh"],
            title=f"{scene.get('name', scene_token)} | frame {i + 1}/{len(ordered)}",
        )
        make_legend(ax)

    ani = animation.FuncAnimation(fig, update, frames=len(ordered), interval=interval_ms, repeat=True)
    ani.save(str(out_path), writer="pillow", dpi=100)
    plt.close(fig)
    print(f"[ANIMATE] {len(ordered)} frames -> {out_path}")


def mode_stats(idx: Dict, out_path: Path) -> None:
    all_boxes = [b for boxes in idx["results"].values() for b in boxes]
    class_counts: Dict[str, int] = {}
    class_scores: Dict[str, List[float]] = {}

    for box in all_boxes:
        cls = box.get("detection_name", "unknown")
        score = float(box.get("detection_score", 0.0))
        class_counts[cls] = class_counts.get(cls, 0) + 1
        class_scores.setdefault(cls, []).append(score)

    classes = sorted(class_counts, key=lambda c: -class_counts[c])
    colors = [CLASS_COLORS.get(c, DEFAULT_COLOR) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0D0D1A")
    fig.suptitle("Full Fusion Detection Statistics", color="white", fontsize=14)

    ax = axes[0]
    ax.set_facecolor("#1C1C2E")
    bars = ax.bar(range(len(classes)), [class_counts[c] for c in classes], color=colors, edgecolor="none")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=35, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Count", color="white")
    ax.set_title("Detections per Class", color="white")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    for bar, cls in zip(bars, classes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(class_counts[cls]),
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
        )

    ax2 = axes[1]
    ax2.set_facecolor("#1C1C2E")
    data = [class_scores[c] for c in classes]
    vparts = ax2.violinplot(data, positions=range(len(classes)), showmedians=True, showextrema=False)
    for body, cls in zip(vparts["bodies"], classes):
        body.set_facecolor(CLASS_COLORS.get(cls, DEFAULT_COLOR))
        body.set_alpha(0.75)
    vparts["cmedians"].set_color("white")
    vparts["cmedians"].set_linewidth(1.5)
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([c.replace("_", " ") for c in classes], rotation=35, ha="right", color="white", fontsize=9)
    ax2.set_ylabel("Detection Score", color="white")
    ax2.set_title("Score Distribution per Class", color="white")
    ax2.set_ylim(0, 1)
    ax2.tick_params(colors="#888888")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444444")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[STATS] -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize full-fusion FUTR3D nuScenes results.")
    parser.add_argument("--mode", default="bev", choices=["bev", "scene_grid", "animate", "stats", "all"])
    parser.add_argument("--results", default="carla_results_lidar_cam/pts_bbox/results_nusc.json")
    parser.add_argument("--nusc-root", default="data/nuscenes/v1.0-mini")
    parser.add_argument("--output-dir", default="outputs/full_fusion")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--scene", default=None, help="Scene token or scene name.")
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--range", dest="bev_range", type=int, default=50)
    parser.add_argument("--max-scene-samples", type=int, default=0, help="Limit number of scene samples for grid/animation. 0 means no limit.")
    parser.add_argument("--gif-interval-ms", type=int, default=700)
    parser.add_argument("--out", default=None, help="Output file path for single mode. Ignored when --mode all.")
    args = parser.parse_args()

    results_path = Path(args.results)
    nusc_root = Path(args.nusc_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx = build_index(results_path=results_path, nusc_root=nusc_root, score_thresh=args.score_thresh)
    all_tokens = list(idx["results"].keys())

    if args.sample:
        first_sample = args.sample
    else:
        first_sample = max(
            all_tokens,
            key=lambda t: max((b.get("detection_score", 0.0) for b in idx["results"].get(t, [])), default=0.0),
        )

    first_scene = resolve_scene_token(idx, args.scene, first_sample)

    modes = ["bev", "scene_grid", "animate", "stats"] if args.mode == "all" else [args.mode]

    for mode in modes:
        out = Path(args.out) if args.out and len(modes) == 1 else None
        if mode == "bev":
            out = out or (output_dir / "bev_single_full_fusion.png")
            mode_bev(idx, first_sample, args.bev_range, out)
        elif mode == "scene_grid":
            out = out or (output_dir / "scene_grid_full_fusion.png")
            mode_scene_grid(idx, first_scene, args.bev_range, out, args.max_scene_samples)
        elif mode == "animate":
            out = out or (output_dir / "scene_animation_full_fusion.gif")
            mode_animate(idx, first_scene, args.bev_range, out, args.gif_interval_ms, args.max_scene_samples)
        elif mode == "stats":
            out = out or (output_dir / "detection_stats_full_fusion.png")
            mode_stats(idx, out)


if __name__ == "__main__":
    main()
