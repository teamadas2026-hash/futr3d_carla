"""
Single-sample multi-modal visualization for a CARLA-generated nuScenes dataset.

For ONE example sample it draws two panels:
  (left)  a camera image with LiDAR (depth-coloured) and radar (magenta stars)
          points projected on top of it
  (right) a bird's-eye view with LiDAR (faint, height-coloured) and radar
          (bold, with ego-motion-compensated velocity arrows) overlaid

Sensor channels are auto-detected. Radar velocity is compensated for ego
motion at load time, since this dataset's stored vx_comp / vy_comp are just
copies of the raw values.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
VERSION            = "v1.0-carla"
DATAROOT           = "data/nuscenes/rgb2"
SAMPLE_TOKEN       = None       # None -> auto-pick the middle sample of scene 0
BEV_RANGE          = 60.0       # half-width of the BEV window (m)
OUTPATH            = "sample_lidar_radar_overlay.png"

SHOW_VELOCITY      = True       # draw radar velocity arrows in the BEV panel
USE_COMPENSATION   = True       # subtract ego motion at load time
FLIP_VELOCITY_SIGN = False      # set True if static returns don't drop to ~0
VELOCITY_SECS      = 0.5        # arrow length = speed * this many seconds (m)
VELOCITY_MIN       = 0.5        # don't draw arrows below this speed (m/s)

# CARLA radar fields don't carry meaningful dynprop / validity flags, so
# turn off the devkit's default filtering or almost everything gets dropped.
RadarPointCloud.disable_filters()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def find_channels(sample):
    keys = list(sample["data"].keys())
    lidar = next((k for k in keys if "LIDAR" in k.upper()), None)
    radars = [k for k in keys if "RADAR" in k.upper()]
    cam = next((k for k in keys if k.upper() == "CAM_FRONT"), None) \
        or next((k for k in keys if "CAM" in k.upper()), None)
    return lidar, radars, cam


def load_pc_in_ego(nusc, sample, channel, is_radar):
    """Load a point cloud and move it from sensor frame into the ego frame.
    Returns (pc, sensor_rotation_matrix, sample_data_record)."""
    sd = nusc.get("sample_data", sample["data"][channel])
    path = os.path.join(nusc.dataroot, sd["filename"])
    pc = (RadarPointCloud if is_radar else LidarPointCloud).from_file(path)

    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    rot = Quaternion(cs["rotation"]).rotation_matrix
    pc.rotate(rot)
    pc.translate(np.array(cs["translation"]))
    return pc, rot, sd


def ego_velocity_in_ego_frame(nusc, sd):
    """Estimate the ego's velocity (3-vector) in the ego frame at sample_data sd,
    by differencing consecutive ego_pose translations."""
    if sd["next"]:
        sd2 = nusc.get("sample_data", sd["next"])
        dt = (sd2["timestamp"] - sd["timestamp"]) / 1e6
        sign = +1.0
    elif sd["prev"]:
        sd2 = nusc.get("sample_data", sd["prev"])
        dt = (sd["timestamp"] - sd2["timestamp"]) / 1e6
        sign = -1.0
    else:
        return None
    if dt <= 0:
        return None

    p0 = np.array(nusc.get("ego_pose", sd["ego_pose_token"])["translation"])
    p1 = np.array(nusc.get("ego_pose", sd2["ego_pose_token"])["translation"])
    v_global = sign * (p1 - p0) / dt

    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    return Quaternion(ego["rotation"]).inverse.rotate(v_global)


def radar_velocity_ego_frame(nusc, pc, sd, rot):
    """Radar velocity (vx, vy) in the ego frame."""
    raw = rot @ np.vstack([
        pc.points[6],
        pc.points[7],
        np.zeros_like(pc.points[6])
    ])
    vx, vy = raw[0], raw[1]

    if USE_COMPENSATION:
        v_ego = ego_velocity_in_ego_frame(nusc, sd)
        if v_ego is not None:
            xy = pc.points[:2]
            rhat = xy / (np.linalg.norm(xy, axis=0) + 1e-6)
            v_radial = v_ego[0] * rhat[0] + v_ego[1] * rhat[1]

            s = -1.0 if FLIP_VELOCITY_SIGN else 1.0
            vx = vx + s * v_radial * rhat[0]
            vy = vy + s * v_radial * rhat[1]

    return vx, vy


def pick_sample(nusc):
    if SAMPLE_TOKEN:
        return nusc.get("sample", SAMPLE_TOKEN)

    scene = nusc.scene[0]
    tokens = []
    t = scene["first_sample_token"]

    while t:
        tokens.append(t)
        t = nusc.get("sample", t)["next"]

    return nusc.get("sample", tokens[len(tokens) // 2])


# ----------------------------------------------------------------------
# Panel 1 -- camera image with LiDAR + radar projected on top
# ----------------------------------------------------------------------
def draw_camera_panel(nusc, sample, lidar_ch, radar_chs, cam_ch, ax):
    cam_token = sample["data"][cam_ch]

    l_pts, l_depth, im = nusc.explorer.map_pointcloud_to_image(
        sample["data"][lidar_ch], cam_token)

    ax.imshow(im)
    ax.scatter(
        l_pts[0], l_pts[1],
        c=l_depth, s=2,
        cmap="viridis", alpha=0.6
    )

    for ch in radar_chs:
        try:
            r_pts, _, _ = nusc.explorer.map_pointcloud_to_image(
                sample["data"][ch], cam_token)

            if r_pts.shape[1]:
                ax.scatter(
                    r_pts[0], r_pts[1],
                    c="magenta",
                    marker="*",
                    s=120,
                    edgecolor="k",
                    linewidths=0.4,
                    label="radar" if ch == radar_chs[0] else None
                )
        except Exception:
            pass

    ax.set_title(f"{cam_ch} + LiDAR (depth) + radar", weight="bold")
    ax.axis("off")

    if radar_chs:
        ax.legend(loc="upper right", frameon=True)


# ----------------------------------------------------------------------
# Panel 2 -- BEV overlay of LiDAR and radar
# ----------------------------------------------------------------------
def draw_bev_panel(nusc, sample, lidar_ch, radar_chs, ax):
    lpc, _, _ = load_pc_in_ego(nusc, sample, lidar_ch, is_radar=False)

    lx, ly, lz = lpc.points[0], lpc.points[1], lpc.points[2]
    ax.scatter(lx, ly, c=lz, s=1, cmap="Greys", alpha=0.5)

    first = True
    for ch in radar_chs:
        rpc, rot, sd = load_pc_in_ego(nusc, sample, ch, is_radar=True)

        rx, ry = rpc.points[0], rpc.points[1]

        ax.scatter(
            rx, ry,
            c="magenta",
            marker="*",
            s=90,
            edgecolor="k",
            linewidths=0.4,
            zorder=5,
            label="radar" if first else None
        )

        if SHOW_VELOCITY:
            vx, vy = radar_velocity_ego_frame(nusc, rpc, sd, rot)
            speed = np.hypot(vx, vy)

            keep = speed > VELOCITY_MIN
            if keep.any():
                ax.quiver(
                    rx[keep], ry[keep],
                    vx[keep] * VELOCITY_SECS,
                    vy[keep] * VELOCITY_SECS,
                    color="#1f77b4",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.004,
                    alpha=0.8,
                    zorder=4
                )

        first = False

    ax.scatter(
    0, 0,
    marker=(3, 0, -90),
    c="red",
    s=300,
    edgecolor="k",
    zorder=6,
    label="ego (facing +x)"
)

    ax.set_xlim(-BEV_RANGE, BEV_RANGE)
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel("x — forward (m)")
    ax.set_ylabel("y — left (m)")

    title = "BEV — LiDAR + radar overlay"
    if SHOW_VELOCITY:
        title += " (velocity " + (
            "compensated" if USE_COMPENSATION else "raw"
        ) + ")"

    ax.set_title(title, weight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    nusc = NuScenes(
        version=VERSION,
        dataroot=DATAROOT,
        verbose=True
    )

    sample = pick_sample(nusc)
    lidar_ch, radar_chs, cam_ch = find_channels(sample)

    print(f"sample      : {sample['token']}")
    print(f"lidar       : {lidar_ch}")
    print(f"radars      : {radar_chs}")
    print(f"camera      : {cam_ch}")

    if lidar_ch is None:
        raise RuntimeError(
            "No LiDAR channel found in this sample."
        )

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    draw_camera_panel(
        nusc,
        sample,
        lidar_ch,
        radar_chs,
        cam_ch,
        axes[0]
    )

    draw_bev_panel(
        nusc,
        sample,
        lidar_ch,
        radar_chs,
        axes[1]
    )

    fig.suptitle(
        "Multi-modal sample: LiDAR + radar",
        fontsize=18,
        weight="bold"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPATH,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nsaved {OUTPATH}")
    plt.show()