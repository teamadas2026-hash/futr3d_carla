"""
Real-time FUTR3D (LiDAR + 6 cam) inference on live CARLA frames.

Bridge: feed a synthetic, pkl-shaped data_info (constant calibration from a real
.pkl entry, dynamic file paths swapped per frame) through the FULL test pipeline,
so it is the same code path as test.py. Multi-sweep accumulation (sweeps_num=9)
is reproduced live with mmdet3d's obtain_sensor2top math.

Spawning reuses your own Client, so live sensor settings are identical to training.

EDIT THE CONFIG BLOCK BELOW. Run from the futr3d root with CARLA already running:
    python realtime/realtime_detect.py
"""
from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
import importlib
import os
import queue
import sys

import cv2
import mmcv
import numpy as np
import torch
import yaml
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from pyquaternion import Quaternion

# =============================================================================
# CONFIG  —  the only environment-specific edits
# =============================================================================
CONFIG     = "plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py"
CHECKPOINT = "work_dirs/lidar_0075v_cam_vov/epoch_12.pth"
PKL        = "./data/nuscenes/rgb2/nuscenes_infos_train.pkl"   # any existing info pkl

CARLA_HOST = "172.21.192.1"        # from your config_rgb.yaml client block
CARLA_PORT = 2000

# your CARLA data-generation package (the one holding client.py / sensor.py / utils.py)
CARLA_PKG_DIR  = "/mnt/d/teamcarla/futr3d/carla_nuscenes"   # the OUTER folder
CARLA_PKG_NAME = "carla_nuscenes"                            # the inner package with __init__.py
CALIB_YAML = os.path.join(CARLA_PKG_DIR, "configs", "calibrated_sensors_rgb.yaml")
MAP_NAME       = "Town10HD_opt"                 # the CARLA map you want to run on; should match your calib yaml

DEVICE     = "cuda:0"
SCORE_THR  = 0.3
INFER_EVERY = 1                   # sync mode pauses the sim while we infer, so 1 is fine
TMP = "/tmp/carla_rt"
os.makedirs(TMP, exist_ok=True)

# --- reuse YOUR OWN code so live inputs are format-identical to training -----
sys.path.insert(0, CARLA_PKG_DIR)
_sensor = importlib.import_module(f"{CARLA_PKG_NAME}.sensor")
_utils = importlib.import_module(f"{CARLA_PKG_NAME}.utils")
Client = importlib.import_module(f"{CARLA_PKG_NAME}.client").Client
parse_image = _sensor.parse_image
parse_lidar_data = _sensor.parse_lidar_data
get_nuscenes_rt = _utils.get_nuscenes_rt

import carla  # noqa: E402  (carla is on path once the package imports it)

# your overlay function (same import that worked in validate_offline.py)
try:
    from tools.misc.visualize_results import project_boxes_to_image
except ImportError:
    sys.path.insert(0, os.path.abspath("tools"))
    from visualize_results import project_boxes_to_image

def chase_transform(ego_tf, distance=8.0, height=5.0, pitch=-15.0):
    import math
    yaw = math.radians(ego_tf.rotation.yaw)
    loc = ego_tf.location + carla.Location(x=-distance * math.cos(yaw),
                                           y=-distance * math.sin(yaw),
                                           z=height)
    return carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=ego_tf.rotation.yaw))

# =============================================================================
#record helper
def build_grid(tiles, h=300):
    rs = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    w = max(t.shape[1] for t in rs)
    pad = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in rs]
    return np.concatenate([np.concatenate(pad[0:3], axis=1),
                           np.concatenate(pad[3:6], axis=1)], axis=0)


def add_velocity(img, boxes_3d, scores, labels, lidar2img,
                 score_thr=0.0, arrow_seconds=1.0, min_speed=0.5):
    """Overlay velocity arrows + speed text. vx,vy are box tensor cols 7,8 (m/s,
    LiDAR frame). Arrow length = distance the object travels in `arrow_seconds`."""
    keep = scores > score_thr
    b = boxes_3d[keep]
    if len(b) == 0:
        return img
    centers = b.gravity_center.numpy()          # (N,3) box centers, LiDAR frame
    vels = b.tensor[:, 7:9].numpy()             # (N,2) vx, vy

    def project(pts3):
        h = np.concatenate([pts3, np.ones((len(pts3), 1))], axis=1)
        p = (lidar2img @ h.T).T
        d = p[:, 2].copy()
        p[:, :2] /= p[:, 2:3] + 1e-6
        return p[:, :2], d

    tips = centers.copy()
    tips[:, 0] += vels[:, 0] * arrow_seconds
    tips[:, 1] += vels[:, 1] * arrow_seconds
    c2d, c_d = project(centers)
    t2d, t_d = project(tips)

    for n in range(len(b)):
        speed = float(np.linalg.norm(vels[n]))   # m/s
        if c_d[n] < 0.1 or speed < min_speed:
            continue
        cx, cy = int(c2d[n, 0]), int(c2d[n, 1])
        if t_d[n] > 0.1:
            tx, ty = int(t2d[n, 0]), int(t2d[n, 1])
            cv2.arrowedLine(img, (cx, cy), (tx, ty), (0, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.3)
        cv2.putText(img, f"{speed * 3.6:.0f} km/h", (cx, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img
# =============================================================================
# 1) ONE-TIME SETUP
# =============================================================================
def add_pipeline_fields(input_dict, box_type_3d="LiDAR"):
    """Replicate Custom3DDataset.pre_pipeline so test-time transforms find keys."""
    box_type, box_mode = get_box_type(box_type_3d)
    input_dict["box_type_3d"] = box_type
    input_dict["box_mode_3d"] = box_mode
    for k in ("img_fields", "bbox3d_fields", "pts_mask_fields",
              "pts_seg_fields", "bbox_fields", "mask_fields", "seg_fields"):
        input_dict[k] = []
    return input_dict


def load_plugin(cfg):
    """Import the FUTR3D plugin so its detector registers."""
    if not getattr(cfg, "plugin", False):
        return
    sys.path.insert(0, os.path.abspath("."))
    plugin_path = cfg.plugin if isinstance(cfg.plugin, str) else getattr(cfg, "plugin_dir", "")
    module_path = plugin_path.strip("/").replace("/", ".")
    importlib.import_module(module_path)
    print("Imported plugin:", module_path)


def build_runtime(config_path, checkpoint_path, pkl_path, device=DEVICE):
    cfg = Config.fromfile(config_path)
    load_plugin(cfg)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = load_checkpoint(model, checkpoint_path, map_location="cpu")
    classes = ckpt.get("meta", {}).get("CLASSES", None) or cfg.get("class_names")
    model.CLASSES = classes
    model = MMDataParallel(model.to(device), device_ids=[int(device.split(":")[-1])])
    model.eval()

    pipeline = Compose(cfg.data.test.pipeline)   # FULL pipeline, loaders included

    infos = mmcv.load(pkl_path)
    data_infos = infos["infos"] if isinstance(infos, dict) else infos
    template = data_infos[0]
    cam_order = list(template["cams"].keys())
    print("Camera order (feed images in THIS order):", cam_order)

    lidar2img = []
    for cam_type in cam_order:
        ci = template["cams"][cam_type]
        l2c_r = np.linalg.inv(ci["sensor2lidar_rotation"])
        l2c_t = ci["sensor2lidar_translation"] @ l2c_r.T
        l2c_rt = np.eye(4)
        l2c_rt[:3, :3] = l2c_r.T
        l2c_rt[3, :3] = -l2c_t
        intr = ci["cam_intrinsic"]
        viewpad = np.eye(4)
        viewpad[: intr.shape[0], : intr.shape[1]] = intr
        lidar2img.append(viewpad @ l2c_rt.T)

    return dict(model=model, cfg=cfg, pipeline=pipeline, template=template,
                cam_order=cam_order, lidar2img=lidar2img, classes=classes)


# =============================================================================
# 2) MULTI-SWEEP ACCUMULATION (reproduces mmdet3d obtain_sensor2top)
# =============================================================================
def _ego_pose_nuscenes(ego_transform):
    quat, trans = get_nuscenes_rt(ego_transform)           # your util, unchanged
    return np.array(trans, dtype=np.float64), Quaternion(quat).rotation_matrix


def _sensor2lidar(l2e_t, l2e_r, e2g_t, e2g_r, l2e_t_s, l2e_r_s, e2g_t_s, e2g_r_s):
    inv = np.linalg.inv
    R = (l2e_r_s.T @ e2g_r_s.T) @ (inv(e2g_r).T @ inv(l2e_r).T)
    T = (l2e_t_s @ e2g_r_s.T + e2g_t_s) @ (inv(e2g_r).T @ inv(l2e_r).T)
    T -= e2g_t @ (inv(e2g_r).T @ inv(l2e_r).T) + l2e_t @ inv(l2e_r).T
    return R.T, T


class SweepBuffer:
    def __init__(self, template, max_sweeps=9, tmp_dir=os.path.join(TMP, "sweeps")):
        os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_dir = tmp_dir
        self.max_sweeps = max_sweeps
        self.l2e_t = np.array(template["lidar2ego_translation"], dtype=np.float64)
        self.l2e_r = Quaternion(template["lidar2ego_rotation"]).rotation_matrix
        self.buf = []
        self._n = 0

    def build_sweeps(self, key_ego_transform):
        e2g_t, e2g_r = _ego_pose_nuscenes(key_ego_transform)
        sweeps = []
        for e in reversed(self.buf):                       # newest previous first
            R, T = _sensor2lidar(self.l2e_t, self.l2e_r, e2g_t, e2g_r,
                                 self.l2e_t, self.l2e_r, e["e2g_t"], e["e2g_r"])
            sweeps.append(dict(data_path=e["path"], timestamp=e["ts_us"],
                               sensor2lidar_rotation=R, sensor2lidar_translation=T))
        return sweeps

    def push(self, points_n5, ego_transform, ts_seconds):
        path = os.path.join(self.tmp_dir, f"sweep_{self._n % self.max_sweeps}.bin")
        points_n5.astype(np.float32).tofile(path)
        e2g_t, e2g_r = _ego_pose_nuscenes(ego_transform)
        self.buf.append(dict(path=path, e2g_t=e2g_t, e2g_r=e2g_r,
                             ts_us=int(ts_seconds * 1e6)))
        if len(self.buf) > self.max_sweeps:
            self.buf.pop(0)
        self._n += 1


# =============================================================================
# 3) PER-FRAME INFERENCE
# =============================================================================
def _write_frame(rt, images_bgra, pts):
    lidar_path = os.path.join(TMP, "lidar.bin")
    pts.astype(np.float32).tofile(lidar_path)
    cam_paths = {}
    for cam in rt["cam_order"]:
        cv2.imwrite(os.path.join(TMP, f"{cam}.png"), images_bgra[cam][:, :, :3])
        cam_paths[cam] = os.path.join(TMP, f"{cam}.png")
    return lidar_path, cam_paths


def make_input_dict(rt, lidar_path, cam_paths, sweeps, ts_seconds):
    input_dict = dict(
        sample_idx=rt["template"]["token"],
        pts_filename=lidar_path,
        sweeps=sweeps,
        timestamp=ts_seconds,
        img_filename=[cam_paths[c] for c in rt["cam_order"]],
        lidar2img=[m.copy() for m in rt["lidar2img"]],
    )
    add_pipeline_fields(input_dict, rt["cfg"].data.test.get("box_type_3d", "LiDAR"))
    return input_dict


@torch.no_grad()
def infer_core(rt, images_bgra, pts, ego_tf, ts, sweep_buf, score_thr=SCORE_THR):
    lidar_path, cam_paths = _write_frame(rt, images_bgra, pts)
    sweeps = sweep_buf.build_sweeps(ego_tf)                 # previous clouds
    data = rt["pipeline"](make_input_dict(rt, lidar_path, cam_paths, sweeps, ts))
    data = collate([data], samples_per_gpu=1)
    result = rt["model"](return_loss=False, rescale=True, **data)
    res = result[0].get("pts_bbox", result[0])
    #-------------------------------------------------------------
    # boxes = res["boxes_3d"]
    # scores = res["scores_3d"]
    # labels = res["labels_3d"]

    # for i in range(min(10, len(scores))):
    #     print(f"\nDetection {i}")
    #     print("Score:", float(scores[i]))
    #     print("Label:", int(labels[i]))
    #     print("Box:", boxes.tensor[i].cpu().numpy())
    #-------------------------------------------------------------
    keep = res["scores_3d"] >= score_thr
    return res["boxes_3d"][keep], res["scores_3d"][keep], res["labels_3d"][keep]


# =============================================================================
# 4) VISUALIZATION
# =============================================================================
def add_track_ids(img, boxes_3d, track_ids, lidar2img, color=(0, 255, 0)):
    """Overlay track IDs on 3D boxes projected to 2D image.
    
    Args:
        img: Image array to draw on
        boxes_3d: 3D boxes (N, 7) [h, w, l, x, y, z, yaw] OR Box3D object
        track_ids: Track IDs (N,)
        lidar2img: Lidar to image projection matrix (4, 4)
        color: Color for text (BGR)
    """
    if len(boxes_3d) == 0:
        return img
    
    # Get box centers for projection
    if hasattr(boxes_3d, 'gravity_center'):
        # mmdet3d Box3D format
        centers = boxes_3d.gravity_center.numpy()
    else:
        # Raw numpy array format: [h, w, l, x, y, z, yaw]
        centers = boxes_3d[:, 3:6]  # Extract x, y, z
    
    # Project centers to image space
    ones = np.ones((len(centers), 1))
    pts_h = np.concatenate([centers, ones], axis=1)  # (N, 4)
    pts_img = (lidar2img @ pts_h.T).T  # (N, 4)
    
    # Get 2D coordinates and depth
    depths = pts_img[:, 2]
    pts_2d = pts_img[:, :2] / (pts_img[:, 2:3] + 1e-6)  # (N, 2)
    
    # Draw track IDs on valid points
    for i in range(len(track_ids)):
        if depths[i] > 0.1:  # Only draw if in front of camera
            x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
            track_id = int(track_ids[i])
            
            # Draw text
            text = f"ID:{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background for text readability
            cv2.rectangle(img, (x - 3, y - text_size[1] - 5),
                         (x + text_size[0] + 3, y + 3), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img


def draw_overlays_with_tracks(rt, boxes, scores, labels, images_bgra, all_tracks_per_cat):
    """Draw detection overlays plus track IDs.
    
    Args:
        all_tracks_per_cat: dict mapping class name to array of tracked boxes
    """
    tiles = []
    for i, cam in enumerate(rt["cam_order"]):
        bgr = images_bgra[cam][:, :, :3].copy()
        
        # Draw detection boxes
        vis = project_boxes_to_image(boxes, scores, labels, rt["lidar2img"][i],
                                     bgr, list(rt["classes"]), score_thr=0.0)
        
        # Add velocity arrows
        vis = add_velocity(vis, boxes, scores, labels, rt["lidar2img"][i], score_thr=0.0)
        
        # Add track IDs for each category
        for cat_idx, cat in enumerate(rt["classes"]):
            if cat in all_tracks_per_cat and len(all_tracks_per_cat[cat]) > 0:
                cat_boxes_tracked = all_tracks_per_cat[cat]['boxes']
                cat_track_ids = all_tracks_per_cat[cat]['track_ids']
                
                # Draw track IDs on this category
                vis = add_track_ids(vis, cat_boxes_tracked, cat_track_ids, 
                                   rt["lidar2img"][i], color=(0, 255, 0))
        
        # Add camera label
        cv2.putText(vis, cam, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(vis)
    
    return tiles



def show_grid(tiles, h=300):
    rs = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    w = max(t.shape[1] for t in rs)
    pad = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in rs]
    grid = np.concatenate([np.concatenate(pad[0:3], axis=1),
                           np.concatenate(pad[3:6], axis=1)], axis=0)
    cv2.imshow("FUTR3D realtime", grid)


# =============================================================================
# 5) FRAME-SYNCED CAPTURE
# =============================================================================
class SyncCapture:
    """Re-listen the 6 cams + lidar onto queues; pull data matched by frame id."""
    def __init__(self, cam_actors, lidar_actor):
        self.cam_q = [queue.Queue() for _ in cam_actors]
        self.lidar_q = queue.Queue()
        for q, a in zip(self.cam_q, cam_actors):
            self._relisten(a, q.put)
        self._relisten(lidar_actor, self.lidar_q.put)

    @staticmethod
    def _relisten(actor, cb):
        # Sensor.set_actor() already attached add_data; a second listen()
        # without stop() crashes CARLA. Clear it first, then listen fresh.
        try:
            actor.stop()
        except RuntimeError:
            pass
        actor.listen(cb)

    def grab(self, frame, timeout=10.0):
        def pop(q):
            while True:
                d = q.get(timeout=timeout)
                if d.frame == frame:
                    return d
        return [pop(q) for q in self.cam_q], pop(self.lidar_q)


# =============================================================================
# 6) BOOTSTRAP via your Client + main loop
# =============================================================================
CLEAR_WEATHER = dict(cloudiness=0, precipitation=0, precipitation_deposits=0,
                     wind_intensity=0, sun_azimuth_angle=0, sun_altitude_angle=90,
                     fog_density=0, fog_distance=0, wetness=0, fog_falloff=0,
                     scattering_intensity=0, mie_scattering_scale=0,
                     rayleigh_scattering_scale=0.0331, dust_storm=0)


def bootstrap_client():
    client = Client({"host": CARLA_HOST, "port": CARLA_PORT, "time_out": 6000.0}, random_seed=0)
    client.generate_world({"map_name": MAP_NAME, "settings": {"fixed_delta_seconds": 0.083333}})
    with open(CALIB_YAML) as f:
        calib = yaml.safe_load(f)
    scene_config = dict(
        custom=True, weather_mode="custom", weather=CLEAR_WEATHER,
        ego_vehicle=dict(bp_name="vehicle.tesla.model3", location=None,
                         rotation={"yaw": 0, "pitch": 0, "roll": 0}, options=None),
        traffic=dict(cars=30, trucks=6, bikes=6, vans=6, walkers=60),
        calibrated_sensors=calib, ego_speed_diff=-20, description="realtime",
    )
    client.generate_scene(scene_config)
    return client


def build_tracker_config():
    """Create a minimal config object for AB3DMOT initialization."""
    class MinimalConfig:
        def __init__(self):
            self.dataset = "nuScenes"
            self.det_name = "centerpoint"
            self.vis = False
            self.ego_com = 0
            self.affi_pro = False
    return MinimalConfig()


def create_trackers(classes):
    """Create trackers for supported categories. Skip unsupported ones."""
    # Map lowercase class names to AB3DMOT's expected capitalized format
    class_name_map = {
        'car': 'Car',
        'truck': 'Truck',
        'construction_vehicle': 'Truck',  # Map to Truck as closest match
        'bus': 'Bus',
        'trailer': 'Trailer',
        'motorcycle': 'Motorcycle',
        'bicycle': 'Bicycle',
        'pedestrian': 'Pedestrian',
    }
    
    # AB3DMOT supports these categories for nuScenes
    supported_cats = {'Car', 'Pedestrian', 'Truck', 'Trailer', 'Bus', 'Motorcycle', 'Bicycle'}
    tracker_cfg = build_tracker_config()
    trackers = {}
    
    for cat in classes:
        # Map lowercase to capitalized format
        mapped_cat = class_name_map.get(cat.lower(), None)
        
        if mapped_cat and mapped_cat in supported_cats:
            try:
                # Suppress verbose logging for tracker setup
                t = AB3DMOT(tracker_cfg, mapped_cat)
                print(f"✓ Created tracker for: {cat} -> {mapped_cat} (min_hits={t.min_hits}, max_age={t.max_age})")
                trackers[cat] = t  # Use original lowercase class name as key
            except Exception as e:
                print(f"✗ Could not create tracker for {cat}: {e}")
    
    print(f"Active trackers: {list(trackers.keys())}\n")
    return trackers


def run_loop(rt, client):
    import traceback
    world = client.world
    ego_actor = client.ego_vehicle.get_actor()
    spectator = world.get_spectator()
    by_name = {s.name: s for s in client.sensors}
    cam_actors = [by_name[c].get_actor() for c in rt["cam_order"]]
    lidar_actor = by_name["LIDAR_TOP"].get_actor()

    keep = {a.id for a in cam_actors} | {lidar_actor.id}
     
    # Create tracker for supported categories
    trackers = create_trackers(rt["classes"])
    for s in client.sensors:
        a = s.get_actor()
        if a is not None and a.id not in keep:
            try:
                a.stop()
            except Exception:
                pass

    sync = SyncCapture(cam_actors, lidar_actor)
    sweep_buf = SweepBuffer(rt["template"], max_sweeps=9)
    writer = None
    SIM_FPS = int(round(1 / 0.083333))      # 12
    tick = 0
    try:
        while True:
            frame = world.tick()
            cams_raw, lidar_raw = sync.grab(frame)
            images_bgra = {rt["cam_order"][i]: parse_image(cams_raw[i]) for i in range(6)}
            pts = parse_lidar_data(lidar_raw)
            ego_tf = ego_actor.get_transform()
            spectator.set_transform(chase_transform(ego_tf))   # follow the ego
            ts = lidar_raw.timestamp
            tick += 1

            if tick % INFER_EVERY == 0:
                boxes, scores, labels = infer_core(rt, images_bgra, pts, ego_tf, ts, sweep_buf)
                
                # Track detections per category (only for supported categories)
                boxes_np = boxes.tensor.cpu().numpy()
                scores_np = scores.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                print(f"\n=== Tick {tick} ===")
                print(f"Detections: {len(scores)}, Trackers: {list(trackers.keys())}")
                
                all_tracks = []
                all_tracks_per_cat = {}  # Store tracks per category for visualization
                
                for cat_idx, cat in enumerate(rt["classes"]):
                    if cat not in trackers:
                        continue
                    
                    cat_mask = labels_np == cat_idx
                    
                    # Build detection array for this category
                    if cat_mask.any():
                        cat_dets = []
                        cat_info = []
                        for i in np.where(cat_mask)[0]:
                            x, y, z, w, l, h, yaw = boxes_np[i][:7]
                            score = scores_np[i]
                            cat_dets.append([h, w, l, x, y, z, yaw])
                            cat_info.append([score, 0, 0, 0, 0, 0, 0, 0])
                        cat_dets = np.array(cat_dets)
                        cat_info = np.array(cat_info)
                    else:
                        cat_dets = np.empty((0, 7))
                        cat_info = np.empty((0, 8))
                    
                    print(f"  {cat}: {len(cat_dets)} dets", end="")
                    
                    # Call AB3DMOT.track() with proper format
                    dets_all = {'dets': cat_dets, 'info': cat_info}
                    try:
                        tracker = trackers[cat]
                        results, affi = tracker.track(dets_all, tick, 'carla_realtime')
                        
                        if len(results) > 0 and results[0].shape[0] > 0:
                            tracks_output = results[0]
                            print(f" -> {tracks_output.shape[0]} tracks")
                            
                            # Extract box data and track IDs for visualization
                            tracked_boxes_list = []
                            track_ids_list = []
                            
                            for t in tracks_output:
                                h, w, l, x, y, z, yaw, track_id = t[:8]
                                tracked_boxes_list.append([h, w, l, x, y, z, yaw])
                                track_ids_list.append(int(track_id))
                                all_tracks.append([h, w, l, x, y, z, yaw, track_id])
                            
                            # Store for visualization
                            if len(tracked_boxes_list) > 0:
                                all_tracks_per_cat[cat] = {
                                    'boxes': np.array(tracked_boxes_list),
                                    'track_ids': np.array(track_ids_list)
                                }
                        else:
                            print(" -> 0 tracks")
                    except Exception as e:
                        print(f" -> ERROR: {e}")
                
                print(f"Result: {len(all_tracks)} total tracks\n")

                # Draw overlays with track IDs
                tiles = draw_overlays_with_tracks(rt, boxes, scores, labels, images_bgra, all_tracks_per_cat)
                grid = build_grid(tiles)
                
                if writer is None:
                    fh, fw = grid.shape[:2]
                    writer = cv2.VideoWriter("realtime_detections.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             SIM_FPS, (fw, fh))
                writer.write(grid)
                cv2.imshow("FUTR3D realtime", grid)
                if cv2.waitKey(1) == 27:        # Esc to quit
                    break

            sweep_buf.push(pts, ego_tf, ts)
    except Exception:
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.release()
        for fn in (client.destroy_scene, client.destroy_world):
            try:
                fn()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rt = build_runtime(CONFIG, CHECKPOINT, PKL)
    client = bootstrap_client()
    run_loop(rt, client)