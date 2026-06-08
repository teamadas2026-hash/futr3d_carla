"""
futr3d_carla.py
---------------
All-in-one CARLA + FUTR3D real-time inference script.

Data pipeline exactly mirrors the training collection pipeline:
  - LiDAR: parse_lidar_data() from sensor.py
      → (N,5) float32: [x, -y, z, intensity*255, channel]
  - Camera: parse_image() from sensor.py
      → (H,W,4) BGRA uint8, then drop alpha → (H,W,3) BGR
  - Calibration: get_nuscenes_rt() from utils.py
      → cameras use mode="zxy", LiDAR uses mode=None

Usage:
    python realtime/futr3d_carla.py \
        --config    plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py \
        --ckpt      work_dirs/lidar_0075v_cam_vov/epoch_12.pth \
        --sensors   realtime/calibrated_sensors_rgb.yaml \
        --host      172.21.192.1 \
        --port      2000 \
        --threshold 0.3
"""

import argparse
import time
import threading
import queue
import random
from collections import Counter

import carla
import numpy as np
import yaml
import torch
from pyquaternion import Quaternion

from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmdet3d.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data parsers — exact copies of sensor.py
# ─────────────────────────────────────────────────────────────────────────────

def parse_image(image):
    """Exact copy of sensor.py parse_image() → (H,W,4) BGRA uint8."""
    array = np.ndarray(
        shape=(image.height, image.width, 4),
        dtype=np.uint8, buffer=image.raw_data, order="C")
    return array


def parse_lidar_data(lidar_data):
    """
    Exact copy of sensor.py parse_lidar_data().
    Returns (N, 5) float32: [x, -y, z, intensity*255, channel_idx]
    This is the exact format the model was trained on.
    """
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    pts = pts[np.isfinite(pts).all(axis=1)].copy()

    # CARLA left-handed → nuScenes right-handed
    pts[:, 1] = -pts[:, 1]

    # Scale intensity 0-1 → 0-255
    pts[:, 3] = np.clip(pts[:, 3] * 255.0, 0, 255)

    # Channel index column
    channels = np.zeros(len(pts), dtype=np.float32)
    idx = 0
    for ch in range(lidar_data.channels):
        count = lidar_data.get_point_count(ch)
        channels[idx:idx + count] = ch
        idx += count

    return np.column_stack([pts, channels]).astype(np.float32)  # (N, 5)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Coordinate helpers — exact copies of utils.py
# ─────────────────────────────────────────────────────────────────────────────

def get_intrinsic(fov, image_size_x, image_size_y):
    """Exact copy of utils.py get_intrinsic()."""
    focal = float(image_size_x) / (2.0 * np.tan(float(fov) * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = float(image_size_x) / 2.0
    K[1, 2] = float(image_size_y) / 2.0
    return K


def get_nuscenes_rt(transform, mode=None):
    """Exact copy of utils.py get_nuscenes_rt()."""
    translation = [
        transform.location.x,
        -transform.location.y,
        transform.location.z,
    ]
    if mode == "zxy":
        R1 = np.array([[0,0,1],[1,0,0],[0,1,0]]) @ \
             np.array([[1,0,0],[0,-1,0],[0,0,1]])
    else:
        R1 = np.array([[1,0,0],[0,-1,0],[0,0,1]])

    R2 = np.array(transform.get_matrix())[:3, :3]
    R3 = np.array([[1,0,0],[0,-1,0],[0,0,1]])
    rotation_matrix = R3 @ R2 @ R1
    quat = Quaternion(matrix=rotation_matrix, rtol=1, atol=1).elements.tolist()
    return quat, translation


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Calibration matrix builders
#     Mirrors nuscenes_converter.py _fill_trainval_infos() exactly
# ─────────────────────────────────────────────────────────────────────────────

def build_lidar2img(cam_actor, lidar_actor):
    """
    Exact same computation as nuscenes_converter.py at training time.
    Uses the world transforms of attached sensor actors.
    """
    cam_quat, cam_tr = get_nuscenes_rt(cam_actor.get_transform(), mode="zxy")
    lid_quat, lid_tr = get_nuscenes_rt(lidar_actor.get_transform(), mode=None)

    l2e_r_mat = Quaternion(lid_quat).rotation_matrix
    l2e_t     = np.array(lid_tr)
    e2g_r_mat = np.eye(3)
    e2g_t     = np.zeros(3)
    c2e_r_mat = Quaternion(cam_quat).rotation_matrix
    c2e_t     = np.array(cam_tr)

    # obtain_sensor2top from nuscenes_converter.py
    R = (c2e_r_mat.T @ e2g_r_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (c2e_t @ e2g_r_mat.T + e2g_t) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= (e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
          + l2e_t @ np.linalg.inv(l2e_r_mat).T)

    s2l_rot = R.T
    s2l_tr  = T

    lidar2cam_r  = np.linalg.inv(s2l_rot)
    lidar2cam_t  = s2l_tr @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[ 3, :3] = -lidar2cam_t

    fov = float(cam_actor.attributes["fov"])
    w   = float(cam_actor.attributes["image_size_x"])
    h   = float(cam_actor.attributes["image_size_y"])
    K33 = get_intrinsic(fov, w, h)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = K33

    return (viewpad @ lidar2cam_rt.T).astype(np.float32)


def build_camera2lidar(cam_actor, lidar_actor):
    """sensor2lidar_rotation/translation packed into 4x4."""
    cam_quat, cam_tr = get_nuscenes_rt(cam_actor.get_transform(), mode="zxy")
    lid_quat, lid_tr = get_nuscenes_rt(lidar_actor.get_transform(), mode=None)

    l2e_r_mat = Quaternion(lid_quat).rotation_matrix
    l2e_t     = np.array(lid_tr)
    e2g_r_mat = np.eye(3)
    e2g_t     = np.zeros(3)
    c2e_r_mat = Quaternion(cam_quat).rotation_matrix
    c2e_t     = np.array(cam_tr)

    R = (c2e_r_mat.T @ e2g_r_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (c2e_t @ e2g_r_mat.T + e2g_t) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= (e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
          + l2e_t @ np.linalg.inv(l2e_r_mat).T)

    cam2lidar = np.eye(4, dtype=np.float32)
    cam2lidar[:3, :3] = R.T
    cam2lidar[:3,  3] = T
    return cam2lidar


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sensor buffer — time-window, LiDAR-triggered
#     Does NOT require all sensors to share the same frame ID.
#     Robust to TCP jitter between WSL and Windows CARLA.
# ─────────────────────────────────────────────────────────────────────────────

CAM_ORDER = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]

class SensorBuffer:
    """
    Keeps the latest data from each sensor.
    Emits a complete bundle every time LiDAR fires,
    as long as every camera has contributed at least once.
    """
    def __init__(self, cam_names):
        self.cam_names = cam_names
        self._lock     = threading.Lock()
        self._latest   = {}
        self.ready_q   = queue.Queue(maxsize=2)

    def on_lidar(self, data):
        with self._lock:
            self._latest["lidar"] = data
            self._try_emit()

    def on_camera(self, name, data):
        with self._lock:
            self._latest[name] = data

    def _try_emit(self):
        needed = set(self.cam_names) | {"lidar"}
        if needed.issubset(self._latest.keys()):
            bundle = dict(self._latest)
            try:
                self.ready_q.put_nowait(bundle)
            except queue.Full:
                pass   # inference slower than LiDAR rate — drop frame


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FUTR3D inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FUTR3DInference:
    def __init__(self, config_path, ckpt_path, score_threshold=0.3):
        self.threshold = score_threshold
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg            = Config.fromfile(config_path)
        self.cfg       = cfg

        # Register FUTR3D plugin
        if hasattr(cfg, "plugin") and cfg.plugin:
            import importlib, os as _os
            _parts = (_os.path.dirname(cfg.plugin_dir)
                      if hasattr(cfg, "plugin_dir")
                      else _os.path.dirname(config_path)).split("/")
            _mod = _parts[0]
            for p in _parts[1:]:
                _mod += "." + p
            print(f"Loading plugin: {_mod}")
            importlib.import_module(_mod)

        self.model = build_model(cfg.model, train_cfg=None,
                                 test_cfg=cfg.get("test_cfg"))
        load_checkpoint(self.model, ckpt_path, map_location="cpu")
        self.model.to(self.device).eval()

        # Build pipeline — skip steps that read from disk or need GT
        skip = {
            "LoadMultiViewImageFromFiles",   # we inject images directly
            "LoadPointsFromFile",            # we inject points directly
            "LoadPointsFromMultiSweeps",     # no sweeps in real-time
            "LoadAnnotations3D",
            "ObjectRangeFilter",
            "ObjectNameFilter",
            "PointsRangeFilter",
        }
        self.pipeline    = Compose([s for s in cfg.test_pipeline
                                    if s["type"] not in skip])
        self.class_names = cfg.class_names

        # Image normalisation params (from config) — applied manually
        norm_cfg = None
        for step in cfg.test_pipeline:
            if step["type"] == "NormalizeMultiviewImage":
                norm_cfg = step
                break
        self.norm_mean = np.array(norm_cfg["mean"], dtype=np.float32) if norm_cfg else np.array([103.530, 116.280, 123.675])
        self.norm_std  = np.array(norm_cfg["std"],  dtype=np.float32) if norm_cfg else np.array([57.375,  57.120,  58.395])
        self.to_rgb    = norm_cfg.get("to_rgb", False) if norm_cfg else False

        print(f"Model ready  device={self.device}  threshold={score_threshold}")

    def _normalize_images(self, images_bgr):
        """
        Manually apply NormalizeMultiviewImage.
        Mirrors mmdet3d NormalizeMultiviewImage transform.
        """
        out = []
        for img in images_bgr:
            img = img.astype(np.float32)
            if self.to_rgb:
                img = img[:, :, ::-1].copy()   # BGR → RGB
            img = (img - self.norm_mean) / self.norm_std
            out.append(img)
        return out

    @torch.no_grad()
    def infer(self, lidar_pts, images_bgr, lidar2img_list, cam2lidar_list):
        """
        lidar_pts    : (N,5) float32 — exact output of parse_lidar_data()
        images_bgr   : list of (H,W,3) uint8 BGR — parse_image()[...,:3]

        We skip LoadMultiViewImageFromFiles and LoadPointsFromFile
        and inject the data directly, then run the rest of the pipeline
        (PhotoMetricDistortion is skipped at test time, PadMultiViewImage
        and DefaultFormatBundle3D still run).
        """
        # Normalise images manually (replaces NormalizeMultiviewImage)
        images_norm = self._normalize_images(images_bgr)

        data = dict(
            # Points — injected directly as np.ndarray
            points=lidar_pts,
            # Images — already normalised, stored as list of float32 arrays
            img=images_norm,
            img_fields=["img"],
            # Calibration
            lidar2img=lidar2img_list,
            camera2lidar=cam2lidar_list,
            # Required empty fields
            bbox3d_fields=[], pts_mask_fields=[], pts_seg_fields=[],
            bbox_fields=[], mask_fields=[], seg_fields=[],
            sweeps=[],
            timestamp=time.time(),
            # img_norm_cfg needed by some pipeline steps
            img_norm_cfg=dict(mean=self.norm_mean.tolist(),
                              std=self.norm_std.tolist(),
                              to_rgb=self.to_rgb),
        )
        data    = self.pipeline(data)
        data    = collate([data], samples_per_gpu=1)
        data    = scatter(data, [self.device])[0]
        results = self.model(return_loss=False, rescale=True, **data)

        boxes  = results[0]["pts_bbox"]["boxes_3d"].tensor.cpu().numpy()
        scores = results[0]["pts_bbox"]["scores_3d"].cpu().numpy()
        labels = results[0]["pts_bbox"]["labels_3d"].cpu().numpy()
        mask   = scores > self.threshold
        return boxes[mask], scores[mask], labels[mask]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CARLA debug draw
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "car":                  carla.Color(  0, 255,   0),
    "truck":                carla.Color(255, 165,   0),
    "bus":                  carla.Color(255, 140,   0),
    "pedestrian":           carla.Color(255,   0,   0),
    "motorcycle":           carla.Color(  0,   0, 255),
    "bicycle":              carla.Color(  0, 200, 200),
    "construction_vehicle": carla.Color(128,   0, 128),
    "trailer":              carla.Color(200, 200,   0),
    "barrier":              carla.Color(128, 128, 128),
    "traffic_cone":         carla.Color(255, 165,   0),
}

def draw_boxes(world, boxes, labels, class_names, life_time=0.15):
    for box, label in zip(boxes, labels):
        cx, cy, cz, l, w, h, yaw = box[:7]
        # nuScenes → CARLA (flip y back)
        loc   = carla.Location(x=float(cx), y=float(-cy), z=float(cz))
        ext   = carla.Vector3D(float(l/2), float(w/2), float(h/2))
        rot   = carla.Rotation(yaw=float(np.degrees(-yaw)))
        bbox  = carla.BoundingBox(loc, ext)
        name  = class_names[int(label)] if int(label) < len(class_names) else "?"
        color = CLASS_COLORS.get(name, carla.Color(255, 255, 255))
        world.debug.draw_box(bbox, rot, thickness=0.08,
                             color=color, life_time=life_time)
        world.debug.draw_string(loc, name, color=color, life_time=life_time)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Test scene helpers
# ─────────────────────────────────────────────────────────────────────────────

CAR_BLUEPRINTS   = ["vehicle.audi.a2", "vehicle.audi.tt",
                    "vehicle.toyota.prius", "vehicle.nissan.micra",
                    "vehicle.citroen.c3", "vehicle.seat.leon"]
TRUCK_BLUEPRINTS = ["vehicle.carlamotors.carlacola", "vehicle.ford.ambulance",
                    "vehicle.mercedes.sprinter", "vehicle.volkswagen.t2"]
MOTO_BLUEPRINTS  = ["vehicle.kawasaki.ninja", "vehicle.yamaha.yzf",
                    "vehicle.harley-davidson.low_rider", "vehicle.vespa.zx125"]

def offset_tf(base_tf, dx, dy, dz=0.0):
    return carla.Transform(
        carla.Location(x=base_tf.location.x + dx,
                       y=base_tf.location.y + dy,
                       z=base_tf.location.z + dz),
        carla.Rotation(yaw=base_tf.rotation.yaw))

def try_spawn_vehicle(world, bp_lib, bp_names, transform, label):
    for bp_name in bp_names:
        matches = bp_lib.filter(bp_name)
        if not matches:
            continue
        bp = matches[0]
        for dz in [0.0, 0.3, 0.6, 1.0]:
            actor = world.try_spawn_actor(bp, offset_tf(transform, 0, 0, dz))
            if actor:
                print(f"  [{label}] '{bp_name}'  id={actor.id}")
                return actor
    print(f"  [{label}] WARNING — all spawn attempts failed")
    return None

def try_spawn_pedestrian(world, bp_lib, transform):
    bps = list(bp_lib.filter("walker.pedestrian.*"))
    random.shuffle(bps)
    for bp in bps:
        for dz in [0.5, 1.0, 1.5]:
            actor = world.try_spawn_actor(bp, offset_tf(transform, 0, 0, dz))
            if actor:
                print(f"  [pedestrian] '{bp.id}'  id={actor.id}")
                return actor
    print("  [pedestrian] WARNING — all spawn attempts failed")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world  = client.get_world()
    print(f"Connected  map={world.get_map().name}\n")

    all_actors = []

    try:
        # ── Strip map layers ──────────────────────────────────────────────────
        print("Stripping map layers...")
        for layer in [carla.MapLayer.Buildings, carla.MapLayer.Foliage,
                      carla.MapLayer.ParkedVehicles, carla.MapLayer.Props,
                      carla.MapLayer.StreetLights, carla.MapLayer.Decals]:
            try:
                world.unload_map_layer(layer)
            except Exception:
                pass
        time.sleep(1.0)
        print("Done.\n")

        bp_lib    = world.get_blueprint_library()
        spawn_pts = world.get_map().get_spawn_points()
        if not spawn_pts:
            raise RuntimeError("No spawn points.")
        base_tf = spawn_pts[0]
        r = args.radius

        # ── Spawn ego ─────────────────────────────────────────────────────────
        print("[ego] Spawning...")
        ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
        ego_bp.set_attribute("color", "255,255,255")
        ego = None
        for dz in [0.0, 0.3, 0.6]:
            ego = world.try_spawn_actor(ego_bp, offset_tf(base_tf, 0, 0, dz))
            if ego:
                break
        if ego is None:
            raise RuntimeError("Cannot spawn ego vehicle.")
        ego.set_simulate_physics(False)
        all_actors.append(ego)
        print(f"  [ego] id={ego.id}  {base_tf.location}\n")
        time.sleep(0.5)

        # ── Spawn test actors ─────────────────────────────────────────────────
        print("Spawning test actors...")
        for label, bp_names, dx, dy in [
            ("car",        CAR_BLUEPRINTS,   r,  r),
            ("truck",      TRUCK_BLUEPRINTS,  r, -r),
            ("motorcycle", MOTO_BLUEPRINTS,  -r,  r),
        ]:
            a = try_spawn_vehicle(world, bp_lib, bp_names,
                                  offset_tf(base_tf, dx, dy), label)
            if a:
                a.set_simulate_physics(False)
                all_actors.append(a)
            time.sleep(0.3)

        ped = try_spawn_pedestrian(world, bp_lib, offset_tf(base_tf, -r, -r))
        if ped:
            ped.set_simulate_physics(False)
            all_actors.append(ped)
        time.sleep(0.3)

        print(f"\nTest scene: {len(all_actors)} actors total\n")

        # ── Load sensor YAML ──────────────────────────────────────────────────
        with open(args.sensors) as f:
            sensor_cfg = yaml.safe_load(f)

        sensor_list = [s for s in sensor_cfg["sensors"]
                       if s["bp_name"] in ("sensor.camera.rgb",
                                           "sensor.lidar.ray_cast")]
        # LiDAR first
        sensor_list.sort(key=lambda s: 0 if "lidar" in s["bp_name"] else 1)

        cam_actors  = {}
        lidar_actor = None
        sensors     = []

        # ── Attach sensors one at a time ──────────────────────────────────────
        print("Attaching sensors...")
        for s in sensor_list:
            bp = bp_lib.find(s["bp_name"])
            if bp is None:
                print(f"  {s['name']} — not found, skipping")
                continue

            if "camera" in s["bp_name"]:
                bp.set_attribute("image_size_x", str(args.cam_width))
                bp.set_attribute("image_size_y", str(args.cam_height))
                bp.set_attribute("sensor_tick",  str(args.sensor_tick))
                # preserve FOV from YAML
                opts = s.get("options") or {}
                if "fov" in opts:
                    bp.set_attribute("fov", str(opts["fov"]))

            if "lidar" in s["bp_name"]:
                bp.set_attribute("points_per_second", "560000")
                bp.set_attribute("sensor_tick",       str(args.sensor_tick))
                bp.set_attribute("channels",          "32")
                bp.set_attribute("range",             "80")
                bp.set_attribute("upper_fov",         "10")
                bp.set_attribute("lower_fov",         "-30")
                bp.set_attribute("dropoff_general_rate",   "0.2")
                bp.set_attribute("dropoff_intensity_limit", "0.6")
                bp.set_attribute("dropoff_zero_intensity",  "0.2")

            tf = carla.Transform(carla.Location(**s["location"]),
                                 carla.Rotation(**s["rotation"]))
            print(f"  {s['name']}...", end=" ", flush=True)
            actor = world.try_spawn_actor(bp, tf, attach_to=ego)
            if actor is None:
                print("FAILED")
                continue
            print(f"ok  id={actor.id}")
            sensors.append(actor)
            all_actors.append(actor)

            if "camera" in s["bp_name"]:
                cam_actors[s["name"]] = actor
            elif "lidar" in s["bp_name"]:
                lidar_actor = actor

            time.sleep(args.spawn_delay)

        if lidar_actor is None:
            raise RuntimeError("LiDAR failed to attach.")
        present_cams = [c for c in CAM_ORDER if c in cam_actors]
        if not present_cams:
            raise RuntimeError("No cameras attached.")

        print(f"\nCameras : {present_cams}")
        print(f"LiDAR   : id={lidar_actor.id}")
        print(f"Res     : {args.cam_width}x{args.cam_height}  tick={args.sensor_tick}s\n")

        # ── Calibration matrices ──────────────────────────────────────────────
        lidar2img_list = [build_lidar2img(cam_actors[c], lidar_actor)
                          for c in present_cams]
        cam2lidar_list = [build_camera2lidar(cam_actors[c], lidar_actor)
                          for c in present_cams]

        # ── Start listening ───────────────────────────────────────────────────
        buf = SensorBuffer(present_cams)
        lidar_actor.listen(buf.on_lidar)
        for name, actor in cam_actors.items():
            actor.listen(lambda data, n=name: buf.on_camera(n, data))

        # ── Load model ────────────────────────────────────────────────────────
        print(f"Waiting {args.warmup}s for sensors to warm up...")
        time.sleep(args.warmup)
        print("\nLoading FUTR3D model...")
        futr3d = FUTR3DInference(args.config, args.ckpt,
                                 score_threshold=args.threshold)

        # ── Inference loop ────────────────────────────────────────────────────
        print("\nReal-time inference started — Ctrl+C to stop\n")
        frame_count = 0
        while True:
            try:
                frame = buf.ready_q.get(timeout=2.0)
            except queue.Empty:
                print("Waiting for sensor frame...")
                continue

            t0 = time.time()

            # ── Parse data using same functions as training pipeline ───────────
            lidar_pts = parse_lidar_data(frame["lidar"])   # (N,5) float32
            images    = [parse_image(frame[c])[:, :, :3]   # (H,W,3) BGR
                         for c in present_cams]

            try:
                boxes, scores, labels = futr3d.infer(
                    lidar_pts, images, lidar2img_list, cam2lidar_list)
            except Exception as e:
                print(f"Inference error: {e}")
                import traceback; traceback.print_exc()
                continue

            draw_boxes(world, boxes, labels, futr3d.class_names,
                       life_time=args.box_lifetime)

            dt     = time.time() - t0
            frame_count += 1
            counts  = Counter(futr3d.class_names[int(l)] for l in labels)
            det_str = "  ".join(f"{k}:{v}" for k, v in counts.items())
            print(f"[{frame_count:04d}]  det={len(boxes):3d}  "
                  f"lat={dt*1000:.0f}ms  {det_str if det_str else 'none'}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nFatal: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("\nCleaning up...")
        for a in reversed(all_actors):
            try:
                if hasattr(a, "stop"):
                    a.stop()
                a.destroy()
            except Exception:
                pass
        for layer in [carla.MapLayer.Buildings, carla.MapLayer.Foliage]:
            try:
                world.load_map_layer(layer)
            except Exception:
                pass
        print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--sensors",     default="calibrated_sensors_rgb.yaml")
    parser.add_argument("--host",        default="172.21.192.1")
    parser.add_argument("--port",        default=2000,  type=int)
    parser.add_argument("--threshold",   default=0.3,   type=float)
    parser.add_argument("--radius",      default=4.0,   type=float)
    parser.add_argument("--cam-width",   default=800,   type=int,   dest="cam_width")
    parser.add_argument("--cam-height",  default=450,   type=int,   dest="cam_height")
    parser.add_argument("--sensor-tick", default=0.2,   type=float, dest="sensor_tick")
    parser.add_argument("--spawn-delay", default=1.5,   type=float, dest="spawn_delay")
    parser.add_argument("--warmup",      default=4.0,   type=float)
    parser.add_argument("--box-lifetime",default=0.15,  type=float, dest="box_lifetime")
    args = parser.parse_args()
    main(args)