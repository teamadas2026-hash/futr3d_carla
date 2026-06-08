import os
import numpy as np
import carla
from .actor import Actor
import queue

#specific carla's semantic label can be found in LibCarla/source/carla/rpc

carla_to_nuscenes_map = {
    0: 0,    # None → noise
    1: 24,   # Roads → flat.driveable_surface
    2: 26,   # Sidewalks → flat.sidewalk
    3: 28,   # Buildings → static.manmade
    4: 28,   # Walls → static.manmade
    5: 28,   # Fences → static.manmade
    6: 28,   # Poles → static.manmade
    7: 28,   # TrafficLight → static.manmade
    8: 28,   # TrafficSigns → static.manmade
    9: 30,   # Vegetation → static.vegetation
    10: 27,  # Terrain → flat.terrain
    11: 29,  # Sky → static.other
    12: 2,   # Pedestrians → human.pedestrian.adult
    13: 5,   # Rider → human.pedestrian.personal_mobility
    14: 17,  # Car → vehicle.car
    15: 23,  # Truck → vehicle.truck
    16: 16,  # Bus → vehicle.bus.rigid
    17: 22,  # Train → vehicle.trailer
    18: 21,  # Motorcycle → vehicle.motorcycle
    19: 14,  # Bicycle → vehicle.bicycle
    20: 28,  # Static → static.manmade
    21: 3,   # Dynamic → human.pedestrian.child
    22: 29,  # Other → static.other
    23: 25,  # Water → flat.other
    24: 24,  # RoadLines → flat.driveable_surface
    25: 27,  # Ground → flat.terrain
    26: 28,  # Bridge → static.manmade
    27: 25,  # RailTrack → flat.other
    28: 28,  # GuardRail → static.manmade
}
def parse_image(image):
    array = np.ndarray(
            shape=(image.height, image.width, 4),
            dtype=np.uint8, buffer=image.raw_data,order="C")
    return array

def parse_lidar_data(lidar_data):
    # Read raw buffer as float32 (x, y, z, intensity) — avoids float64 conversion
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)

    # Filter CARLA no-hit sentinel values
    pts = pts[np.isfinite(pts).all(axis=1)].copy()

    # CARLA left-handed → nuScenes right-handed coordinate system
    pts[:, 1] = -pts[:, 1]

    # Scale intensity [0.0, 1.0] → [0.0, 255.0] to match nuScenes
    pts[:, 3] = np.clip(pts[:, 3] * 255.0, 0, 255)

    # Build channel index column (replaces the manual loop)
    channels = np.zeros(len(pts), dtype=np.float32)
    idx = 0
    for ch in range(lidar_data.channels):
        count = lidar_data.get_point_count(ch)
        channels[idx:idx + count] = ch
        idx += count

    return np.column_stack([pts, channels])  # (N, 5) float32


def parse_semlidar_data(semlidar_data):
    # ✅ No changes needed — uint8 tags are correct
    tags = []
    for idx, data in enumerate(semlidar_data):
        tag = data.object_tag
        tags.append(carla_to_nuscenes_map[tag])
    return np.array(tags, dtype=np.uint8)

# def parse_radar_data(radar_data):
#     points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).copy()
#     return points

def parse_radar_data(radar_data, ego_velocity_xy=None, sensor_yaw_rad=0.0):
    """
    Convert CARLA RadarMeasurement -> (N, 18) float32 array in nuScenes
    RADAR_FRONT field order:
      x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
      is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
      pdh0, vx_rms, vy_rms

    Args:
        radar_data: carla.RadarMeasurement.
        ego_velocity_xy: optional (vx_world, vy_world) of the ego vehicle in m/s.
                         If given, vx_comp/vy_comp are ego-motion compensated.
        sensor_yaw_rad: yaw of this radar relative to the ego, in radians.
    """
    raw = np.frombuffer(radar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
    if len(raw) == 0:
        return np.zeros((0, 18), dtype=np.float32)

    # CARLA raw layout is [velocity, azimuth, altitude, depth]
    vel, az, alt, depth = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

    # Spherical -> Cartesian, then CARLA (left-handed) -> nuScenes (right-handed)
    x =  depth * np.cos(alt) * np.cos(az)
    y = -depth * np.cos(alt) * np.sin(az)            # negate y for handedness flip
    z =  np.zeros_like(x, dtype=np.float32)          # match nuScenes convention (z = 0)

    # Radial Doppler decomposed onto x/y in sensor frame
    vx = (vel * np.cos(az)).astype(np.float32)
    vy = (-vel * np.sin(az)).astype(np.float32)

    # Ego-motion compensated radial velocity
    if ego_velocity_xy is not None:
        ego_vx_w, ego_vy_w = ego_velocity_xy
        c, s = np.cos(-sensor_yaw_rad), np.sin(-sensor_yaw_rad)
        ego_vx_s = c * ego_vx_w - s * ego_vy_w
        ego_vy_s = s * ego_vx_w + c * ego_vy_w
        radial_ego = ego_vx_s * np.cos(az) + ego_vy_s * np.sin(az)
        vel_comp = vel + radial_ego
        vx_comp = (vel_comp * np.cos(az)).astype(np.float32)
        vy_comp = (-vel_comp * np.sin(az)).astype(np.float32)
    else:
        vx_comp, vy_comp = vx.copy(), vy.copy()

    n = len(raw)
    # dyn_prop: keep a simple thresholded mapping aligned with nuScenes conventions
    speed = np.abs(vel)
    dyn_prop = np.where(speed < 0.2, 3,
                np.where(vel < 0.0, 0, 2)).astype(np.float32)

    ids              = np.arange(n, dtype=np.float32)
    rcs              = np.zeros(n, dtype=np.float32)       # CARLA gives no RCS
    is_quality_valid = np.ones(n, dtype=np.float32)
    ambig_state      = np.full(n, 3, dtype=np.float32)     # 3 = unambiguous
    x_rms            = np.zeros(n, dtype=np.float32)
    y_rms            = np.zeros(n, dtype=np.float32)
    invalid_state    = np.zeros(n, dtype=np.float32)
    pdh0             = np.ones(n, dtype=np.float32)
    vx_rms           = np.zeros(n, dtype=np.float32)
    vy_rms           = np.zeros(n, dtype=np.float32)

    return np.column_stack([
        x, y, z, dyn_prop, ids, rcs, vx, vy, vx_comp, vy_comp,
        is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
        pdh0, vx_rms, vy_rms,
    ]).astype(np.float32)

def write_nuscenes_radar_pcd(points, path):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 18:
        raise ValueError("Radar points must have shape (N, 18)")

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    n = points.shape[0]
    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("dyn_prop", "u1"),
        ("id", "<u2"),
        ("rcs", "<f4"),
        ("vx", "<f4"),
        ("vy", "<f4"),
        ("vx_comp", "<f4"),
        ("vy_comp", "<f4"),
        ("is_quality_valid", "u1"),
        ("ambig_state", "u1"),
        ("x_rms", "u1"),
        ("y_rms", "u1"),
        ("invalid_state", "u1"),
        ("pdh0", "u1"),
        ("vx_rms", "u1"),
        ("vy_rms", "u1"),
    ], align=False)

    assert dtype.itemsize == 43

    packed = np.zeros(n, dtype=dtype)
    packed["x"] = points[:, 0].astype(np.float32)
    packed["y"] = points[:, 1].astype(np.float32)
    packed["z"] = points[:, 2].astype(np.float32)
    packed["dyn_prop"] = np.rint(points[:, 3]).astype(np.uint8)
    packed["id"] = np.rint(points[:, 4]).astype(np.uint16)
    packed["rcs"] = points[:, 5].astype(np.float32)
    packed["vx"] = points[:, 6].astype(np.float32)
    packed["vy"] = points[:, 7].astype(np.float32)
    packed["vx_comp"] = points[:, 8].astype(np.float32)
    packed["vy_comp"] = points[:, 9].astype(np.float32)
    packed["is_quality_valid"] = np.rint(points[:, 10]).astype(np.uint8)
    packed["ambig_state"] = np.rint(points[:, 11]).astype(np.uint8)
    packed["x_rms"] = np.rint(points[:, 12]).astype(np.uint8)
    packed["y_rms"] = np.rint(points[:, 13]).astype(np.uint8)
    packed["invalid_state"] = np.rint(points[:, 14]).astype(np.uint8)
    packed["pdh0"] = np.rint(points[:, 15]).astype(np.uint8)
    packed["vx_rms"] = np.rint(points[:, 16]).astype(np.uint8)
    packed["vy_rms"] = np.rint(points[:, 17]).astype(np.uint8)

    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms",
        "SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1",
        "TYPE F F F I I F F F F F I I I I I I I I",
        "COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA binary",
        "",
    ])

    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(packed.tobytes())
        handle.write(b"\n")
    
# def parse_data(data):
#     if isinstance(data,carla.Image):
#         return parse_image(data)
#     elif isinstance(data,carla.RadarMeasurement):
#         return parse_radar_data(data)
#     elif isinstance(data,carla.LidarMeasurement):
#         return parse_lidar_data(data)
#     elif isinstance(data, carla.SemanticLidarMeasurement):
#         return parse_semlidar_data(data)

def get_data_shape(data):
    if isinstance(data,carla.Image):
        return data.height,data.width
    else:
        return 0,0
class Sensor(Actor):
    def __init__(self, name, **args):
        super().__init__(**args)
        self.name = name
        self.data_list = []
        self.data_queue = queue.Queue()
        self.vehicle = None
    def get_data_list(self):
        return self.data_list
    def add_vehicle(self, vehicle):
        self.vehicle=vehicle
    def set_actor(self, id):
        super().set_actor(id)
        self.actor.listen(self.add_data)
    
    def spawn_actor(self):
        super().spawn_actor()
        self.actor.listen(self.add_data)

    def get_last_data(self):
        if self.data_list:
            return self.data_list[-1]
        else:
            return None
            
    def add_data(self,data):
        try:
            if self.vehicle is not None:
                self.data_list.append((self.actor.parent.get_transform(),data,self.vehicle.get_transform()))
            else:
                self.data_list.append((self.actor.parent.get_transform(),data))
        except:
            self.data_list.append((self.actor.parent.get_transform(),data))

    def get_transform(self):
        return self.actor.get_transform()