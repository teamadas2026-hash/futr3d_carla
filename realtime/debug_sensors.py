"""
debug_sensors.py
----------------
Diagnoses why SensorBuffer never fills by printing each sensor's
frame ID as data arrives. Run this INSTEAD of futr3d_carla.py to
see what frame IDs each sensor produces.

Usage:
    python realtime/debug_sensors.py \
        --sensors realtime/calibrated_sensors_rgb.yaml \
        --host 172.21.192.1 --port 2000
"""

import argparse
import time
import threading
import yaml
import carla

lock = threading.Lock()
frame_log = {}   # sensor_name -> list of frame_ids

def make_callback(name):
    def cb(data):
        with lock:
            if name not in frame_log:
                frame_log[name] = []
            frame_log[name].append(data.frame)
            # Print live
            print(f"  {name:20s}  frame={data.frame}")
    return cb

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world  = client.get_world()
    print(f"Connected  map={world.get_map().name}\n")

    bp_lib    = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()

    # Spawn ego
    ego_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
    ego    = None
    for sp in spawn_pts[:5]:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego: break
    if ego is None:
        raise RuntimeError("Cannot spawn ego")
    ego.set_simulate_physics(False)
    print(f"Ego spawned id={ego.id}\n")

    with open(args.sensors) as f:
        sensor_cfg = yaml.safe_load(f)

    sensor_list = [s for s in sensor_cfg["sensors"]
                   if s["bp_name"] in ("sensor.camera.rgb",
                                       "sensor.lidar.ray_cast")]
    sensor_list.sort(key=lambda s: 0 if "lidar" in s["bp_name"] else 1)

    attached = []
    print("Attaching sensors...")
    for s in sensor_list:
        bp = bp_lib.find(s["bp_name"])
        if bp is None: continue
        if "camera" in s["bp_name"]:
            bp.set_attribute("image_size_x", "400")
            bp.set_attribute("image_size_y", "225")
            bp.set_attribute("sensor_tick",  "0.5")
            if s.get("options") and "fov" in s["options"]:
                bp.set_attribute("fov", str(s["options"]["fov"]))
        if "lidar" in s["bp_name"]:
            bp.set_attribute("points_per_second", "100000")
            bp.set_attribute("sensor_tick", "0.5")
            bp.set_attribute("channels", "16")
        tf    = carla.Transform(carla.Location(**s["location"]),
                                carla.Rotation(**s["rotation"]))
        actor = world.try_spawn_actor(bp, tf, attach_to=ego)
        if actor is None:
            print(f"  {s['name']} FAILED")
            continue
        actor.listen(make_callback(s["name"]))
        attached.append(actor)
        print(f"  {s['name']} ok id={actor.id}")
        time.sleep(1.0)

    print(f"\nListening for 10 seconds — watch frame IDs...\n")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass

    try:
        # Analysis
        print("\n─── Frame ID analysis ───")
        for name, frames in frame_log.items():
            print(f"  {name:20s}  frames received: {len(frames)}"
                  + (f"  ids: {frames[:5]}" if frames else "  NONE"))

        if frame_log:
            all_frames = set()
            for frames in frame_log.values():
                all_frames.update(frames)
            print(f"\n  Unique frame IDs across all sensors: {len(all_frames)}")

            sensor_names = list(frame_log.keys())
            complete = 0
            for fid in all_frames:
                if all(fid in frame_log.get(n, []) for n in sensor_names):
                    complete += 1
            print(f"  Frames where ALL sensors fired together: {complete}")
            if complete == 0:
                print("\n  No complete frames — sensors are on different ticks!")
                print("  The time-window buffer in futr3d_carla.py handles this.")
    finally:
        for a in reversed(attached):
            try:
                a.stop()
                a.destroy()
            except Exception:
                pass
        try:
            ego.destroy()
        except Exception:
            pass
        print("\nCleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", default="calibrated_sensors_rgb.yaml")
    parser.add_argument("--host",    default="172.21.192.1")
    parser.add_argument("--port",    default=2000, type=int)
    args = parser.parse_args()
    main(args)