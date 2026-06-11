"""
Record the CARLA scene ALONE (no 3D detection) as a smooth third-person
driving video. Reuses bootstrap_client from realtime_detect.py so the scene
matches (same random_seed=0, same map, same traffic). Runs fast because the
heavy sensors + model are off.

Run from the futr3d root with CARLA up:
    python realtime/record_scene.py

NOTE ON ALIGNMENT: same seed reproduces the scene closely, but CARLA sync mode
is not bit-perfect run-to-run, so this clean video may drift from the detection
video over time. Good for side-by-side / picture-in-picture editing. For exact
box-overlay onto clean footage, record clean + boxed in ONE run instead.
"""

import queue

import carla
import cv2
import numpy as np

# reuse the IDENTICAL scene + helpers from the detection script
from realtime_detect import bootstrap_client, parse_image, chase_transform

# --- config ------------------------------------------------------------------
SECONDS  = 30                     # sim-seconds to record
SIM_FPS  = int(round(1 / 0.083333))   # 12 -> real-time-paced playback
OUT      = "scene_drive.mp4"
CHASE_W, CHASE_H, CHASE_FOV = 1280, 720, 90


def main():
    client = bootstrap_client()                  # same scene as the detection run
    world = client.world
    ego = client.ego_vehicle.get_actor()
    spectator = world.get_spectator()

    # turn off the model's heavy sensors — we only need a chase view here
    for s in client.sensors:
        a = s.get_actor()
        if a is not None:
            try:
                a.stop()
            except RuntimeError:
                pass

    # dedicated chase camera, attached to the ego (rides with it automatically)
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(CHASE_W))
    bp.set_attribute("image_size_y", str(CHASE_H))
    bp.set_attribute("fov", str(CHASE_FOV))
    chase_tf = carla.Transform(carla.Location(x=-6.0, z=3.0),
                               carla.Rotation(pitch=-15.0))   # behind + above, ego-local
    chase = world.spawn_actor(bp, chase_tf, attach_to=ego)
    q = queue.Queue()
    chase.listen(q.put)

    writer = None
    n_ticks = int(SECONDS / 0.083333)
    try:
        for i in range(n_ticks):
            frame = world.tick()
            spectator.set_transform(chase_transform(ego.get_transform()))  # live preview
            while True:                                  # frame-matched grab
                img = q.get(timeout=10.0)
                if img.frame == frame:
                    break
            bgr = parse_image(img)[:, :, :3]
            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"),
                                         SIM_FPS, (w, h))
            writer.write(bgr)
            if i % SIM_FPS == 0:
                print(f"recorded {i}/{n_ticks} frames")
    finally:
        if writer is not None:
            writer.release()
        try:
            chase.stop()
            chase.destroy()
        except RuntimeError:
            pass
        client.destroy_scene()
        client.destroy_world()
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()