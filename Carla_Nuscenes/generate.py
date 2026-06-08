import argparse
import os
import copy
import yaml
import itertools
import random
from yamlinclude import YamlIncludeConstructor
from carla_nuscenes.generator import Generator

# ── Register !include support ──────────────────────────────────
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

# ── Diversity definitions ──────────────────────────────────────
# These are the dimensions we vary across scenes to maximise
# domain coverage for fine-tuning. Edit counts/values freely.

TOWNS = [
    "Town01",       # simple loop, rural feel
    "Town02",       # small town, tight streets
    "Town03",       # urban, complex junctions, highway
    "Town05",       # grid layout, multilane roads
    "Town06",       # highway with exits (if available)
    "Town10HD_Opt", # high-detail downtown — _Opt suffix required by CARLA
]

# Maps that need the _Opt suffix to load correctly in CARLA.
# Add any others here if you encounter loading errors.
OPT_MAPS = {"Town10HD_Opt", "Town06_Opt", "Town07_Opt"}

# Each weather preset is a dict that maps directly to
# carla.WeatherParameters — values from CARLA docs.
# sun_altitude_angle: >0=day, ~0=sunset, <0=night
WEATHER_PRESETS = {
    "ClearNoon": {
        "cloudiness": 0, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_azimuth_angle": 180, "sun_altitude_angle": 75,
        "fog_density": 0, "fog_distance": 0, "wetness": 0,
        "fog_falloff": 0.2, "scattering_intensity": 0,
        "mie_scattering_scale": 0, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "CloudyNoon": {
        "cloudiness": 70, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 30, "sun_azimuth_angle": 180, "sun_altitude_angle": 75,
        "fog_density": 5, "fog_distance": 0, "wetness": 0,
        "fog_falloff": 0.2, "scattering_intensity": 0,
        "mie_scattering_scale": 0, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "WetNoon": {
        "cloudiness": 50, "precipitation": 0, "precipitation_deposits": 60,
        "wind_intensity": 20, "sun_azimuth_angle": 180, "sun_altitude_angle": 75,
        "fog_density": 5, "fog_distance": 0, "wetness": 80,
        "fog_falloff": 0.2, "scattering_intensity": 0,
        "mie_scattering_scale": 0, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "HardRainNoon": {
        "cloudiness": 100, "precipitation": 80, "precipitation_deposits": 90,
        "wind_intensity": 70, "sun_azimuth_angle": 180, "sun_altitude_angle": 60,
        "fog_density": 20, "fog_distance": 0, "wetness": 100,
        "fog_falloff": 0.2, "scattering_intensity": 0,
        "mie_scattering_scale": 0, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "ClearSunset": {
        "cloudiness": 15, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_azimuth_angle": 270, "sun_altitude_angle": 5,
        "fog_density": 0, "fog_distance": 0, "wetness": 0,
        "fog_falloff": 0.2, "scattering_intensity": 0.5,
        "mie_scattering_scale": 0.1, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "ClearMorning": {
        "cloudiness": 5, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 8, "sun_azimuth_angle": 90, "sun_altitude_angle": 25,
        "fog_density": 8, "fog_distance": 20, "wetness": 5,
        "fog_falloff": 0.3, "scattering_intensity": 0.2,
        "mie_scattering_scale": 0.05, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "NormalDay": {
        "cloudiness": 25, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 15, "sun_azimuth_angle": 180, "sun_altitude_angle": 55,
        "fog_density": 2, "fog_distance": 0, "wetness": 10,
        "fog_falloff": 0.2, "scattering_intensity": 0.1,
        "mie_scattering_scale": 0.02, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "ClearNight": {
        "cloudiness": 10, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 5, "sun_azimuth_angle": 180, "sun_altitude_angle": -90,
        "fog_density": 0, "fog_distance": 0, "wetness": 0,
        "fog_falloff": 0.2, "scattering_intensity": 0,
        "mie_scattering_scale": 0, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "FoggyMorning": {
        "cloudiness": 60, "precipitation": 0, "precipitation_deposits": 20,
        "wind_intensity": 5, "sun_azimuth_angle": 90, "sun_altitude_angle": 15,
        "fog_density": 60, "fog_distance": 10, "wetness": 30,
        "fog_falloff": 0.5, "scattering_intensity": 0.2,
        "mie_scattering_scale": 0.3, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0,
    },
    "DustStorm": {
        "cloudiness": 40, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 90, "sun_azimuth_angle": 180, "sun_altitude_angle": 45,
        "fog_density": 30, "fog_distance": 0, "wetness": 0,
        "fog_falloff": 0.2, "scattering_intensity": 0.8,
        "mie_scattering_scale": 0.5, "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 80,
    },
}

# Traffic density presets — map to traffic block in scene config
# (cars, trucks, bikes, vans, walkers)
TRAFFIC_PRESETS = {
    "sparse":  {"cars": 10, "trucks": 2,  "bikes": 2,  "vans": 2,  "walkers": 30},  # was 5
    "medium":  {"cars": 30, "trucks": 6,  "bikes": 6,  "vans": 6,  "walkers": 60},  # was 20
    "dense":   {"cars": 60, "trucks": 12, "bikes": 10, "vans": 10, "walkers": 100}, # was 40
}

# Ego speed presets — expressed as vehicle_percentage_speed_difference
# negative = faster than speed limit, positive = slower
# Stored in scene config so generate_custom_scene can apply them.
EGO_SPEED_PRESETS = {
    "slow":   30,    # 30% below speed limit
    "medium": -20,   # 20% above speed limit  (current default)
    "fast":   -50,   # 50% above speed limit
}


# ── Argument parsing ───────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="CARLA Dataset Generator — diverse scene collection for fine-tuning"
)
parser.add_argument("--mode", type=str, required=True,
                    choices=["rgb", "segmentation"])
parser.add_argument("--image-size", type=int, nargs=2,
                    metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--count", type=int, default=5,
                    help="Key frames per scene (default: 5)")
parser.add_argument("--random-seed", type=int, default=0,
                    help="Global random seed")
parser.add_argument("--root", type=str,
                    help="Override dataset root path")

# Diversity controls
parser.add_argument("--towns", type=str, nargs="+",
                    default=["Town01", "Town03", "Town05", "Town10HD_opt"],
                    help="CARLA maps to iterate over")
parser.add_argument("--weathers", type=str, nargs="+",
                    default=["ClearNoon", "CloudyNoon", "WetNoon",
                             "ClearSunset", "ClearNight"],
                    choices=list(WEATHER_PRESETS.keys()),
                    help="Weather presets to use")
parser.add_argument("--traffic", type=str, nargs="+",
                    default=["sparse", "medium", "dense"],
                    choices=list(TRAFFIC_PRESETS.keys()),
                    help="Traffic density presets")
parser.add_argument("--ego-speeds", type=str, nargs="+",
                    default=["slow", "medium", "fast"],
                    choices=list(EGO_SPEED_PRESETS.keys()),
                    help="Ego vehicle speed presets")
parser.add_argument("--scenes-per-combo", type=int, default=1,
                    help="Scenes to generate per (town, weather, traffic, speed) combo")
parser.add_argument("--max-scenes", type=int, default=None,
                    help="Hard cap on total scenes (random sample if needed)")
parser.add_argument("--shuffle-combos", action="store_true", default=True,
                    help="Shuffle the combination order (default: True)")

args = parser.parse_args()

# ── Load base config ───────────────────────────────────────────
config_map = {
    "rgb":           "./configs/config_rgb.yaml",
    "segmentation":  "./configs/config_segmentation.yaml",
}
with open(config_map[args.mode], "r") as f:
    base_config = yaml.load(f.read(), Loader=yaml.FullLoader)

# ── Build diversity combinations ──────────────────────────────
combos = list(itertools.product(
    args.towns,
    args.weathers,
    args.traffic,
    args.ego_speeds,
))

# Optionally repeat each combo
combos = combos * args.scenes_per_combo

if args.shuffle_combos:
    rng = random.Random(args.random_seed)
    rng.shuffle(combos)

if args.max_scenes is not None:
    combos = combos[:args.max_scenes]

print(f"\n{'='*60}")
print(f"Diversity generation plan")
print(f"{'='*60}")
print(f"  Towns:         {args.towns}")
print(f"  Weathers:      {args.weathers}")
print(f"  Traffic:       {args.traffic}")
print(f"  Ego speeds:    {args.ego_speeds}")
print(f"  Scenes/combo:  {args.scenes_per_combo}")
print(f"  Total scenes:  {len(combos)}")
print(f"  Key frames ea: {args.count}")
print(f"  Total frames:  {len(combos) * args.count}")
print(f"{'='*60}\n")


def apply_image_size(config, width, height):
    """Apply --image-size override to all camera sensors in all scenes."""
    for world in config.get("worlds", []):
        for capture in world.get("captures", []):
            for scene in capture.get("scenes", []):
                for sensor in scene["calibrated_sensors"].get("sensors", []):
                    if sensor.get("bp_name") in (
                        "sensor.camera.rgb",
                        "sensor.camera.semantic_segmentation",
                    ):
                        opts = sensor.setdefault("options", {})
                        opts["image_size_x"] = str(width)
                        opts["image_size_y"] = str(height)
    return config


def build_scene_config(base_scene, town, weather_name, traffic_name,
                       ego_speed_name, scene_idx, key_frames):
    """
    Clone base_scene and inject all diversity parameters.
    Returns the modified scene config dict.
    """
    scene = copy.deepcopy(base_scene)

    # ── Key frames ──────────────────────────────────────────
    scene["count"] = key_frames

    # ── Weather ─────────────────────────────────────────────
    scene["weather_mode"] = "custom"
    scene["weather"] = copy.deepcopy(WEATHER_PRESETS[weather_name])

    # ── Traffic ─────────────────────────────────────────────
    # Overwrite with the chosen density preset.
    # This also means the hardcoded vehicles.yaml / walkers.yaml
    # entries are ignored — _get_traffic_config() takes priority
    # when a "traffic" block is present, which is correct behaviour.
    scene["traffic"] = copy.deepcopy(TRAFFIC_PRESETS[traffic_name])

    # ── Ego vehicle location ─────────────────────────────────
    # The base config has a hardcoded Town10HD spawn point.
    # For any other map those coords are invalid.
    # Setting location/rotation to None signals generate_custom_scene
    # to use self.spawn_points[self.count] (the map's own spawn points).
    # We keep bp_name and other fields intact.
    if "ego_vehicle" in scene:
        scene["ego_vehicle"]["location"] = None
        scene["ego_vehicle"]["rotation"] = None

    # ── Clear hardcoded vehicles / walkers ──────────────────
    # vehicles.yaml and walkers.yaml have Town10HD-specific coords.
    # Since traffic: is set above, these lists are never used anyway
    # (_get_traffic_config takes priority), but clearing them avoids
    # confusing errors if the Generator iterates them first.
    scene["vehicles"] = []
    scene["walkers"]  = []

    # ── Ego speed ───────────────────────────────────────────
    # Stored here; applied in client.generate_custom_scene via
    # trafficmanager.vehicle_percentage_speed_difference
    scene["ego_speed_diff"] = EGO_SPEED_PRESETS[ego_speed_name]

    # ── Scene description (useful for debugging) ────────────
    scene["description"] = (
        f"{town}|{weather_name}|traffic:{traffic_name}|speed:{ego_speed_name}"
        f"|scene:{scene_idx}"
    )

    return scene


def build_world_config(base_config, town, scenes_for_this_town):
    """
    Find the matching world block for `town` in base_config, or clone
    the first world block and override the map name.
    Returns a single world config dict with the given scenes injected.
    """
    # Try to find an existing world block for this town
    world = None
    for w in base_config.get("worlds", []):
        if w.get("map_name") == town:
            world = copy.deepcopy(w)
            break

    # Fall back to first world block with map_name replaced
    if world is None:
        world = copy.deepcopy(base_config["worlds"][0])
        world["map_name"] = town

    # Replace all scenes in all captures with our generated scenes
    # (keep first capture as template, discard the rest)
    if world.get("captures"):
        capture_template = copy.deepcopy(world["captures"][0])
    else:
        capture_template = {"scenes": []}

    capture_template["scenes"] = scenes_for_this_town
    world["captures"] = [capture_template]
    return world


# ── Group combos by town (one world load per town) ────────────
from collections import defaultdict
town_combos = defaultdict(list)
for combo in combos:
    town, weather, traffic, ego_speed = combo
    town_combos[town].append((weather, traffic, ego_speed))

# ── Build the full config for this run ────────────────────────
# Get one representative base scene to use as template
base_scene_template = (
    base_config["worlds"][0]["captures"][0]["scenes"][0]
)

run_config = copy.deepcopy(base_config)
run_config["worlds"] = []

scene_counter = 0
for town, scene_combos in town_combos.items():
    scenes_for_town = []
    for weather_name, traffic_name, ego_speed_name in scene_combos:
        scene = build_scene_config(
            base_scene=base_scene_template,
            town=town,
            weather_name=weather_name,
            traffic_name=traffic_name,
            ego_speed_name=ego_speed_name,
            scene_idx=scene_counter,
            key_frames=args.count,
        )
        scenes_for_town.append(scene)
        scene_counter += 1
        print(
            f"  [{scene_counter:03d}] {town:12s}  "
            f"weather={weather_name:15s}  "
            f"traffic={traffic_name:8s}  "
            f"speed={ego_speed_name}"
        )

    world_cfg = build_world_config(base_config, town, scenes_for_town)
    run_config["worlds"].append(world_cfg)

# Apply image size override if requested
if args.image_size:
    w, h = args.image_size
    run_config = apply_image_size(run_config, w, h)

# Apply root override
if args.root:
    run_config["dataset"]["root"] = args.root

print(f"\nTotal scenes to generate: {scene_counter}")
print(f"Total key frames:         {scene_counter * args.count}\n")

# ── Run the generator ─────────────────────────────────────────
runner = Generator(run_config, random_seed=args.random_seed)

if os.path.exists(run_config["dataset"]["root"]):
    runner.generate_dataset(True)
else:
    runner.generate_dataset(False)
