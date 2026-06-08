import json

SCENE_JSON = "./data/nuscenes/rgb2/v1.0-carla/scene.json"

with open(SCENE_JSON) as f:
    scenes = json.load(f)

print(f"Total scenes before: {len(scenes)}")

# Rename sequentially with 4-digit zero-padded numbers
for i, scene in enumerate(scenes, start=1):
    old_name = scene['name']
    scene['name'] = f"scene-{i:04d}"
    print(f"  {old_name}  →  {scene['name']}")

# Verify all names are unique
names = [s['name'] for s in scenes]
assert len(names) == len(set(names)), "Duplicates still exist!"

print(f"\nTotal scenes after: {len(scenes)}")
print("All names unique: OK")

with open(SCENE_JSON, 'w') as f:
    json.dump(scenes, f, indent=2)

print("Done.")
