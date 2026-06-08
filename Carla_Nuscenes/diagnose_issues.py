"""
diagnose_issues.py  (v2)
------------------------
python diagnose_issues.py --root ../data/nuscenes/rgb
"""
import json, argparse
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root).expanduser().resolve()
prefix = None
for c in sorted(root.iterdir()):
    if c.is_dir() and (c / "sample.json").exists():
        prefix = c
        break
if prefix is None:
    prefix = root

print(f"Root   : {root}")
print(f"Prefix : {prefix.name}\n")

def load(name):
    with open(prefix / f"{name}.json") as f:
        return json.load(f)

sensors     = load("sensor")
cal_sensors = load("calibrated_sensor")
categories  = load("category")
annotations = load("sample_annotation")
instances   = load("instance")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ISSUE 1 · calibrated_sensor → sensor (dangling tokens)")
print("=" * 60)
sensor_tokens = {s["token"]: s for s in sensors}
bad_cal = [c for c in cal_sensors if c["sensor_token"] not in sensor_tokens]

# Group dangling by their sensor_token to find duplicate rigs
dangling_sensor_tokens = defaultdict(list)
for c in bad_cal:
    dangling_sensor_tokens[c["sensor_token"]].append(c["token"])

print(f"  sensor.json records         : {len(sensors)}")
print(f"  calibrated_sensor rows      : {len(cal_sensors)}")
print(f"  Dangling rows               : {len(bad_cal)}")
print(f"  Distinct unknown sen_tokens : {len(dangling_sensor_tokens)}")
print(f"\n  Unknown sensor_token values (i.e. tokens NOT in sensor.json):")
for tok, cal_toks in list(dangling_sensor_tokens.items())[:15]:
    print(f"    [{tok}]  appears in {len(cal_toks)} calibrated_sensor row(s)")

# Check if these tokens look like they match a channel by value pattern
all_cal_tokens = {c["sensor_token"] for c in cal_sensors}
print(f"\n  Total distinct sensor_tokens referenced in cal_sensor : {len(all_cal_tokens)}")
print(f"  Total tokens present in sensor.json                   : {len(sensor_tokens)}")
missing = all_cal_tokens - set(sensor_tokens.keys())
print(f"  Missing from sensor.json                              : {len(missing)}")
print(f"  Missing token values:")
for t in sorted(missing):
    print(f"    {t}")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ISSUE 2 · sample_annotation fields & category mapping")
print("=" * 60)

print(f"\n  sample_annotation count : {len(annotations)}")
if annotations:
    print(f"  Keys in first annotation: {list(annotations[0].keys())}")
    print(f"\n  First annotation (full):")
    for k, v in annotations[0].items():
        print(f"    {k:<30} : {v}")

print(f"\n  instance count          : {len(instances)}")
if instances:
    print(f"  Keys in first instance  : {list(instances[0].keys())}")
    print(f"\n  First instance (full):")
    for k, v in instances[0].items():
        print(f"    {k:<30} : {v}")

# Check what category field is called
cat_by_token = {c["token"]: c["name"] for c in categories}
cat_by_name  = {c["name"]: c["token"] for c in categories}

# Try all string fields in annotations to find which one holds category info
if annotations:
    print(f"\n  Scanning annotation string fields for category clues...")
    for key, val in annotations[0].items():
        if isinstance(val, str):
            in_tok = val in cat_by_token
            in_name = val in cat_by_name
            print(f"    field '{key}' = '{val}'  → matches_cat_token={in_tok}  matches_cat_name={in_name}")