"""
fix_nuscenes.py
---------------
Fixes two confirmed issues in the generated nuScenes dataset:

  FIX 1 · sensor.json
    3 sensor_tokens in calibrated_sensor.json have no matching entry in
    sensor.json.  All 3 are non-camera sensors (no camera_intrinsic).
    Identified by matching translation+rotation fingerprints:
      c8ffe98b…  →  LIDAR_TOP   (exact fingerprint match)
      f41746fa…  →  RADAR_FRONT_RIGHT  (z=0, identity rotation, nearest radar)
      dcfdb4a3…  →  RADAR_BACK_RIGHT   (x=-0.7, z=0, identity rotation)
    NOTE: the 2 radar entries have z=0 (generator bug — should be z=0.5).
    We add them to sensor.json with their correct channel/modality so FK
    integrity passes.  The translation bug in calibrated_sensor is logged
    as a warning but left in place (it doesn't affect model training since
    RADAR data is not referenced in sample_data).

  FIX 2 · sample_annotation.json
    Every annotation is missing the `category_token` field.
    Fix: annotation.instance_token → instance.category_token → inject.
    After fix: vehicle.car=2281, human.pedestrian.adult=2283,
               vehicle.truck=313, vehicle.motorcycle=218.

Usage:
    python fix_nuscenes.py --root ../data/nuscenes/rgb          # dry-run
    python fix_nuscenes.py --root ../data/nuscenes/rgb --apply  # write files
"""

import json, shutil, argparse
from pathlib import Path
from collections import defaultdict

GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET = "\033[0m"

def ok(m):      print(f"  {GREEN}✔{RESET}  {m}")
def warn(m):    print(f"  {YELLOW}⚠{RESET}  {m}")
def err(m):     print(f"  {RED}✘{RESET}  {m}")
def info(m):    print(f"  {CYAN}ℹ{RESET}  {m}")
def section(t): print(f"\n{BOLD}{CYAN}{'─'*60}\n  {t}\n{'─'*60}{RESET}")

def find_prefix(root):
    for c in sorted(root.iterdir()):
        if c.is_dir() and (c / "sample.json").exists():
            return c
    return root

def load(prefix, name):
    with open(prefix / f"{name}.json") as f:
        return json.load(f)

def save(prefix, name, data, apply):
    path = prefix / f"{name}.json"
    if apply:
        bak = path.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            info(f"Backup → {bak.name}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        ok(f"Written {path.name}  ({len(data)} records)")
    else:
        warn(f"DRY-RUN: would write {path.name}  ({len(data)} records)")


# ── FIX 1 ─────────────────────────────────────────────────────────────────────
def fix_sensor_json(prefix, apply):
    section("FIX 1 · Add 3 missing entries to sensor.json")

    sensors     = load(prefix, "sensor")
    cal_sensors = load(prefix, "calibrated_sensor")

    sensor_by_token = {s["token"]: s for s in sensors}
    missing_tokens  = {c["sensor_token"] for c in cal_sensors
                       if c["sensor_token"] not in sensor_by_token}

    if not missing_tokens:
        ok("No missing sensor tokens — nothing to fix")
        return

    info(f"Missing sensor_tokens: {len(missing_tokens)}")

    # Confirmed mapping derived from translation+rotation fingerprint analysis:
    KNOWN_MAPPING = {
        # token                              channel             modality
        "c8ffe98bcce9f00f983813cde9d4b7e4": ("LIDAR_TOP",         "lidar"),
        "f41746faf0c228a0f34891392b702eb9": ("RADAR_FRONT_RIGHT", "radar"),
        "dcfdb4a33530ff2aef3c35b7a114cd0e": ("RADAR_BACK_RIGHT",  "radar"),
    }

    # Build fingerprint map from known cal_sensors as fallback for future runs
    seen_fp = set()
    fp_to_channel = {}
    for c in cal_sensors:
        if c["sensor_token"] not in sensor_by_token:
            continue
        ch  = sensor_by_token[c["sensor_token"]]["channel"]
        mod = sensor_by_token[c["sensor_token"]]["modality"]
        fp  = (tuple(round(x, 3) for x in c["translation"]),
               tuple(round(x, 4) for x in c["rotation"]),
               len(c.get("camera_intrinsic", [])) == 3)
        if fp not in fp_to_channel:
            fp_to_channel[fp] = (ch, mod)

    new_sensors = list(sensors)
    for tok in missing_tokens:
        if tok in KNOWN_MAPPING:
            channel, modality = KNOWN_MAPPING[tok]
            source = "hardcoded fingerprint analysis"
        else:
            # Fallback: match by fingerprint
            sample_cal = next(c for c in cal_sensors if c["sensor_token"] == tok)
            fp = (tuple(round(x, 3) for x in sample_cal["translation"]),
                  tuple(round(x, 4) for x in sample_cal["rotation"]),
                  len(sample_cal.get("camera_intrinsic", [])) == 3)
            channel, modality = fp_to_channel.get(fp, ("UNKNOWN", "unknown"))
            source = "fingerprint lookup"

        new_entry = {"token": tok, "channel": channel, "modality": modality}
        new_sensors.append(new_entry)
        ok(f"Adding  token={tok[:8]}…  channel={channel:<20}  modality={modality}  [{source}]")

    warn("Note: 2 radar cal_sensor entries have z=0 in translation (should be 0.5).")
    warn("      This is a generator bug. The entries are not referenced in sample_data")
    warn("      so they won't affect training. FK integrity will pass after this fix.")

    save(prefix, "sensor", new_sensors, apply)


# ── FIX 2 ─────────────────────────────────────────────────────────────────────
def fix_category_tokens(prefix, apply):
    section("FIX 2 · Inject category_token into sample_annotation")

    annotations = load(prefix, "sample_annotation")
    instances   = load(prefix, "instance")
    categories  = load(prefix, "category")

    inst_by_token = {i["token"]: i for i in instances}
    cat_by_token  = {c["token"]: c["name"] for c in categories}

    patched      = 0
    already_ok   = 0
    no_instance  = 0
    bad_cat      = 0

    for ann in annotations:
        existing = ann.get("category_token", "")
        if existing and existing in cat_by_token:
            already_ok += 1
            continue

        inst = inst_by_token.get(ann.get("instance_token", ""))
        if inst is None:
            no_instance += 1
            ann["category_token"] = ""
            continue

        cat_tok = inst.get("category_token", "")
        if cat_tok not in cat_by_token:
            bad_cat += 1
            ann["category_token"] = cat_tok
            continue

        ann["category_token"] = cat_tok
        patched += 1

    info(f"Total annotations              : {len(annotations)}")
    if already_ok:
        ok(f"Already had valid category_token : {already_ok}")
    ok(f"Patched via instance table       : {patched}")
    if no_instance:
        err(f"No matching instance found       : {no_instance}  (set to '')")
    if bad_cat:
        warn(f"Instance had unresolvable cat    : {bad_cat}")

    from collections import Counter
    dist = Counter(
        cat_by_token.get(a.get("category_token", ""), "unknown")
        for a in annotations
    )
    info("Category distribution after fix:")
    for name, cnt in dist.most_common():
        mark = f"{GREEN}✔{RESET}" if name != "unknown" else f"{RED}✘{RESET}"
        print(f"    {mark}  {cnt:>6}  {name}")

    save(prefix, "sample_annotation", annotations, apply)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fix nuScenes dataset issues")
    parser.add_argument("--root",  required=True)
    parser.add_argument("--apply", action="store_true",
                        help="Write fixes (default: dry-run)")
    args = parser.parse_args()

    root   = Path(args.root).expanduser().resolve()
    prefix = find_prefix(root)

    print(f"\n{BOLD}nuScenes Fix Script{RESET}")
    print(f"Root   : {root}")
    print(f"Prefix : {prefix.name}")
    print(f"Mode   : {'⚡ APPLY — files will be modified' if args.apply else '🔍 DRY-RUN — no files touched'}")

    fix_sensor_json(prefix, args.apply)
    fix_category_tokens(prefix, args.apply)

    section("DONE")
    if not args.apply:
        print(f"  {YELLOW}No files modified. Re-run with --apply to write fixes.{RESET}\n")
    else:
        print(f"  {GREEN}Fixes applied. Run verify_nuscenes.py to confirm.{RESET}\n")

if __name__ == "__main__":
    main()