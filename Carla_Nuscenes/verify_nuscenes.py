"""
verify_nuscenes.py
==================
Verifies that a dataset rooted at --root conforms to the nuScenes format.

Checks performed
----------------
1. STRUCTURE       – all mandatory JSON tables exist
2. FOREIGN KEYS    – every token reference resolves
3. CAMERAS         – image files exist, filenames non-empty, intrinsics present
4. LIDAR           – .pcd.bin files exist, point clouds loadable & non-empty
5. GT BBOXES       – sample_annotation tokens, size/translation/rotation fields,
                     category tokens resolve, visibility tokens resolve,
                     3-D box dimensions are positive
6. CALIBRATION     – camera & lidar calibrated_sensor records have
                     translation, rotation, camera_intrinsic where expected

Usage
-----
    python verify_nuscenes.py --root ../data/nuscenes/rgb
    python verify_nuscenes.py --root ../data/nuscenes/rgb --verbose
"""

import argparse
import json
import os
import sys
import struct
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✘{RESET}  {msg}")
def info(msg): print(f"  {CYAN}ℹ{RESET}  {msg}")
def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

# ── Mandatory nuScenes JSON tables ────────────────────────────────────────────
REQUIRED_TABLES = [
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_table(prefix: Path, name: str) -> list:
    p = prefix / f"{name}.json"
    with open(p) as f:
        return json.load(f)

def build_index(records: list, key="token") -> dict:
    return {r[key]: r for r in records}

def load_bin_pointcloud(filepath: str):
    """Load a nuScenes-style .pcd.bin (float32 x,y,z,intensity,ring)."""
    pts = np.fromfile(filepath, dtype=np.float32)
    if pts.size % 5 == 0:
        return pts.reshape(-1, 5)
    elif pts.size % 4 == 0:
        return pts.reshape(-1, 4)
    else:
        return pts  # unknown layout – still non-empty if size > 0


# ── Check functions ───────────────────────────────────────────────────────────

def check_structure(root: Path, prefix: Path, verbose: bool) -> bool:
    section("1 · FILE STRUCTURE")
    all_ok = True
    for table in REQUIRED_TABLES:
        p = prefix / f"{table}.json"
        if p.exists():
            if verbose:
                ok(f"{table}.json")
        else:
            err(f"Missing table: {table}.json  (looked in {prefix})")
            all_ok = False
    if all_ok:
        ok("All mandatory JSON tables present")
    return all_ok


def check_foreign_keys(tables: dict, verbose: bool) -> int:
    section("2 · FOREIGN KEY INTEGRITY")
    errors = 0

    # Build token sets
    token_sets = {name: {r["token"] for r in recs}
                  for name, recs in tables.items()}

    checks = [
        # (source_table, field, target_table)
        ("scene",             "log_token",                "log"),
        ("scene",             "first_sample_token",       "sample"),
        ("scene",             "last_sample_token",        "sample"),
        ("sample",            "scene_token",              "scene"),
        ("sample_data",       "sample_token",             "sample"),
        ("sample_data",       "calibrated_sensor_token",  "calibrated_sensor"),
        ("sample_data",       "ego_pose_token",           "ego_pose"),
        ("sample_annotation", "sample_token",             "sample"),
        ("sample_annotation", "instance_token",           "instance"),
        ("sample_annotation", "category_token",           "category"),
        ("sample_annotation", "visibility_token",         "visibility"),
        ("instance",          "category_token",           "category"),
        ("calibrated_sensor", "sensor_token",             "sensor"),
    ]

    for src, field, tgt in checks:
        if src not in tables or tgt not in tables:
            continue
        bad = [r["token"] for r in tables[src]
               if field in r and r[field] and r[field] not in token_sets[tgt]]
        if bad:
            err(f"{src}.{field} → {tgt}: {len(bad)} dangling refs  "
                f"(e.g. {bad[0]})")
            errors += len(bad)
        elif verbose:
            ok(f"{src}.{field} → {tgt}")

    if errors == 0:
        ok("All foreign keys resolve")
    else:
        err(f"{errors} dangling foreign key reference(s)")
    return errors


def check_cameras(root: Path, tables: dict, verbose: bool):
    section("3 · CAMERAS")
    sd_index   = build_index(tables["sample_data"])
    cal_index  = build_index(tables["calibrated_sensor"])
    sen_index  = build_index(tables["sensor"])

    cam_sds = [sd for sd in tables["sample_data"]
               if sen_index.get(
                   cal_index.get(sd.get("calibrated_sensor_token"), {})
                              .get("sensor_token"), {}
               ).get("modality") == "camera"]

    if not cam_sds:
        warn("No camera sample_data records found")
        return

    missing_files = 0
    missing_intrinsics = 0
    empty_filenames = 0

    for sd in cam_sds:
        fn = sd.get("filename", "")
        if not fn:
            empty_filenames += 1
            continue
        fpath = root / fn
        if not fpath.exists():
            missing_files += 1
            if verbose:
                err(f"Image not found: {fpath}")

        cal = cal_index.get(sd.get("calibrated_sensor_token"), {})
        intrinsic = cal.get("camera_intrinsic", [])
        if not intrinsic or intrinsic == [[]] or len(intrinsic) != 3:
            missing_intrinsics += 1
            if verbose:
                warn(f"Bad/missing intrinsic for sd {sd['token']}")

    total = len(cam_sds)
    info(f"Camera sample_data records : {total}")
    if empty_filenames:
        err(f"Empty filename field       : {empty_filenames}")
    if missing_files:
        err(f"Image files not found      : {missing_files} / {total}")
    else:
        ok("All camera image files exist")
    if missing_intrinsics:
        err(f"Missing/bad camera intrinsics : {missing_intrinsics}")
    else:
        ok("All camera calibrations have intrinsics")


def check_lidar(root: Path, tables: dict, verbose: bool):
    section("4 · LIDAR")
    cal_index  = build_index(tables["calibrated_sensor"])
    sen_index  = build_index(tables["sensor"])

    lidar_sds = [sd for sd in tables["sample_data"]
                 if sen_index.get(
                     cal_index.get(sd.get("calibrated_sensor_token"), {})
                               .get("sensor_token"), {}
                 ).get("modality") == "lidar"]

    if not lidar_sds:
        warn("No lidar sample_data records found")
        return

    missing   = 0
    empty_pts = 0
    bad_shape = 0

    for sd in lidar_sds:
        fn = sd.get("filename", "")
        if not fn:
            missing += 1
            continue
        fpath = root / fn
        if not fpath.exists():
            missing += 1
            if verbose:
                err(f"Lidar file not found: {fpath}")
            continue
        try:
            pts = load_bin_pointcloud(str(fpath))
            if pts.size == 0:
                empty_pts += 1
                if verbose:
                    warn(f"Empty point cloud: {fpath}")
            elif pts.ndim == 2 and pts.shape[1] not in (4, 5):
                bad_shape += 1
                if verbose:
                    warn(f"Unexpected point cloud shape {pts.shape}: {fpath}")
        except Exception as e:
            bad_shape += 1
            if verbose:
                err(f"Could not load {fpath}: {e}")

    total = len(lidar_sds)
    info(f"Lidar sample_data records  : {total}")
    if missing:
        err(f"Lidar files not found      : {missing} / {total}")
    else:
        ok("All lidar .bin files exist")
    if empty_pts:
        err(f"Empty point clouds         : {empty_pts}")
    else:
        ok("All point clouds are non-empty")
    if bad_shape:
        warn(f"Unexpected point cloud shape (not 4/5 cols): {bad_shape}")


def check_gt_bboxes(tables: dict, verbose: bool):
    section("5 · GROUND-TRUTH BOUNDING BOXES")
    annotations = tables.get("sample_annotation", [])
    cat_index   = build_index(tables["category"])
    vis_index   = build_index(tables["visibility"])

    if not annotations:
        warn("No sample_annotation records found — no GT boxes to verify")
        return

    bad_size       = []
    bad_translation= []
    bad_rotation   = []
    bad_cat        = []
    bad_vis        = []

    for ann in annotations:
        tok = ann.get("token", "?")

        # size: [w, l, h] all positive
        size = ann.get("size", [])
        if len(size) != 3 or any(v <= 0 for v in size):
            bad_size.append(tok)
            if verbose:
                err(f"Bad size {size} in annotation {tok}")

        # translation: [x, y, z]
        t = ann.get("translation", [])
        if len(t) != 3:
            bad_translation.append(tok)
            if verbose:
                err(f"Bad translation {t} in annotation {tok}")

        # rotation: quaternion [w, x, y, z]
        r = ann.get("rotation", [])
        if len(r) != 4:
            bad_rotation.append(tok)
            if verbose:
                err(f"Bad rotation {r} in annotation {tok}")

        # category token must resolve
        if ann.get("category_token") not in cat_index:
            bad_cat.append(tok)
            if verbose:
                err(f"Unresolved category_token in annotation {tok}")

        # visibility token must resolve (empty string is allowed by nuScenes)
        vis_tok = ann.get("visibility_token", "")
        if vis_tok and vis_tok not in vis_index:
            bad_vis.append(tok)
            if verbose:
                warn(f"Unresolved visibility_token {vis_tok} in annotation {tok}")

    total = len(annotations)
    info(f"sample_annotation records  : {total}")

    # Category distribution
    cat_counts = defaultdict(int)
    for ann in annotations:
        cat = cat_index.get(ann.get("category_token", ""), {}).get("name", "unknown")
        cat_counts[cat] += 1
    info("Category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"       {cnt:>6}  {cat}")

    for label, bad in [("Bad box sizes",       bad_size),
                       ("Bad translations",    bad_translation),
                       ("Bad rotations",       bad_rotation),
                       ("Bad category tokens", bad_cat),
                       ("Bad visibility tokens", bad_vis)]:
        if bad:
            err(f"{label:<25}: {len(bad)} / {total}")
        else:
            ok(f"{label:<25}: all valid")


def check_calibration(tables: dict, verbose: bool):
    section("6 · SENSOR CALIBRATION")
    sen_index = build_index(tables["sensor"])
    bad = []

    for cal in tables.get("calibrated_sensor", []):
        tok = cal.get("token", "?")
        sensor = sen_index.get(cal.get("sensor_token"), {})
        modality = sensor.get("modality", "")

        t = cal.get("translation", [])
        r = cal.get("rotation", [])
        if len(t) != 3:
            bad.append((tok, "translation not length-3"))
        if len(r) != 4:
            bad.append((tok, "rotation not length-4"))

        if modality == "camera":
            intrinsic = cal.get("camera_intrinsic", [])
            if not intrinsic or len(intrinsic) != 3 or any(len(row) != 3 for row in intrinsic):
                bad.append((tok, "camera_intrinsic not 3×3"))

    if bad:
        for tok, reason in bad:
            err(f"calibrated_sensor {tok}: {reason}")
    else:
        ok(f"All {len(tables['calibrated_sensor'])} calibrated_sensor records valid")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify nuScenes dataset format")
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-record details for failures")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"{RED}ERROR: root path does not exist: {root}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}nuScenes Format Verifier{RESET}")
    print(f"Root : {root}")

    # ── Determine JSON prefix ────────────────────────────────────────────────
    # Support v1.0-trainval, v1.0-test, v1.14, or flat layout
    prefix = None
    for candidate in sorted((root).iterdir()) if root.is_dir() else []:
        if candidate.is_dir() and (candidate / "sample.json").exists():
            prefix = candidate
            break
    if prefix is None:
        prefix = root  # flat layout fallback
    print(f"Metadata prefix : {prefix.relative_to(root)}")

    # ── Load all tables ──────────────────────────────────────────────────────
    tables = {}
    missing_tables = []
    for name in REQUIRED_TABLES:
        p = prefix / f"{name}.json"
        if p.exists():
            with open(p) as f:
                tables[name] = json.load(f)
        else:
            missing_tables.append(name)
            tables[name] = []

    # ── Run checks ───────────────────────────────────────────────────────────
    check_structure(root, prefix, args.verbose)

    if missing_tables:
        warn(f"Skipping FK / content checks for missing tables: {missing_tables}")
    else:
        check_foreign_keys(tables, args.verbose)

    check_cameras(root, tables, args.verbose)
    check_lidar(root, tables, args.verbose)
    check_gt_bboxes(tables, args.verbose)
    check_calibration(tables, args.verbose)

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"  Scenes            : {len(tables.get('scene', []))}")
    print(f"  Samples           : {len(tables.get('sample', []))}")
    print(f"  Sample data       : {len(tables.get('sample_data', []))}")
    print(f"  Annotations       : {len(tables.get('sample_annotation', []))}")
    print(f"  Instances         : {len(tables.get('instance', []))}")
    print(f"  Calibrated sensors: {len(tables.get('calibrated_sensor', []))}")
    print()

if __name__ == "__main__":
    main()
