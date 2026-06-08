#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

# Minimal valid 1x1 transparent PNG
DUMMY_PNG = (
    b'\x89PNG\r\n\x1a\n'
    b'\x00\x00\x00\rIHDR'
    b'\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
    b'\x00\x00\x00\x0cIDAT'
    b'\x08\xd7c\xf8\x0f\x00\x01\x01\x01\x00'
    b'\x18\xdd\x8d\xb1'
    b'\x00\x00\x00\x00IEND\xaeB`\x82'
)

def main():
    parser = argparse.ArgumentParser(
        description="Create dummy PNG files from JSON metadata."
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON file"
    )
    parser.add_argument(
        "root_dir",
        help="Root directory where files should be created"
    )

    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        data = json.load(f)

    root_dir = Path(args.root_dir)

    for item in data:
        rel_path = item["filename"]  # e.g. maps/e672bf3d....png

        # If root_dir already points to .../maps, use only the basename
        if root_dir.name == "maps":
            target = root_dir / Path(rel_path).name
        else:
            target = root_dir / rel_path

        target.parent.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            with open(target, "wb") as f:
                f.write(DUMMY_PNG)

            print(f"Created: {target}")
        else:
            print(f"Exists:  {target}")

if __name__ == "__main__":
    main()