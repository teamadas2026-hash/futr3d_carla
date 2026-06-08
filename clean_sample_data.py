import json
import os
from pathlib import Path

def clean_sample_data_json():
    json_path = r"D:\teamcarla\futr3d\data\nuscenes\v1.0-trainval\sample_data.json"
    samples_dir = r"D:\teamcarla\futr3d\data\nuscenes\samples"

    with open(json_path, 'r') as f:
        data = json.load(f)

    existing_files = set()
    for root, dirs, files in os.walk(samples_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), samples_dir)
            existing_files.add(rel_path.replace('\\', '/'))

    print(f"Found {len(existing_files)} files in samples directory")

    filtered_data = []
    removed_count = 0

    for entry in data:
        filename = entry.get('filename', '')
        if filename.startswith('samples/'):
            rel_path = filename.replace('samples/', '')
            if rel_path in existing_files:
                filtered_data.append(entry)
            else:
                removed_count += 1
                print(f"Removing: {rel_path}")
        else:
            filtered_data.append(entry)

    print(f"\nRemoved {removed_count} entries")
    print(f"Kept {len(filtered_data)} entries")

    backup_path = json_path + '.backup'
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Backup saved to: {backup_path}")

    with open(json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Cleaned JSON saved to: {json_path}")

if __name__ == '__main__':
    clean_sample_data_json()
