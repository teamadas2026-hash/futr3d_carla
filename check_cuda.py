import numpy as np

# Load the .bin file
points = np.fromfile(r'D:\teamcarla\futr3d\data\nuscenes\samples\LIDAR_TOP\n001-2025-11-04-13-38-28-0500__LIDAR_TOP__13974485.pcd.bin', dtype=np.float32).reshape(-1, 4)
print(f"Total raw points loaded: {len(points)}")
print(f"Columns (should be x,y,z,intensity): {points.shape}")

# --- Clean the data first ---
# Remove rows with NaN or Inf
valid_mask = np.all(np.isfinite(points), axis=1)
points = points[valid_mask]
print(f"Points after removing NaN/Inf: {len(points)}")

# Remove zero points (common in LiDAR padding)
nonzero_mask = ~np.all(points[:, :3] == 0, axis=1)
points = points[nonzero_mask]
print(f"Points after removing zero points: {len(points)}")

if len(points) == 0:
    print("ERROR: No valid points found! Check your .bin file path or format.")
    print("Try reshaping differently — some formats use 5 columns:")
    # Try 5 columns (x, y, z, intensity, ring)
    points_raw = np.fromfile(r'D:\teamcarla\futr3d\data\nuscenes\samples\LIDAR_TOP\n001-2025-11-04-13-38-28-0500__LIDAR_TOP__13974485.pcd.bin', dtype=np.float32)
    print(f"Total float values in file: {len(points_raw)}")
    print(f"Divisible by 3: {len(points_raw) % 3 == 0}")
    print(f"Divisible by 4: {len(points_raw) % 4 == 0}")
    print(f"Divisible by 5: {len(points_raw) % 5 == 0}")
    print(f"Divisible by 6: {len(points_raw) % 6 == 0}")
else:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Remove points too close to origin (sensor noise)
    r = np.sqrt(x**2 + y**2 + z**2)
    r_mask = r > 0.5  # at least 0.5m from sensor
    x, y, z, r = x[r_mask], y[r_mask], z[r_mask], r[r_mask]
    print(f"Points after removing near-origin: {len(x)}")

    if len(x) == 0:
        print("ERROR: All points were within 0.5m of origin — unusual data")
    else:
        # Compute pitch
        pitch = np.arcsin(np.clip(z / r, -1.0, 1.0)) * 180 / np.pi

        pitch_rounded = np.round(pitch, 1)
        unique_pitches = np.sort(np.unique(pitch_rounded))

        print(f"\n✓ Number of beams detected: {len(unique_pitches)}")
        print(f"✓ Pitch angle range: {unique_pitches.min():.1f}° to {unique_pitches.max():.1f}°")
        print(f"✓ Unique pitch angles: {unique_pitches}")