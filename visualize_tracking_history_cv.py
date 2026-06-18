"""
Visualize tracking history from YAML file using OpenCV.
Shows ego vehicle trajectory and all tracked objects.
"""
import yaml
import numpy as np
import cv2
import argparse


def load_history(yaml_path):
    """Load tracking history from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def get_class_color(class_name):
    """Return BGR color for each object class."""
    colors = {
        'car': (0, 0, 255),              # Red
        'pedestrian': (0, 255, 255),     # Cyan
        'bicycle': (255, 0, 0),          # Blue
        'motorcycle': (0, 165, 255),     # Orange
        'truck': (255, 255, 0),          # Cyan/Yellow
        'bus': (255, 0, 255),            # Magenta
        'trailer': (255, 0, 128),        # Purple-ish
        'construction_vehicle': (255, 192, 203)  # Pink
    }
    return colors.get(class_name, (128, 128, 128))  # Gray as default


def convert_to_pixel(pos, img_height, img_width, scale=5, offset_x=250, offset_y=250):
    """Convert world coordinates to image pixel coordinates."""
    x_pixel = int(offset_x + pos[0] * scale)
    y_pixel = int(offset_y - pos[1] * scale)  # Flip Y axis
    return x_pixel, y_pixel


def draw_tracking_history(data, output_path="tracking_history_cv.png", img_size=1000):
    """
    Draw tracking history using OpenCV.
    
    Args:
        data: Dictionary loaded from YAML with ego_history and track_history
        output_path: Path to save the image
        img_size: Size of the output image (square)
    """
    # Create blank image
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Extract data
    ego_history = data.get('ego_history', [])
    track_history = data.get('track_history', {})
    
    offset_x = img_size // 2
    offset_y = img_size // 2
    scale = 3  # pixels per meter
    
    # Draw grid
    for i in range(-100, 101, 10):
        x_pixel = offset_x + i * scale
        cv2.line(img, (x_pixel, 0), (x_pixel, img_size), (230, 230, 230), 1)
        y_pixel = offset_y - i * scale
        cv2.line(img, (0, y_pixel), (img_size, y_pixel), (230, 230, 230), 1)
    
    # Draw axes
    cv2.line(img, (offset_x, 0), (offset_x, img_size), (200, 200, 200), 2)
    cv2.line(img, (0, offset_y), (img_size, offset_y), (200, 200, 200), 2)
    
    # Draw ego vehicle trajectory
    if ego_history:
        ego_positions = [convert_to_pixel(state['pos'], img_size, img_size, scale, offset_x, offset_y) 
                        for state in ego_history]
        
        # Draw trajectory polyline
        pts = np.array(ego_positions, np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 255), 3)
        
        # Draw start point (green)
        cv2.circle(img, ego_positions[0], 8, (0, 255, 0), -1)
        cv2.putText(img, "START", (ego_positions[0][0]+10, ego_positions[0][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw end point (red star)
        cv2.circle(img, ego_positions[-1], 10, (0, 0, 255), -1)
        cv2.putText(img, "EGO END", (ego_positions[-1][0]+10, ego_positions[-1][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw track trajectories
    for track_id, track_data in sorted(track_history.items()):
        if not track_data['states']:
            continue
        
        class_name = track_data['class']
        color = get_class_color(class_name)
        
        positions = [convert_to_pixel(state['pos'], img_size, img_size, scale, offset_x, offset_y)
                    for state in track_data['states']]
        
        # Draw trajectory polyline
        pts = np.array(positions, np.int32)
        cv2.polylines(img, [pts], False, color, 2)
        
        # Draw start point
        cv2.circle(img, positions[0], 4, color, -1)
        
        # Draw end point (current position)
        cv2.circle(img, positions[-1], 6, color, -1)
        
        # Add track ID label at end position
        text = f"ID{track_id}"
        cv2.putText(img, text, (positions[-1][0]+5, positions[-1][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add title and legend
    cv2.putText(img, "3D Object Tracking History - Top-Down View", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, f"Scale: {scale} pixels/meter", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
    
    # Add legend
    legend_y = img_size - 120
    cv2.putText(img, "Legend:", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    legend_items = [
        ("Ego Vehicle", (0, 0, 255)),
        ("Car", get_class_color('car')),
        ("Pedestrian", get_class_color('pedestrian')),
        ("Other", (128, 128, 128)),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y + 30 + i * 25
        cv2.circle(img, (35, y), 4, color, -1)
        cv2.putText(img, label, (60, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"Saved tracking history visualization to {output_path}")
    return img


def print_summary(data):
    """Print summary statistics."""
    ego_history = data.get('ego_history', [])
    track_history = data.get('track_history', {})
    
    print("\n" + "="*70)
    print("TRACKING HISTORY SUMMARY")
    print("="*70)
    
    if ego_history:
        ego_pos_start = ego_history[0]['pos']
        ego_pos_end = ego_history[-1]['pos']
        ego_dist = np.sqrt((ego_pos_end[0] - ego_pos_start[0])**2 + 
                          (ego_pos_end[1] - ego_pos_start[1])**2)
        ego_time = ego_history[-1]['time'] - ego_history[0]['time']
        
        print(f"\nEgo Vehicle:")
        print(f"  Duration: {ego_time:.2f} seconds ({len(ego_history)} states)")
        print(f"  Start position: ({ego_pos_start[0]:.2f}, {ego_pos_start[1]:.2f})")
        print(f"  End position: ({ego_pos_end[0]:.2f}, {ego_pos_end[1]:.2f})")
        print(f"  Distance traveled: {ego_dist:.2f} m")
        
        # Calculate average velocity
        avg_vels = [np.sqrt(state['vel'][0]**2 + state['vel'][1]**2) for state in ego_history]
        print(f"  Average speed: {np.mean(avg_vels):.2f} m/s")
        print(f"  Max speed: {np.max(avg_vels):.2f} m/s")
    
    print(f"\nTracked Objects: {len(track_history)} tracks")
    
    class_counts = {}
    for track_id, track_data in sorted(track_history.items()):
        if not track_data['states']:
            continue
        
        class_name = track_data['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        states = track_data['states']
        duration = states[-1]['time'] - states[0]['time']
        pos_start = states[0]['pos']
        pos_end = states[-1]['pos']
        dist = np.sqrt((pos_end[0] - pos_start[0])**2 + (pos_end[1] - pos_start[1])**2)
        
        speeds = [np.sqrt(state['vel'][0]**2 + state['vel'][1]**2) for state in states]
        
        print(f"\n  Track {track_id:2d} ({class_name:15s}): {len(states):2d} states, {duration:5.2f}s")
        print(f"               Distance: {dist:6.2f}m | Avg speed: {np.mean(speeds):5.2f}m/s | Max: {np.max(speeds):5.2f}m/s")
    
    print(f"\nClass Distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:20s}: {count:2d} tracks")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize tracking history from YAML file')
    parser.add_argument('--yaml', type=str, default='tracking_history.yaml',
                       help='Path to tracking history YAML file')
    parser.add_argument('--output', type=str, default='tracking_history_cv.png',
                       help='Path to save the output image')
    parser.add_argument('--size', type=int, default=1200,
                       help='Image size in pixels')
    args = parser.parse_args()
    
    # Load and visualize
    print(f"Loading tracking history from {args.yaml}...")
    data = load_history(args.yaml)
    
    # Print summary
    print_summary(data)
    
    # Create visualization
    img = draw_tracking_history(data, output_path=args.output, img_size=args.size)
    
    print(f"\nVisualization saved to: {args.output}")
    print("Open the image file to view the tracking history.")
