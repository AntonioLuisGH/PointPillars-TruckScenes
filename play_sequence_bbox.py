import open3d as o3d
import numpy as np
import os
import glob
import argparse
import time
import traceback # --- NEW: For better error printing ---

# --- Get the absolute path of this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- NEW: Color map for bounding boxes ---
CLASS_COLORS = {
    'Car': [0, 1, 0],           # Green
    'Pedestrian': [1, 0, 0],    # Red
    'Cyclist': [0, 1, 1],       # Cyan
    'Van': [1, 1, 0],           # Yellow
    'Truck': [0.5, 0, 0.5],     # Purple
    'Misc': [0.5, 0.5, 0.5],    # Gray
    'DontCare': [1, 1, 1],      # White
}

# --- NEW: Helper function to read calibration file ---
def read_calib_file(filepath):
    """
    Reads a KITTI calibration file and returns the
    full 4x4 transformation matrix from Velodyne to
    Rectified Camera 0.
    """
    calib = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            try:
                calib[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass # Skip lines like 'calib_time'

    # Get R0_rect (3x3) and make it 4x4
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)

    # Get Tr_velo_to_cam (3x4) and make it 4x4
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)

    # Calculate the full transform
    T_full = R0_rect @ Tr_velo_to_cam
    return T_full

# --- NEW: Helper function to read label file ---
def read_label_file(filepath):
    """
    Reads a KITTI label file and returns a list of
    Open3D OrientedBoundingBox objects.
    """
    boxes = []
    if not os.path.exists(filepath):
        return []
        
    with open(filepath, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            
            obj_type = parts[0]
            if obj_type == 'DontCare':
                continue
                
            try:
                # h, w, l, x, y, z, ry
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                ry = float(parts[14])
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed line in {filepath}: {line}")
                continue

            # Create 3x3 rotation matrix (rotation around Y-axis)
            R_y = np.array([
                [np.cos(ry),  0, np.sin(ry)],
                [0,           1, 0          ],
                [-np.sin(ry), 0, np.cos(ry)]
            ])

            # KITTI (x,y,z) is bottom-center. O3D needs box center.
            # (x, y, z) -> (x, y - h/2, z)
            center = [x, y - h / 2.0, z]
            
            # KITTI (w, h, l) maps to O3D (x-dim, y-dim, z-dim)
            extent = [w, h, l]
            
            obb = o3d.geometry.OrientedBoundingBox(center, R_y, extent)
            
            # Set color
            color = CLASS_COLORS.get(obj_type, [1, 1, 1]) # Default to white
            obb.color = color
            
            boxes.append(obb)
    return boxes


class PlayerState:
    """Helper class to store the player's state"""
    # --- NEW: Added 'label_files' and 'calib_transform' ---
    def __init__(self, scan_files, label_files, calib_transform, fps):
        self.is_paused = False
        self.scan_files = scan_files
        self.label_files = label_files       # --- NEW ---
        self.calib_transform = calib_transform # --- NEW ---
        self.current_boxes = []              # --- NEW ---
        
        self.total_frames = len(scan_files)
        self.current_frame = 0
        
        self.fps = fps
        self.frame_duration = 1.0 / self.fps
        self.last_frame_time = time.time()

    def toggle_pause(self, vis):
        """Callback for SPACE key press"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            print(">> Playback Paused")
        else:
            print(">> Playback Resumed")
        return True # Keep the callback registered

    def advance_frame(self, vis, pcd):
        """Callback for animation ticks. This is our new 'loop'."""
        
        vis.poll_events()

        if self.is_paused:
            vis.update_renderer()
            return True

        current_time = time.time()
        if (current_time - self.last_frame_time) < self.frame_duration:
            vis.update_renderer()
            return True
        
        self.last_frame_time = current_time
        
        try:
            # --- NEW: Remove Old Boxes ---
            for box in self.current_boxes:
                vis.remove_geometry(box, reset_bounding_box=False)
            self.current_boxes.clear()
            
            # --- Load Scan ---
            scan_path = self.scan_files[self.current_frame]
            points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            
            # --- NEW: Transform Points to Camera Coords ---
            points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
            points_transformed = (self.calib_transform @ points_hom.T).T
            
            # Update PointCloud geometry
            pcd.points = o3d.utility.Vector3dVector(points_transformed[:, :3])
            pcd.paint_uniform_color([0.8, 0.8, 0.8])
            vis.update_geometry(pcd) # Update the points
            
            # --- NEW: Load Labels and Add New Boxes ---
            label_path = self.label_files[self.current_frame]
            if label_path is not None:
                self.current_boxes = read_label_file(label_path)
                for box in self.current_boxes:
                    vis.add_geometry(box, reset_bounding_box=False)

            # --- Advance frame index ---
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                print("End of sequence. Looping.")
                self.current_frame = 0
                
        except Exception as e:
            print(f"Error in animation callback: {e}")
            traceback.print_exc() # Print full error
            return False # Stop animation on error

        vis.update_renderer()
        return True # Tell Open3D to keep running


def play_sequence(sequence_dir, fps):
    """
    Plays a LiDAR sequence from a directory.
    Assumes 'velodyne', 'label_2', and 'calib.txt'.
    """
    
    scan_dir = os.path.join(sequence_dir, 'velodyne')
    label_dir = os.path.join(sequence_dir, 'label_2')   # --- NEW ---
    calib_file = os.path.join(sequence_dir, 'calib.txt') # --- NEW ---
    
    if not os.path.exists(scan_dir):
        print(f"Error: Velodyne directory not found: {scan_dir}")
        return

    # --- NEW: Check for labels and calibration ---
    load_labels = True
    if not os.path.exists(label_dir):
        print(f"Warning: Label directory not found: {label_dir}. Will not display bounding boxes.")
        load_labels = False
    
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file not found: {calib_file}.")
        print("This file is required to align point clouds and labels.")
        return
    
    print(f"Loading calibration from {calib_file}")
    calib_transform = read_calib_file(calib_file)
    # ---------------------------------------------

    scan_files = sorted(glob.glob(os.path.join(scan_dir, '*.bin')))
    if not scan_files:
        print(f"Error: No .bin files found in {scan_dir}")
        return

    # --- NEW: Generate corresponding label file paths ---
    if load_labels:
        label_files = [
            os.path.join(label_dir, f"{os.path.splitext(os.path.basename(f))[0]}.txt") 
            for f in scan_files
        ]
        print(f"Found {len(scan_files)} scans and {len(label_files)} corresponding label files.")
    else:
        label_files = [None] * len(scan_files) # Pass a list of Nones
    # ----------------------------------------------------
    
    # --- NEW: Pass new args to state ---
    state = PlayerState(scan_files, label_files, calib_transform, fps=fps)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='LiDAR Sequence Player (Press SPACE to pause)')
    
    vis.register_key_callback(32, state.toggle_pause) # 32 = SPACE
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    # --- Create Geometries ---
    # --- NEW: Must transform the *first* frame too! ---
    first_frame_points_raw = np.fromfile(scan_files[0], dtype=np.float32).reshape(-1, 4)
    points_hom = np.hstack((first_frame_points_raw[:, :3], np.ones((first_frame_points_raw.shape[0], 1))))
    points_transformed = (calib_transform @ points_hom.T).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_transformed[:, :3])
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    # ----------------------------------------------------
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
    
    # --- Try to load the camera view ---
    ctr = vis.get_view_control()
    viewpoint_path = os.path.join(SCRIPT_DIR, 'viewpoint.json')
    
    if os.path.exists(viewpoint_path):
        try:
            param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
            ctr.convert_from_pinhole_camera_parameters(param)
            print(f"Successfully loaded camera view from {viewpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load '{viewpoint_path}'. Using fallback view. Error: {e}")
            ctr.set_front([0, -1, 0.2]); ctr.set_lookat([0, 0, 0]); ctr.set_up([0, 0, 1]); ctr.set_zoom(0.1)
    else:
        print(f"Warning: 'viewpoint.json' not found. Using fallback view.")
        # --- NEW: Adjusted fallback view to be better for KITTI camera coords ---
        # Look from behind the camera (positive Z) towards the origin
        ctr.set_front([0, -0.5, -1]); # Look "down" and "into" the scene
        ctr.set_lookat([0, 0, 20]);   # Look at a point 20m ahead
        ctr.set_up([0, -1, 0]);      # "Up" in camera coords is negative Y
        ctr.set_zoom(0.05)

    # --- Register the animation callback ---
    vis.register_animation_callback(lambda vis: state.advance_frame(vis, pcd))

    print(f"Starting visualization at {fps} FPS. Close the window to exit.")
    
    try:
        vis.run() 
    finally:
        vis.destroy_window()
        print("Playback finished or window closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a LiDAR sequence with 3D Bounding Boxes.')
    parser.add_argument('--seq_dir', 
                        type=str, 
                        required=True,
                        help='Path to the sequence directory (e.g., /path/to/kitti/training/0001)')
    parser.add_argument('--fps',
                        type=int,
                        default=10,
                        help='Playback frames per second (default: 10)')
    
    args = parser.parse_args()
    play_sequence(args.seq_dir, args.fps)