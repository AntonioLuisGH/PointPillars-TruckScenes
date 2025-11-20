import open3d as o3d
import numpy as np
import os
import glob
import argparse
import time
import traceback
import sys 

# --- Get the absolute path of this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Color map for bounding boxes ---
CLASS_COLORS = {
    'Car': [0, 1, 0],           # Green
    'Pedestrian': [1, 0, 0],    # Red
    'Cyclist': [0, 1, 1],       # Cyan
    'Drone': [1, 1, 0],         # Yellow
    'DontCare': [1, 1, 1],      # White
}

# --- Helper function to read calibration file ---
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
                pass 

    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect'].reshape(3, 3)

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)

    T_full = R0_rect @ Tr_velo_to_cam
    return T_full

# --- Helper function to read label file ---
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
                continue

            # Create 3x3 rotation matrix (rotation around Y-axis)
            R_y = np.array([
                [np.cos(ry),  0, np.sin(ry)],
                [0,           1, 0          ],
                [-np.sin(ry), 0, np.cos(ry)]
            ])

            # KITTI (x,y,z) is bottom-center. O3D needs box center.
            center = [x, y - h / 2.0, z]
            extent = [w, h, l]
            
            obb = o3d.geometry.OrientedBoundingBox(center, R_y, extent)
            color = CLASS_COLORS.get(obj_type, [1, 1, 1]) 
            obb.color = color
            
            boxes.append(obb)
    return boxes


class PlayerState:
    """Helper class to store the player's state"""
    # --- NEW: Added start_frame argument ---
    def __init__(self, scan_files, label_files, calib_transform, fps, start_frame=0):
        self.is_paused = False
        self.scan_files = scan_files
        self.label_files = label_files
        self.calib_transform = calib_transform
        self.current_boxes = []
        
        self.total_frames = len(scan_files)
        
        # --- NEW: Validate and set start frame ---
        if start_frame < 0 or start_frame >= self.total_frames:
            print(f"Warning: Start frame {start_frame} is out of bounds (0-{self.total_frames-1}). Resetting to 0.")
            self.current_frame = 0
        else:
            self.current_frame = start_frame
            print(f"Starting playback at frame {self.current_frame}")
        # -----------------------------------------
        
        self.fps = fps
        self.frame_duration = 1.0 / self.fps
        self.last_frame_time = time.time()

    def toggle_pause(self, vis):
        """Callback for SPACE key press"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("\n>> Playback Paused")
        else:
            print("\n>> Playback Resumed")
        return True 

    def advance_frame(self, vis, pcd):
        """Callback for animation ticks."""
        
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
            # --- Remove Old Boxes ---
            for box in self.current_boxes:
                vis.remove_geometry(box, reset_bounding_box=False)
            self.current_boxes.clear()
            
            # --- Load Scan ---
            scan_path = self.scan_files[self.current_frame]
            
            # --- Display Filename in Console ---
            file_name = os.path.basename(scan_path)
            status_msg = f"Playing Frame [{self.current_frame+1}/{self.total_frames}]: {file_name}"
            print(f"\r{status_msg:<60}", end="", flush=True)

            points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.paint_uniform_color([0.8, 0.8, 0.8])
            vis.update_geometry(pcd)
            
            # --- Load Labels and Transform to LiDAR Coords ---
            label_path = self.label_files[self.current_frame]
            if label_path is not None:
                camera_boxes = read_label_file(label_path)
                
                # Get the INVERSE transform (Camera -> LiDAR)
                T_lidar_from_cam = np.linalg.inv(self.calib_transform)
                R_lidar_from_cam = T_lidar_from_cam[:3, :3]

                self.current_boxes = []
                for cam_box in camera_boxes:
                    # Transform logic
                    cam_center_hom = np.append(cam_box.center, 1)
                    lidar_center_hom = T_lidar_from_cam @ cam_center_hom
                    lidar_center = lidar_center_hom[:3]
                    lidar_R = R_lidar_from_cam @ cam_box.R
                    lidar_extent = cam_box.extent

                    lidar_box = o3d.geometry.OrientedBoundingBox(lidar_center, 
                                                                  lidar_R, 
                                                                  lidar_extent)
                    lidar_box.color = cam_box.color 
                    
                    self.current_boxes.append(lidar_box)
                    vis.add_geometry(lidar_box, reset_bounding_box=False)

            # --- Advance frame index ---
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                print("\nEnd of sequence. Looping.")
                self.current_frame = 0
                
        except Exception as e:
            print(f"\nError in animation callback: {e}")
            traceback.print_exc() 
            return False 

        vis.update_renderer()
        return True


def play_sequence(sequence_dir, fps, start_frame):
    """
    Plays a LiDAR sequence from a directory.
    """
    scan_dir = os.path.join(sequence_dir, 'velodyne')
    label_dir = os.path.join(sequence_dir, 'label_2')
    calib_file = os.path.join(sequence_dir, 'calib.txt')
    
    if not os.path.exists(scan_dir):
        print(f"Error: Velodyne directory not found: {scan_dir}")
        return

    load_labels = True
    if not os.path.exists(label_dir):
        print(f"Warning: Label directory not found: {label_dir}. Will not display bounding boxes.")
        load_labels = False
    
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file not found: {calib_file}.")
        return
    
    print(f"Loading calibration from {calib_file}")
    calib_transform = read_calib_file(calib_file)

    scan_files = sorted(glob.glob(os.path.join(scan_dir, '*.bin')))
    if not scan_files:
        print(f"Error: No .bin files found in {scan_dir}")
        return

    if load_labels:
        label_files = [
            os.path.join(label_dir, f"{os.path.splitext(os.path.basename(f))[0]}.txt") 
            for f in scan_files
        ]
    else:
        label_files = [None] * len(scan_files)
    
    # --- NEW: Pass start_frame to State ---
    state = PlayerState(scan_files, label_files, calib_transform, fps=fps, start_frame=start_frame)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='LiDAR Sequence Player (Press SPACE to pause)')
    
    vis.register_key_callback(32, state.toggle_pause) 
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # --- NEW: Load the specific start_frame initially ---
    # Use state.current_frame because the class might have reset it to 0 if it was out of bounds
    initial_scan_file = scan_files[state.current_frame]
    first_frame_points_raw = np.fromfile(initial_scan_file, dtype=np.float32).reshape(-1, 4)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_frame_points_raw[:, :3])
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    # ----------------------------------------------------
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
    
    # --- Camera Setup ---
    ctr = vis.get_view_control()
    viewpoint_path = os.path.join(SCRIPT_DIR, 'viewpoint.json')
    
    if os.path.exists(viewpoint_path):
        try:
            param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
            ctr.convert_from_pinhole_camera_parameters(param)
            print(f"Successfully loaded camera view from {viewpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load viewpoint. {e}")
            ctr.set_front([-1, 0, -0.3])
            ctr.set_lookat([20, 0, 0])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.05)
    else:
        ctr.set_front([-1, 0, -0.3])
        ctr.set_lookat([20, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.05)

    vis.register_animation_callback(lambda vis: state.advance_frame(vis, pcd))

    print(f"Starting visualization at {fps} FPS.")
    print("Check this terminal window for the current filename.")
    print("-" * 60)
    
    try:
        vis.run() 
    finally:
        vis.destroy_window()
        print("\nPlayback finished.")

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
    # --- NEW ARGUMENT ---
    parser.add_argument('--start_frame',
                        type=int,
                        default=0,
                        help='Frame index to start playback from (default: 0)')
    
    args = parser.parse_args()
    play_sequence(args.seq_dir, args.fps, args.start_frame)