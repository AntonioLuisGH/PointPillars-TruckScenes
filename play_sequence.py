import open3d as o3d
import numpy as np
import os
import glob
import argparse
import time  # --- NEW: Import time module ---

# --- Get the absolute path of this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class PlayerState:
    """Helper class to store the player's state"""
    # --- NEW: Added 'fps' parameter ---
    def __init__(self, scan_files, fps):
        self.is_paused = False
        self.scan_files = scan_files
        self.total_frames = len(scan_files)
        self.current_frame = 0
        
        # --- NEW: FPS control variables ---
        self.fps = fps
        self.frame_duration = 1.0 / self.fps
        self.last_frame_time = time.time()
        # ----------------------------------

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
        
        # Always poll for events (like key presses, window close)
        vis.poll_events()

        # --- Check for pause ---
        if self.is_paused:
            vis.update_renderer() # Keep window responsive
            return True # Tell Open3D to keep running

        # --- NEW: Check for FPS timing ---
        current_time = time.time()
        if (current_time - self.last_frame_time) < self.frame_duration:
            vis.update_renderer() # Keep window responsive
            return True # Not time for next frame, but keep running
        
        # --- If not paused and time is up, advance the frame ---
        self.last_frame_time = current_time # Reset timer
        # -----------------------------------
        
        try:
            file_path = self.scan_files[self.current_frame]
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            
            # Update geometry
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.paint_uniform_color([0.8, 0.8, 0.8])
            vis.update_geometry(pcd)
            
            # Advance frame index
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                print("End of sequence. Looping.")
                self.current_frame = 0
                
        except Exception as e:
            print(f"Error in animation callback: {e}")
            return False # Stop animation on error

        # Frame was advanced, so update the renderer
        vis.update_renderer()
        return True # Tell Open3D to keep running


# --- NEW: Added 'fps' argument ---
def play_sequence(sequence_dir, fps):
    """
    Plays a LiDAR sequence from a directory.
    Assumes the directory contains a 'velodyne' subdir.
    """
    
    scan_dir = os.path.join(sequence_dir, 'velodyne')
    if not os.path.exists(scan_dir):
        print(f"Error: Directory not found: {scan_dir}")
        return

    scan_files = sorted(glob.glob(os.path.join(scan_dir, '*.bin')))
    if not scan_files:
        print(f"Error: No .bin files found in {scan_dir}")
        return

    print(f"Found {len(scan_files)} scans.")
    print("--- Press SPACEBAR in the window to Pause/Resume ---")
    
    # --- NEW: Pass 'fps' to the state ---
    state = PlayerState(scan_files, fps=fps)

    # --- Use VisualizerWithKeyCallback ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='LiDAR Sequence Player (Press SPACE to pause)')
    
    vis.register_key_callback(32, state.toggle_pause) # 32 = SPACE
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    # --- Create Geometries ---
    first_frame_points = np.fromfile(scan_files[0], dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_frame_points[:, :3])
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    
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
        ctr.set_front([0, -1, 0.2]); ctr.set_lookat([0, 0, 0]); ctr.set_up([0, 0, 1]); ctr.set_zoom(0.1)

    # --- Register the animation callback ---
    vis.register_animation_callback(lambda vis: state.advance_frame(vis, pcd))

    print(f"Starting visualization at {fps} FPS. Close the window to exit.")
    
    # --- Run the blocking visualizer ---
    try:
        vis.run() # This blocks the main thread until the window is closed
    finally:
        vis.destroy_window()
        print("Playback finished or window closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a LiDAR sequence.')
    parser.add_argument('--seq_dir', 
                        type=str, 
                        required=True,
                        help='Path to the sequence directory (e.g., /home/antonio/datasets/nuscenes_kitti/0061)')
    # --- NEW: FPS command-line argument ---
    parser.add_argument('--fps',
                        type=int,
                        default=10,
                        help='Playback frames per second (default: 10)')
    
    args = parser.parse_args()
    # --- NEW: Pass 'args.fps' to the function ---
    play_sequence(args.seq_dir, args.fps)