import argparse
import cv2
import numpy as np
import os
import torch
import glob
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\nWarning: matplotlib not found.")
    print("Please install it to generate plots: pip install matplotlib\n")
    plt = None


from pointpillars.utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range
from pointpillars.model import PointPillars

# --- Utils from your test.py needed for GT loading ---
from pointpillars.utils import bbox_camera2lidar
# ---------------------------------------------------

# --- Helper function for simple error calculation ---
def compute_center_distances(gt_boxes, pred_boxes):
    """
    Computes pairwise Euclidean distances between the centers (x, y, z) 
    of two sets of boxes.
    """
    pred_centers = pred_boxes[:, :3]
    gt_centers = gt_boxes[:, :3]
    diff = pred_centers[:, np.newaxis, :] - gt_centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    return dist_matrix


# --- Function to save plots ---
def save_plots(stats, classes, plot_dir):
    if plt is None:
        print("Skipping plot generation because matplotlib is not installed.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    class_names = list(classes.keys())
    
    # --- Data Preparation ---
    avg_errors = []
    precisions = []
    recalls = []
    
    for name in class_names:
        matches = stats['total_matches_found'][name]
        gts = stats['total_ground_truths'][name]
        preds = stats['total_predictions'][name]
        
        # Avg Error
        if matches > 0:
            avg_err = stats['total_distance_error'][name] / matches
        else:
            avg_err = 0
        avg_errors.append(avg_err)
        
        # Precision & Recall
        prec = matches / preds if preds > 0 else 0
        rec = matches / gts if gts > 0 else 0
        precisions.append(prec)
        recalls.append(rec)

    # --- 1. Plot: Average Center Distance Error ---
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, avg_errors, color=['blue', 'orange', 'green'])
    plt.title('Average Center Distance Error for Matched Detections')
    plt.ylabel('Distance (meters)')
    plt.ylim(0, max(avg_errors) * 1.2 + 0.1) # Dynamic y-limit
    for i, v in enumerate(avg_errors):
        plt.text(i, v + 0.01, f"{v:.3f}m", ha='center', fontweight='bold')
    plt.savefig(os.path.join(plot_dir, 'avg_center_error.png'))
    plt.close()

    # --- 2. Plot: Precision & Recall ---
    x = np.arange(len(class_names))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, precisions, width, label='Precision', color='cornflowerblue')
    rects2 = ax.bar(x + width/2, recalls, width, label='Recall', color='salmon')

    ax.set_ylabel('Score')
    ax.set_title('Approximate Precision and Recall by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.0)

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall.png'))
    plt.close()

    # --- 3. Plot: Detection Counts ---
    gts_counts = [stats['total_ground_truths'][name] for name in class_names]
    preds_counts = [stats['total_predictions'][name] for name in class_names]
    matches_counts = [stats['total_matches_found'][name] for name in class_names]
    
    width = 0.25
    x = np.arange(len(class_names))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, gts_counts, width, label='Ground Truths', color='forestgreen')
    rects2 = ax.bar(x, preds_counts, width, label='Predictions', color='goldenrod')
    rects3 = ax.bar(x + width, matches_counts, width, label='Matches', color='firebrick')

    ax.set_ylabel('Count')
    ax.set_title('Detection Counts by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    # Make y-axis logarithmic if counts are very different
    max_val = max(gts_counts + preds_counts + matches_counts)
    if max_val > 1000:
        ax.set_yscale('log')
        ax.set_ylabel('Count (log scale)')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'detection_counts.png'))
    plt.close()
    
    print(f"\nPlots saved to {plot_dir}")


def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
    }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    # === 1. Setup Model ===
    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(args.ckpt))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(args.ckpt, map_location=torch.device('cpu')))
    
    model.eval()

    # === 2. Setup Dataloader ===
    pc_dir = os.path.join(args.data_dir, args.split, 'velodyne')
    calib_dir = os.path.join(args.data_dir, args.split, 'calib')
    img_dir = os.path.join(args.data_dir, args.split, 'image_2') # For image_shape
    label_dir = os.path.join(args.data_dir, args.split, 'label_2') # For GT
    
    pc_files = sorted(glob.glob(os.path.join(pc_dir, '*.bin')))

    # --- Limit number of samples ---
    if args.num_samples is not None:
        pc_files = pc_files[:args.num_samples]
    # --- END ---

    print(f"Found {len(pc_files)} point cloud files to process.")
    print(f"Comparing against labels in: {label_dir}")

    # --- Stats tracking dictionaries ---
    stats = {
        'total_distance_error': {k: 0.0 for k in CLASSES},
        'total_matches_found': {k: 0 for k in CLASSES},
        'total_predictions': {k: 0 for k in CLASSES},
        'total_ground_truths': {k: 0 for k in CLASSES}
    }
    # --- END ---

    # === 3. Run Evaluation Loop ===
    for pc_path in tqdm(pc_files):
        file_id = os.path.basename(pc_path).split('.')[0]
        
        calib_path = os.path.join(calib_dir, file_id + '.txt')
        # --- FIX: Changed img__path to img_path ---
        img_path = os.path.join(img_dir, file_id + '.png')
        # --- END FIX ---
        gt_path = os.path.join(label_dir, file_id + '.txt') # Path to GT

        # if not os.path.exists(calib_path) or \
        #    not os.path.exists(img_path) or \
        #    not os.path.exists(gt_path):
        #     continue

        # --- Load Data (PC, Calib, Img) ---
        pc = read_points(pc_path)
        pc_torch = torch.from_numpy(pc)
        calib_info = read_calib(calib_path)
        # img = cv2.imread(img_path)
        # image_shape = img.shape[:2]
        
        # --- Run Inference ---
        with torch.no_grad():
            if not args.no_cuda:
                pc_torch = pc_torch.cuda()
            
            result_filter = model(batched_pts=[pc_torch], 
                                  mode='test')[0]

        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        #result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)
        result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)

        pred_lidar_bboxes = result_filter['lidar_bboxes']
        pred_labels = result_filter['labels']
        pred_scores = result_filter['scores']

        # --- Load Ground Truth (from test.py) ---
        gt_label_data = read_label(gt_path)
        
        gt_names = gt_label_data['name']
        valid_gt_mask = np.array([name in CLASSES for name in gt_names])
        if np.sum(valid_gt_mask) == 0:
            for label_idx in pred_labels:
                class_name = LABEL2CLASSES[label_idx]
                stats['total_predictions'][class_name] += 1
            continue 

        dimensions = gt_label_data['dimensions'][valid_gt_mask]
        location = gt_label_data['location'][valid_gt_mask]
        rotation_y = gt_label_data['rotation_y'][valid_gt_mask]
        gt_names_filtered = gt_names[valid_gt_mask]
        
        gt_labels_idx = np.array([CLASSES[name] for name in gt_names_filtered])
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
        # --- END GT LOADING ---

        # --- Calculate Simple Error Metrics ---
        for class_name, class_idx in CLASSES.items():
            pred_mask_class = (pred_labels == class_idx)
            gt_mask_class = (gt_labels_idx == class_idx)
            
            class_pred_boxes = pred_lidar_bboxes[pred_mask_class]
            class_gt_boxes = gt_lidar_bboxes[gt_mask_class]
            
            num_preds = class_pred_boxes.shape[0]
            num_gts = class_gt_boxes.shape[0]
            
            stats['total_predictions'][class_name] += num_preds
            stats['total_ground_truths'][class_name] += num_gts

            if num_preds == 0 or num_gts == 0:
                continue 

            dist_matrix = compute_center_distances(class_gt_boxes, class_pred_boxes)
            min_dist_to_gt = np.min(dist_matrix, axis=1) 
            
            match_threshold = args.match_thresh 
            matches = min_dist_to_gt < match_threshold
            
            num_matches = np.sum(matches)
            if num_matches > 0:
                stats['total_distance_error'][class_name] += np.sum(min_dist_to_gt[matches])
                stats['total_matches_found'][class_name] += num_matches

    # --- 4. Print Final Report ---
    print("\n--- 3D Detection Error Report ---")
    print(f"Matching Threshold: {args.match_thresh} meters")
    print("-----------------------------------")
    
    for class_name in CLASSES:
        print(f"Class: {class_name}")
        
        total_preds = stats['total_predictions'][class_name]
        total_gts = stats['total_ground_truths'][class_name]
        total_matches = stats['total_matches_found'][class_name]
        
        if total_matches > 0:
            avg_error = stats['total_distance_error'][class_name] / total_matches
            print(f"  > Average Center Distance Error: {avg_error:.3f} meters")
        else:
            print(f"  > Average Center Distance Error: N/A (No matches found)")
            
        recall = total_matches / total_gts if total_gts > 0 else 0
        precision = total_matches / total_preds if total_preds > 0 else 0
        
        print(f"  > Total Ground Truths: {total_gts}")
        print(f"  > Total Predictions:   {total_preds}")
        print(f"  > Matches Found:       {total_matches}")
        print(f"  > Simple Recall (approx): {recall:.2%}")
        print(f"  > Simple Precision (approx): {precision:.2%}")
        print("-----------------------------------")

    # --- 5. Save Plots (if requested) ---
    if args.plot_dir:
        save_plots(stats, CLASSES, args.plot_dir)
    # --- END NEW ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', 
                        help='your checkpoint for kitti')
    parser.add_argument('--data_dir', default='/home/antonio/datasets/kitti',
                        help='Path to the root of the KITTI dataset')
    parser.add_argument('--split', default='training',
                        help='Dataset split to evaluate (e.g., "training" or "testing")')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--match_thresh', type=float, default=2.0,
                        help='Matching threshold in meters for center distance')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    
    # --- NEW ARGUMENT ---
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory to save metric plots (e.g., build/plots)')
    # --- END NEW ---
    
    args = parser.parse_args()

    main(args)