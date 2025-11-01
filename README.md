# PointPillars for TruckScenes

This repository adapts the original [PointPillars](https://github.com/zhulf0804/PointPillars) implementation to work seamlessly with the **ManTruckScenes (TruckScenes)** LiDAR dataset.

The TruckScenes dataset is first converted into the KITTI format using the [truckscenes_to_kitti](https://github.com/AntonioLuisGH/truckscenes_to_kitti) conversion pipeline.  
Once converted, this version of PointPillars can be used for training, evaluation, and visualization on TruckScenes data.

## Key Modifications

- **`play_sequence.py`** — visualize LiDAR data sequences in motion.  
- **`evaluate_pp_simple.py`** — simplified evaluation script that works without a `.pkl` file.  
- **`train.py`** — made more robust for non-KITTI datasets (e.g., TruckScenes).  
- **`test.py`** — modified to skip filtering for greater flexibility.  
- **`preprocess_kitti.py`**, **`process.py`**, and **`vis_o3d.py`** — adapted for broader dataset compatibility.

## Acknowledgment

This project is based on the original [PointPillars implementation by zhulf0804](https://github.com/zhulf0804/PointPillars).
