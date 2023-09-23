# The Devil is in the Details: RGB-LiDAR Late Fusion for Long-Tailed 3D Detection
Late Fusion for LiDAR and RGB detections.

## Get started
1. Clone the LT3D reposityory from https://github.com/neeharperi/LT3D and the LT3D nuscenes-devkit from https://github.com/neeharperi/nuscenes-lt3d.git

2. You should mkdir a file folder in  "./tools" (e.g. "fusion" folder) containing all the files from this project when you need to use mmdetection3d framework.

3. You should get 3D detections and 2D detections results with .pkl and .txt format respectively. The .pkl results from lidar should be getted by "tools/test.py" in mmdetection3d codebase and the .txt results from camera should be getted by some 2D dectector methods (e.g. DINO). I will provide above two results in lidar_results (https://drive.google.com/file/d/1qbfghLnyBHl_OrY6h4qfKxQFifm2tutd/view?usp=sharing) and camera_results (https://drive.google.com/file/d/1Z8B_ljPVnZ9d2-k1_bjs-jq569hcsv2D/view?usp=share_link) getted by Centerpoint (Hierarchy) and DINO respectively.


4. Calibrate the 18 classes for long-tailed derections, you can get the best metrics (AP) for each classes from the logs.

```shell
python tools/fusion/main_tune.py

```

5. Add overlapping mechanism for evaluating the metrics (mAP).

```shell
python tools/fusion/main.py
```

6. You can get metircs (mAP) for "Many, medium, few" classes.

```shell
python get_LCA0_mmf.py
```

7. The 'cal.py' script contain precomputed calibrations by different 3D (e.g. Centerpoint) and 2D detections (e.g. DINO).
