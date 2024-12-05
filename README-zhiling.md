# FAST-MIX-SLAM
This is MIX SLAM, a faster version folked from ORB-SLAM2 (see https://github.com/raulmur/ORB_SLAM2 and the README-ORB-SLAM2.md in this repo). We put the direct tracking in SVO to accelerate the feature matching in ORB-SLAM2. We can get an average 3x speed up and keep almost same accuracy. In addition we also support monocular Visual-Inertial SLAM (VI-SLAM), following idea proposed in Raul's paper.

# Dependency
If you are using ubuntu, just type "./install_dependency.sh" to install all the dependencies except pangolin.

- Pangolin (for visualization): https://github.com/stevenlovegrove/Pangolin 
- Eigen3: sudo apt-get install libeigen3-dev
- g2o: sudo apt-get install libcxsparse-dev libqt4-dev libcholmod3.0.6 libsuitesparse-dev qt4-qmake 
- OpenCV: sudo apt-get install libopencv-dev
- glog (for logging): sudo apt-get install libgoogle-glog-dev

# Compile
run "./generate.sh" to compile all the things, or follow the steps in generate.sh

# Examples
We support all the examples in the original ORB-SLAM2, and also the monocular-inertial examples. You can try the EUROC dataset (http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and run the monocular/mono-inertial examples by typing:

cd ~/catkin_ws/src/FAST-MIX-SLAM
-----------easy--------------
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.bin Examples/Monocular/EuRoC.yaml /home/wch/Dataset/EUROC/V1_01_easy/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/V101.txt 
```
----------medium----------
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.bin Examples/Monocular/EuRoC.yaml /home/wch/Dataset/EUROC/V1_02_medium/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/V102.txt 
--------------------------

just like in ORB-SLAM2. For VIO examples, try: 
-----------easy----------
./Examples/Monocular/mono_euroc_vins ./Vocabulary/ORBvoc.bin ./Examples/Monocular/EuRoC.yaml /home/wch/Dataset/EUROC/V1_01_easy/mav0/cam0/data ./Examples/Monocular/EuRoC_TimeStamps/V101.txt /home/wch/Dataset/EUROC/V1_01_easy/mav0/imu0/data.csv
-----------medium-------
 ./Examples/Monocular/mono_euroc_vins ./Vocabulary/ORBvoc.bin ./Examples/Monocular/EuRoC.yaml /home/wch/Dataset/EUROC/V1_02_medium/mav0/cam0/data ./Examples/Monocular/EuRoC_TimeStamps/V102.txt /home/wch/Dataset/EUROC/V1_02_medium/mav0/imu0/data.csv
-----------difficult--------
 ./Examples/Monocular/mono_euroc_vins ./Vocabulary/ORBvoc.bin ./Examples/Monocular/EuRoC.yaml /home/wch/Dataset/EUROC/Monocular_data/V1_03_difficult/mav0/cam0/data ./Examples/Monocular/EuRoC_TimeStamps/V103.txt /home/wch/Dataset/EUROC/Monocular_data/V1_03_difficult/mav0/imu0/data.csv

to run the VIO case.


------------RGBD----------
cd ~/catkin_ws/src/FAST-MIX-SLAM/Examples/RGB-D
python2 associate.py /home/wch/Dataset/rgbd_dataset_freiburg2_pioneer_360/rgb.txt /home/wch/Dataset/rgbd_dataset_freiburg2_pioneer_360/depth.txt > associations_my.txt
./rgbd_tum ../../Vocabulary/ORBvoc.bin TUM2.yaml ../../rgbd_dataset_freiburg2_pioneer_360 associations_my.txt
---------------------------

----------------TUM-----------------------
cd ~/catkin_ws/src/FAST-MIX-SLAM
./Examples/Monocular/mono_tum  Vocabulary/ORBvoc.bin ./Examples/Monocular/TUM1.yaml  /home/wch/Dataset/TUM/rgbd_dataset_freiburg1_xyz
./Examples/Monocular/mono_tum  Vocabulary/ORBvoc.bin ./Examples/Monocular/TUM1.yaml  /home/wch/Dataset/TUM/rgbd_dataset_freiburg1_room
./Examples/Monocular/mono_tum  Vocabulary/ORBvoc.bin ./Examples/Monocular/TUM1.yaml  /home/wch/Dataset/TUM/rgbd_dataset_freiburg1_360
./Examples/Monocular/mono_tum  Vocabulary/ORBvoc.bin ./Examples/Monocular/TUM_Test2.yaml  /home/wch/Dataset/TUM/TUM2
--------------------------------------

//---------------------评估轨迹MIX-------------------------
workon evaluation
evo_ape euroc /home/wch/Dataset/EUROC/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv /home/wch/Dataset/Trajectory/MIX/KeyFrameTrajectory_v101_vio_.txt  -va --plot --save_results results/ORB.zip
evo_rpe euroc /home/wch/Dataset/EUROC/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv /home/wch/Dataset/Trajectory/MIX/KeyFrameTrajectory.txt  -va --plot --save_results results/ORB.zip
evo_ape euroc /home/wch/Dataset/EUROC/V1_02_medium/mav0/state_groundtruth_estimate0/data.csv /home/wch/Dataset/Trajectory/MIX/KeyFrameTrajectory_v102.txt  -va --plot --save_results results/ORB.zip

//----------------------评估轨迹ORB---------------------------------
workon evaluation
evo_ape euroc /home/wch/Dataset/EUROC/V2_01_easy/mav0/state_groundtruth_estimate0/data.csv /home/wch/Dataset/Trajectory/ORB/KeyFrameTrajectory_v201.txt  -va --plot --save_results results/ORB.zip
evo_rpe euroc /home/wch/Dataset/EUROC/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv /home/wch/Dataset/Trajectory/ORB/KeyFrameTrajectory.txt  -va --plot --save_results results/ORB.zip

