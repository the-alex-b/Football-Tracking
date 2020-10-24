# Football Tracking
*Turning broadcast footage into a 3d digital representation*

This repository combines machine learning and computer vision to turn broadcasted match footage into a valid 3d digital representation. Below a short description of the inner workings. If you have any questions on getting stuff working or if you want to contribute feel free to let us know!


## How it works (WIP)

### Extraction (WIP)


### Analysis (WIP_



## To-Do's

- [X] Determining basic field homography
    - [ ] Improving and smoothing detected homography
- [X] Detecting and tracking players 
    - [ ] Detecting and storing teams
    - [ ] Detecting player trajectories
    - [ ] Fixing occlusions
- [X] Tracking the ball 
    - [ ] Improving tracking
    - [ ] Creating a trajectory
- [X] Combine homography and player coordinates to create digital representation
    - [X] 3d -> 2d (top down) view
- [ ] Score and time detection
- [ ] Setting up a file mirror for relevant NN weights and files.
    - [ ] Create a script to put these files in the right place
- [ ] Improve documentation
    - [ ] Add footage and images



## How to run and required files (WIP)

To run this code you will need some files:

* COCO Humanpose trained network:

* Linedetection files

* SCCvSD camera features and HoG file


* Yolo Darknet weights
Get it: https://pjreddie.com/darknet/yolo/ and place the weights at ./BallExtractor/YoloDarknet/

* Some football footage to analyze!

<!-- # Get required repositories
# ! ./GetTwoGanModelFromGithub.sh
# ! ./makepix2pixdirs.sh


# Install libraries
!pip install pyflann-py3
!pip install pykalman
!pip install faiss-gpu

# Get files from drive
# COCO Network
!cp ../drive/My\ Drive/FootballNetworks/mask_rcnn_coco_humanpose.h5 ./PlayerDetector/trained_networks

# SCCvSD
!cp ../drive/My\ Drive/FootballNetworks/database_camera_feature_HoG.mat ./HomographyDetector/SCCvSD/trained_networks
!cp ../drive/My\ Drive/FootballNetworks/worldcup2014.mat ./HomographyDetector/SCCvSD/trained_networks

# Linedetection
# !mkdir ./checkpoints/Linedetection
!cp ../drive/My\ Drive/FootballNetworks/detec_latest_net_G.pth ./HomographyDetector/Pix2PixModel/trained_networks
!cp ../drive/My\ Drive/FootballNetworks/seg_latest_net_G.pth ./HomographyDetector/Pix2PixModel/trained_networks

# Input footage
!cp ../drive/My\ Drive/FootballNetworks/video/1080_HQ.mp4 ./input_footage/video
# !cp ../drive/My\ Drive/FootballNetworks/video/540_HQ.mp4 ./input_footage/video
# !cp ../drive/My\ Drive/FootballNetworks/video/540_LQ.mp4 ./input_footage/video


# COPY Yolo Weights
%cd /content/Football-Tracking/
!cp ../drive/My\ Drive/FootballNetworks/yolov3.weights ./BallExtractor/YoloDarknet

# # Build Yolo
%cd /content/Football-Tracking/BallExtractor/YoloDarknet/
!make -->







<!-- OLD! -->
<!-- 1. Create a conda environment based upon environment.yml 
2. Run GetTwoGanModelFromGithub.sh to pull the pytorch-two-GAN repo, you will need this to extract pitch lines from frames.
3. Train the two-gan network and move network paths to ./checkpoints/Linedetection (see readme in folder)
4. Acquire the camera features and worldcup file from SCCvSD repo. (See ./PreTrainedNetworks/SCCvSD)
5. Put your footage in ./input_footage/video or ./input_footage/picture
6. Reference your footage (or still image) in the run.py file. This is the entry point.
7. Place mask_rcnn_coco_humanpose.h5 in ./PreTrainedNetworks/MaskRCNN, after downloading from https://github.com/Superlee506/Mask_RCNN_Humanpose/releases  
8. Run makepix2pixdirs.sh to create the pix2pix folders 
9. python run.py


Extra (should be fixed but seems to be necessary to get this working right now):
* Open ./ExtractPitchLines/options/base_options.py, remove required flag from --dataroot argument & set --gpu-ids flag to default '-1'(when not using gpu)
* Folder ./ExtractPitchLines/datasets should contain a folder name soccer_seg_detecion with test, train_phase_1, train_phase_2 & val folders (pix2pix requirement.) -->


## Relevant repositories

This code is based upon and heavily influenced by the following repositories:

* https://github.com/lood339/pytorch-two-GAN
* https://github.com/lood339/SCCvSD
* https://github.com/pjreddie/darknet



