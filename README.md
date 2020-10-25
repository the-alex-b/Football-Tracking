# Football Tracking with Python
*Turning broadcast footage into a 3d digital representation*

This repository combines machine learning and computer vision to turn broadcasted match footage into a valid 3d digital representation. Below a short description of the inner workings. If you have any questions on getting stuff working or if you want to contribute feel free to let us know!


## How it works
Code execution can be split in two distinct steps. An extraction step that will use neural networks and computation heavy analysis to extract features from each frame, and an analysis step that will use the extracted features to determine player trajectories, create 2d top down view etc. This splitted approach will allow faster and easier development in analysis step since it will not be constrained by GPU usage.


### 1. Extraction
Extraction step can be done by switching run_extraction=True in run.py. This will yield an extracted_frames.p data file. This data file can be used as an import for step 2.

You will need GPU (or Google Colab) to run this step,

### 2. Analysis
Analysis is done based on a .p data file. Homography is determined, player trajectories are created and visualization happens here. 



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
- [ ] Standardize output to match metrica format: https://github.com/metrica-sports/sample-data


## How to run and required files (WIP)

To run this code you will need some files:

#### COCO Humanpose trained network:
Get the mask_rcnn_coc_humanpose.h5 file from: https://github.com/Superlee506/Mask_RCNN_Humanpose/releases and move it to  ./PlayerDetector/trained_networks

#### Two-GAN Network 
https://github.com/lood339/pytorch-two-GAN
https://github.com/lood339/SCCvSD

Train the two-gan network and move network paths:

* Linedetection
Move detec_latest_net_G.pth and seg_latest_net_G.pth to ./HomographyDetector/Pix2PixModel/trained_networks

* Homography and camera calibration
Move database_camera_feature_HoG.mat and worldcup2014.mat to ./HomographyDetector/SCCvSD/trained_networks

#### Yolo Darknet
Get the weights from https://pjreddie.com/darknet/yolo/ and place them at ./BallExtractor/YoloDarknet/ 

#### Footage
Some football footage to analyze!

## Relevant repositories

This code is based upon and heavily influenced by the following repositories:

* https://github.com/lood339/pytorch-two-GAN
* https://github.com/lood339/SCCvSD
* https://github.com/pjreddie/darknet



