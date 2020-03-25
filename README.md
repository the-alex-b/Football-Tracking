# Football Tracking (Work in Progress)
*Turning broadcast footage into a 3d digital representation*



### Steps/To-Do's

- [X] Determining basic field homography
- [ ] Refine homography by applying Lucas-Kanade algorithm
- [ ] Detecting and tracking players 
- [ ] Track the ball 
- [ ] Combine homography and player coordinates to create digital representation



### How to run:

1. Create a conda environment based upon environment.yml 
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
* Folder ./ExtractPitchLines/datasets should contain a folder name soccer_seg_detecion with test, train_phase_1, train_phase_2 & val folders (pix2pix requirement.)


### Relevant repositories

This code is based upon and heavily influenced by the following repositories:

* https://github.com/lood339/pytorch-two-GAN
* https://github.com/lood339/SCCvSD



