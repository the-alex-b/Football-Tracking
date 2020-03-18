import cv2
import numpy as np
import time
import scipy.io as sio
from functions import CreatePix2PixModel

from frame import NewFrame

import sys
sys.path.append('./ExtractPitchLines')


# Create pix2pix model
pix2pix_model = CreatePix2PixModel()
# pix2pix_model = 0

# database
# HoG or Deep
data = sio.loadmat('./PreTrainedNetworks/SCCvSD/database_camera_feature_HoG.mat')
# data = sio.loadmat('./deepthings/feature_camera_10k.mat')

database_features = data['features']
database_cameras = data['cameras']

# World Cup soccer template
data = sio.loadmat('./PreTrainedNetworks/SCCvSD/worldcup2014.mat')
model_points = data['points']
model_line_index = data['line_segment_index']

# Ground truth homography
# data = sio.loadmat('./mat/test.mat')
# annotation = data['annotation']
# gt_h = annotation[0][query_index][1]  # ground truth

# img = cv2.imread('./16.jpg')

# NewFrame(img, database_features, database_cameras, model_points, model_line_index, pix2pix_model)




# # Main video loop
cap = cv2.VideoCapture('./input_footage/video/540_LQ.mp4')
i = 0 

# cap = cv2.VideoCapture('./video/1080')
while (True):
    i = i + 1
    # print(i)
    ret, frame = cap.read()

    if not ret:
        break
    if i % 500 == 0:
        print("----"+str(i)+"----")
        print("Analyze the frame and display result")
        NewFrame(frame, database_features, database_cameras, model_points, model_line_index, pix2pix_model)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean and clear
cap.release()

cv2.destroyAllWindows()
