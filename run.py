import cv2
import numpy as np
import time
import scipy.io as sio
from functions import CreatePix2PixModel

import sys
path = 'PATHTO/Keypoints-of-humanpose-with-Mask-R-CNN'
sys.path.append(path)

from frame import Frame

# ----- Loading trained models and datasets ----

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


# --- Running the model -----

# Run on single frame
img = cv2.imread('./input_footage/picture/16.jpg')
Frame(img, database_features, database_cameras, model_points, model_line_index, pix2pix_model,1)

# # Main video loop
# cap = cv2.VideoCapture('./input_footage/video/540_LQ.mp4')
# i = 0 

# # cap = cv2.VideoCapture('./video/1080')
# while (True):
#     i = i + 1
#     # print(i)
#     ret, frame = cap.read()

#     if not ret:
#         break
#     # Modulo i is used to skip frames. If you want to analyze full video set modulo to 1
#     modulo = 100
#     if i % modulo == 0:
#         print("----"+str(i)+"----")
#         Frame(frame, database_features, database_cameras, model_points, model_line_index, pix2pix_model, i)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Clean and clear
# cap.release()
# cv2.destroyAllWindows()
