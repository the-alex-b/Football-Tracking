import cv2
import numpy as np
import time
import scipy.io as sio
from functions import CreatePix2PixModel

import twodvisualisation as twodvis 
from utilities.ANN import NNSearcher
from utilities.smoothing import smooth_traj_kalman
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

import os
import sys
path = './src'
sys.path.append(path)

import playerdetection_maskrcnn as pldec
import playertracking as pltrack
from frame import Frame
import model as modellib 



# ----- Loading trained models and datasets ----
pldec.config_tf()
# Create pix2pix model
pix2pix_model = CreatePix2PixModel(gpu=False)
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


nnsearcher = NNSearcher(database_features, anntype='faiss', useGpu=True) ## faiss
# nnsearcher = NNSearcher(database_features, anntype='flann') ## flann


# Coco model
MODEL_DIR = os.path.join(os.getcwd(), "./PreTrainedNetworks/MaskRCNN/")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco_humanpose.h5")
coco_config = pldec.InferenceConfig()
coco_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=coco_config) 
coco_model.load_weights(COCO_MODEL_PATH,by_name = True)

# --- Running the model -----

# Run on single frame
#img = cv2.imread('./input_footage/picture/16.jpg')
#Frame(img, database_features, database_cameras, model_points, model_line_index, pix2pix_model,1)

# Main video loop
cap = cv2.VideoCapture('./input_footage/video/1080_HQ.mp4')
start_time_in_ms = 0 ## where to begin reading the video from
cap.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)
end_time_in_ms = 10000
input_resolution = (1920,1080)
target_resolution = (1280,720)
i = 0
# Modulo i is used to skip frames. If you want to analyze full video set modulo to 1
modulo = 1
# Max number of frames that will be processed
max_number_of_frames = 100

frames = []

while (True):
     ret, fr = cap.read()
     if not ret:
         break 
     if (i % modulo == 0) and (cap.get(cv2.CAP_PROP_POS_MSEC) < end_time_in_ms):
         print("----"+str(i)+"----")
         start_time = time.time()
         fr = cv2.resize(fr, target_resolution, interpolation=cv2.INTER_CUBIC)
         fr = Frame(fr, database_features, database_cameras, model_points, model_line_index, pix2pix_model, nnsearcher, i, coco_config, coco_model, write_timestamps=True)
         fr.process()
         frames.append(fr)
         print("Analysis of frame {} took {} seconds".format(i, time.time()-start_time))
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break
     i = i + 1
     if i>=max_number_of_frames : break



# this step will assign object ids to detections 
res = pltrack.track(frames)

# Creating 2D coordinates
twodcoos = []
for j,fr in enumerate(frames): 
    twodcoo = fr.calculate_2d_coordinates(fr.final_homography, fr.playersfeetcoos[:,:2])
    twodcoo = np.c_[twodcoo,res[j][:,2]] # add labels
    twodcoos.append(twodcoo)
    twodvis.twodvisualisation(twodcoo[:,:2],twodcoo[:,2],j,'afterbasictrack',np.zeros((twodcoo.shape[0],3)))

# at this point, all usefull info has been stored in frame instance and twodcoos

# obj-indexed, pitch coo, foot positions
# fp: list of fr 2-d arrays of shape 1,2 
unique_objs = np.unique([item[2] for sublist in twodcoos for item in sublist])
fp = []
for obj in list(unique_objs):
    fp.append([twodcoos[fr][:,:2][twodcoos[fr][:,2] == obj] for fr in range(len(frames))])

# obj-indexed, boolean, has obj been detected in that particular frame
# list of 2-d arrays of shape fr,2 
fpp = [None] * len(fp)
was_detected = [None] * len(fp)
for k in range(len(fp)):
  fpp[k] = np.zeros((len(fp[k]),2))
  for i in range(len(fp[k])):
    if fp[k][i].shape[0] > 0:
      if fp[k][i].size == 2:
        fpp[k][i,:] = fp[k][i]
      if fp[k][i].size > 2:
        fpp[k][i,:] = fp[k][i][0,:]
  was_detected[k] = ~(fpp[k][:,0] == 0)

# obj-indexed, frame coo, torso polygon vertices
torsos = []
for obj in list(unique_objs):
    tmp = []
    for fr in range(len(frames)):
        pts = frames[fr].playertorsokeypoints[twodcoos[fr][:,2] == obj]  
        if pts.size != 0:
            pts = pts[0,...]
            pts = pts[ConvexHull(pts).vertices]  
        tmp.append(pts)
    torsos.append(tmp)

# checking if obj is actually on the pitch for most of the time (check for negative pitch coos)
# and removing those objs 
is_on_field = [np.all(np.mean(fpp[k],axis = 0) > np.array([0,0])) for k in range(len(fp))] 
to_keep = list(np.where(is_on_field)[0])
fpp = [fpp[i] for i in to_keep]
was_detected = [was_detected[i] for i in to_keep]
torsos = [torsos[i] for i in to_keep]

## determine objects which are likely to be the same via an overlap score (i.e. objects where a new index has been created for a previously detected object)
## need to specify 2 parameters -- min_frac (the minimum fraction of the total number of frames before a detection is called a genuine object)
##                              -- max_overlap (the maximum percentage of total frames in which double detections are allowed to overlap)

min_frac = 0.25
max_overlap = 0.05

wd = np.array(was_detected) * 1.0
tmp = np.tile(np.sum(wd,axis = -1),reps = (len(fpp),1))
where_false_dawn = np.array(np.where((np.minimum(1 - np.matmul(wd,wd.T) / tmp.T,1 - np.matmul(wd,wd.T) / tmp)*tmp.T*tmp) > 
                            min_frac * (1 - min_frac) * (1 - max_overlap) * len(frames) **2))

n_pairs = where_false_dawn.shape[1] // 2
pairs = where_false_dawn[:,range(n_pairs)]
thresh_merge = 15
for p in range(n_pairs):
  dd = np.min(cdist(fpp[pairs[0,p]][was_detected[pairs[0,p]],:],fpp[pairs[1,p]][was_detected[pairs[1,p]],:]))
  if dd < thresh_merge:
    fpp[pairs[0,p]] = (fpp[pairs[0,p]] * wd[pairs[0,p],:][:,None] + fpp[pairs[1,p]] * wd[pairs[1,p],:][:,None]) / np.sum(wd[pairs[:,p],:],axis = 0)[:,None]
    was_detected[pairs[0,p]] = (was_detected[pairs[0,p]] | was_detected[pairs[1,p]])
    fpp[pairs[0,p]][~np.isfinite(fpp[pairs[0,p]])] = 0
    fpp[pairs[1,p]] = -100 * np.ones_like(fpp[pairs[1,p]])
    tmp = ((wd[pairs[0,p]] + wd[pairs[1,p]]) == 1)
    for j in range(len(frames)):
      if ~(tmp[j] & (wd[pairs[0,p],j] == 1)):
        torsos[pairs[0,p]][j] = np.array([])
      if (tmp[j] & (wd[pairs[1,p],j] == 1)):
        torsos[pairs[0,p]][j] = torsos[pairs[1,p]][j]
fps = [None] * len(fpp)
for k in range(len(fpp)):
  try:
    fps[k] = smooth_traj_kalman(k, fpp, was_detected)[:,[0,2]]
  except:
    fps[k] = fpp[k].copy()


for j in range(len(frames)):
  twodvis.twodvisualisation([f[j] for f in fps],list(range(len(fps))),j,'aftersmoothing',np.zeros((len(fps),3)))


# determining the affiliation of the detection (teamA, teamB, referee)
# coversion to LAB will make the clustering less dependent on the lighting conditions
lab_images = [cv2.cvtColor(frames[k].frame,cv2.COLOR_RGB2Lab) for k in range(len(frames))]
to_use = [np.sum(was_detected[k]) > 3 for k in range(len(was_detected))]

col_obj = [None] * len(torsos)
im_size = frames[0].frame.shape[0:2]
for obj in list(np.arange(len(torsos))[to_use]):
for fr in range(len(frames)):
    if (torsos[obj][fr].size > 0):
    mask = np.zeros(im_size)
    cv2.fillPoly(mask, torsos[obj][fr].astype('int32')[None,:,:], 1) # this modifies the mask by eliminating the area inside the polygon
    mask = mask.astype(bool)
    if col_obj[obj] is None: # fr == min(np.where(was_detected[obj])[0]):
        col_obj[obj] = lab_images[fr][mask]
    else:
        col_obj[obj] = np.r_[col_obj[obj],lab_images[fr][mask]]

med_cols = np.array([np.median(col_obj[j],axis = 0) for j in list(np.arange(len(torsos))[to_use])])
kmeans = KMeans(n_clusters=3, random_state=0).fit(med_cols)
cluster_cols = [cv2.cvtColor(kmeans.cluster_centers_[j][None,None,:].astype(np.uint8),cv2.COLOR_Lab2RGB)[0,0,:] for j in range(3)]

colours = np.array(cluster_cols)[kmeans.labels_]

res = [f for idx, f in enumerate(fps) if to_use[idx]]
for j in range(len(frames)): 
    player_coos = [r[j] for r in res]
    twodvis.twodvisualisation(player_coos,list(range(len(player_coos))),j,'coloring',detection_color=colours)

# # Clean and clear
cap.release()
cv2.destroyAllWindows()