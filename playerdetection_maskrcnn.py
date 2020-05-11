import cv2
import numpy as np
import tensorflow as tf
import random
import os
import math
import skimage.io
import matplotlib.pyplot as plt
import coco
#import utils
import model as modellib
import visualize
from model import log
#from google.colab.patches import cv2_imshow
from tqdm import tqdm

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "./PreTrainedNetworks/MaskRCNN/")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco_humanpose.h5")

def resize_images():
    pass

def config_tf():
    #config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.device('/device:GPU:0')

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

def detectplayerskeypoints(image):
    print('starting inference')
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inference_config)
    model.load_weights(COCO_MODEL_PATH,by_name = True) #, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"

    #gap = int(frames.shape[0] / (n_det_frames-1))

    #frames_to_use = np.uint8(np.linspace(0,gap * (n_det_frames-1),n_det_frames))

    #images = frames[frames_to_use,...].copy()[:,:,:,::-1]
    #results = [None] * n_det_frames
    #images = [images[j] for j in range(n_det_frames)]
    #for j in tqdm(range(n_det_frames)):
    #results[j] = model.detect_keypoint([images[j]], verbose=0)
    result = model.detect_keypoint([image], verbose=0)

    #results = [results[j][0] for j in range(n_det_frames)]
    return result

def findplayersfeetcoos(result): 
    fp = np.mean(result['keypoints'][...,[0,1]][:,[15,16],:],axis = 1)[...,0:2]
    foot_positions = np.c_[fp,np.ones(fp.shape[0])]
    return foot_positions

def findplayersneckcoos(result): 
    tp = np.mean(result['keypoints'][...,[0,1]][:,[0,1],:],axis = 1)[...,0:2]
    neck_positions = np.c_[tp,np.ones(tp.shape[0])]
    return neck_positions

def findplayerkeypointsall(result):
    all_keypoints = result['keypoints']
    return all_keypoints

def findplayergeneralfeatures(result):
    general_features = result['features']
    return general_features

def findplayerkeypointstorso(result): 
    torso_keypoints = result['keypoints'][...,[0,1]][:,[1,2,3,4,5,6,7,8,11],:]
    return torso_keypoints

def save_result(r, image, i):
    print('saving res')
    class_names = ['BG', 'person']
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figoutpath='./output_images/{}_pldetection.jpg'.format(i))
    # Change to save
    return
