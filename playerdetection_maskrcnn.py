import cv2
import numpy as np
import tensorflow as tf
import random
import os
import math
import skimage.io
import matplotlib.pyplot as plt
import scr.coco as coco
#import utils
import scr.model as modellib
import scr.visualize as visualize
from scr.model import log
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

def infer(image):
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


def save_result(r, image):
    print('saving res')
    class_names = ['BG', 'person']
    # visualize.display_keypoints(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    visualize.display_keypoints(image, r['rois'], r['keypoints'], r['class_ids'], class_names)
    # Change to save
    return
