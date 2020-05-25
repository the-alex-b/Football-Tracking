import os
import sys
path = './src'
sys.path.append(path)

from .src import model as modellib 
from . import playerdetection_maskrcnn as pldec


class PlayerDetector:
    def __init__(self, useGpu):
        print("Initializing player detector")


        MODEL_DIR = os.path.join(os.getcwd(), "PlayerDetector/trained_networks/")
        COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco_humanpose.h5")
        coco_config = pldec.InferenceConfig()
        self.coco_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=coco_config) 
        self.coco_model.load_weights(COCO_MODEL_PATH,by_name = True)
        
