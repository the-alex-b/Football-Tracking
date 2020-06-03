import os
import sys
path = './src'
sys.path.append(path)

from .src import model as modellib 
from . import playerdetection_maskrcnn as pldec
from Logger import Logger

class PlayerDetector:
    def __init__(self, useGpu):
        logger = Logger("Initializing player detector")

        logger.log("Initializing")
        MODEL_DIR = os.path.join(os.getcwd(), "PlayerDetector/trained_networks/")
        COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco_humanpose.h5")
        self.coco_config = pldec.InferenceConfig()
        self.coco_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.coco_config) 
        self.coco_model.load_weights(COCO_MODEL_PATH,by_name = True)
        logger.log("Initialized")

    def detect_players(self, frame):
        detectionLogger = Logger("Player detection")
        detectionLogger.log("Start")
        detection_result  = pldec.detectplayerskeypoints(frame, self.coco_config, self.coco_model)[0]

        # The feet coordinates are returned
        playersfeetcoos = pldec.findplayersfeetcoos(detection_result)

        # These coordinates are not yet used but can later be added to analysis
        playersneckcoos = pldec.findplayersneckcoos(detection_result)
        playerkeypoints = pldec.findplayerkeypointsall(detection_result)
        playertorsokeypoints = pldec.findplayerkeypointstorso(detection_result)
        playergeneralfeatures = pldec.findplayergeneralfeatures(detection_result)

        detectionLogger.log("Completed")

        return playersfeetcoos, detection_result