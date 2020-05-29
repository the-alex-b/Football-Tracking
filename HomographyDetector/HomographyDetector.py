# Pix2Pix Model
# Faiss matcher
import scipy.io as sio
from .Pix2PixModel.CreatePix2PixModel import CreatePix2PixModel
from .ANN import NNSearcher
import os
from Logger import Logger

class HomographyDetector:
    def __init__(self, useGpu):
        logger = Logger("Homography Detector Initializer")
        logger.log("Initializing")

        self.pix2pix_model = CreatePix2PixModel(gpu=useGpu)


        # Datasets
        path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'SCCvSD/trained_networks'))

        data =  sio.loadmat(path+'/database_camera_feature_HoG.mat')
        self.database_features = data['features']
        self.database_cameras = data['cameras']

        data = sio.loadmat(path+'/worldcup2014.mat')
        self.model_points = data['points']
        self.model_line_index = data['line_segment_index']

        # Create a Faiss Searcher
        self.nnsearcher = NNSearcher(self.database_features, anntype='faiss', useGpu=useGpu)
        logger.log("Initialized")