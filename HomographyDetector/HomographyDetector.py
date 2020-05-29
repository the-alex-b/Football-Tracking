# Pix2Pix Model
# Faiss matcher
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import scipy.io as sio
from .Pix2PixModel.CreatePix2PixModel import CreatePix2PixModel
from .ANN import NNSearcher
import os
from Logger import Logger

from .SCCvSD_Utils.synthetic_util import SyntheticUtil
from .SCCvSD_Utils.iou_util import IouUtil
from .SCCvSD_Utils.projective_camera import ProjectiveCamera

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

        # Set template size and resolution
        self.template_h = 74
        self.template_w = 115

        self.resolution = (1280,720)

    def detect_homography(self, frame):
        # Main function that will call all other functions on this object to determine the final homography
        pix_lines = self.extract_pitch_lines(frame)

        # Do some transformations on the pix lines 
        pix_lines = cv2.cvtColor(pix_lines, cv2.COLOR_BGR2GRAY)
        pix_lines = cv2.threshold(pix_lines, 127, 255, cv2.THRESH_BINARY)[1]


        # cv2.imshow('lines', pix_lines)
        features = self.generate_hog_features(pix_lines)
        pix_lines, retrieved_image, retrieved_homography = self.retrieve_a_camera(pix_lines, features)
        
        # Refine the homography to determine final homography
        homography = self.refine_homography(pix_lines, retrieved_image, retrieved_homography) 

        return homography


    def extract_pitch_lines(self, frame):
        load_size = 256
        input_image = {}

        # Resize frame for pitch detection
        resized_image = np.array(Image.fromarray(frame).resize((load_size,load_size),resample = Image.BICUBIC))
        resized_image = Image.fromarray(np.concatenate((resized_image,np.zeros_like(resized_image)),axis = 1))
        resized_image.convert("RGB")
        resized_image = resized_image.resize((load_size *2, load_size), Image.BICUBIC)
        resized_image = transforms.ToTensor()(resized_image)

        w_total = resized_image.size(2)
        w = int(w_total/2)

        input_image['A'] = resized_image[:,:,0:load_size]
        input_image['B'] = resized_image[:,:,load_size:]
        
        input_image['A'] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image['A']).unsqueeze(0)
        input_image['B'] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image['B']).unsqueeze(0)
        
        input_image['A_paths'] = None
        input_image['B_paths'] = None

        self.pix2pix_model.set_input(input_image)
        self.pix2pix_model.test()
        
        vis = self.pix2pix_model.get_current_visuals()
        
        # self.pix_original = vis['real_A']
        # self.pix_no_crowd = vis['fake_C']
        pix_lines = vis['fake_D']
        return pix_lines

    def generate_hog_features(self, pix_lines):
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (32, 32)
        cell_size = (32, 32)
        n_bins = 9
        im_h, im_w = 180, 320

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
        temp_frame = cv2.resize(pix_lines, (im_w,im_h))

        # compute hog and transpose so that shape matches the database features
        features = hog.compute(temp_frame).T
        return features 


    def retrieve_a_camera(self, pix_lines, features):

        # Actual seeking part
        retrieved_index = self.nnsearcher.seek_nn(features)
        retrieved_camera_data = self.database_cameras[retrieved_index]

        u, v, fl = retrieved_camera_data[0:3]
        rod_rot = retrieved_camera_data[3:6]
        cc = retrieved_camera_data[6:9]
        retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

        # Which homography to use? One corrected for the field or not (seems to mess with visualization, for calculation the first one might be better)
        retrieved_homography = IouUtil.template_to_image_homography_uot(retrieved_camera, self.template_h, self.template_w)

        # Turn the camera to an image with a template
        retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, self.model_points, self.model_line_index, im_h=self.resolution[1], im_w=self.resolution[0], line_width=4)
        
        pix_lines = cv2.resize(pix_lines, self.resolution, interpolation=cv2.INTER_CUBIC)[:, :, None]

        return pix_lines, retrieved_image, retrieved_homography

    def refine_homography(self, pix_lines, retrieved_image, retrieved_homography):
         # TODO: this step should be optimized (if possible), most time is consumed by determining the warp by the opencv find transform algorithm.

        # Refine using lucas kanade algorithm
        dist_threshold = 150 # Not really sure how this works, lower value leads to higher execution time.
        query_dist = SyntheticUtil.distance_transform(pix_lines)
        retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)
        query_dist[query_dist > dist_threshold] = dist_threshold
        retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

        warp = SyntheticUtil.find_transform(retrieved_dist, query_dist)
        # This is the homography mapping the field to a top down view
        homography = warp@retrieved_homography

        # self.refined_retrieved_image = cv2.warpPerspective(retrieved_image, warp, self.resolution)
        
        # self.picstosave.append(('refined',self.refined_retrieved_image))
        return homography