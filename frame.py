import cv2
import numpy as np
import pyflann

from PIL import Image

from SCCvSD_Utils.synthetic_util import SyntheticUtil
from SCCvSD_Utils.iou_util import IouUtil
from SCCvSD_Utils.projective_camera import ProjectiveCamera

import playerdetection_maskrcnn as pldec
import sys

import torchvision.transforms as transforms

class Frame:
    def __init__(self, frame, database_features, database_cameras, model_points, model_line_index, pix2pix_model, iteration):
        # Initialize variables
        self.frame = frame
        self.original = frame
        self.features = None
        self.temp_frame = None
        self.i = iteration
        self.homography = None

        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        self.database_features = database_features
        self.database_cameras = database_cameras
        self.model_points = model_points
        self.model_line_index = model_line_index

        self.pix2pix_model = pix2pix_model
        self.input_image = {}

        self.template_h = 74
        self.template_w = 115

        # Initialize a flann
        self.flann = pyflann.FLANN()
        self.retrieved_image = None

        # Analyze the frame
        # Player detection
        # ---------------
        pldec.config_tf()
        result = pldec.detectplayersbox(frame) # returns list of detections over multiple frames
        pldec.save_result(result[0], frame, self.i) # only 1 frame being processed
        playercoos = pldec.findplayerscoos(result[0])
        # ---------------

        self.extract_pitch_lines()
        self.create_line_image()
        self.generate_hog_features()
        self.retrieve_a_camera()
        
        # Visualize or save the frame:
        # self.visualize()
        self.save()


    # Pix2Pix model to extract lines and field area 
    def extract_pitch_lines(self):
        load_size = 256
        # Resize frame for pitch detection
        self.resized_image = np.array(Image.fromarray(self.frame).resize((256,256),resample = Image.BICUBIC))
        self.resized_image = Image.fromarray(np.concatenate((self.resized_image,np.zeros_like(self.resized_image)),axis = 1))

        self.resized_image.convert("RGB")
        self.resized_image = self.resized_image.resize((load_size *2, load_size), Image.BICUBIC)
        self.resized_image = transforms.ToTensor()(self.resized_image)

        w_total = self.resized_image.size(2)
        w = int(w_total/2)

        self.input_image['A'] = self.resized_image[:,:,0:load_size]
        self.input_image['B'] = self.resized_image[:,:,load_size:]
        
        self.input_image['A'] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.input_image['A']).unsqueeze(0)
        self.input_image['B'] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.input_image['B']).unsqueeze(0)
        
        self.input_image['A_paths'] = None
        self.input_image['B_paths'] = None

        self.pix2pix_model.set_input(self.input_image)
        self.pix2pix_model.test()
        
        vis = self.pix2pix_model.get_current_visuals()
        
        self.pix_original = vis['real_A']
        self.pix_no_crowd = vis['fake_C']
        self.pix_lines = vis['fake_D']

    def create_line_image(self):
        self.pix_lines = cv2.cvtColor(self.pix_lines, cv2.COLOR_BGR2GRAY)
        self.pix_lines = cv2.threshold(self.pix_lines, 127, 255, cv2.THRESH_BINARY)[1]

    def generate_hog_features(self):
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (32, 32)
        cell_size = (32, 32)
        n_bins = 9
        im_h, im_w = 180, 320

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)

        self.temp_frame = cv2.resize(self.pix_lines, (im_w,im_h))

        # compute hog and transpose so that shape matches the database features
        self.features = hog.compute(self.temp_frame).T

    def retrieve_a_camera(self):
        # print(self.database_features.shape)
        # print(self.features.shape)

        # Find the nearest neighbour feature match
        result, _ = self.flann.nn(self.database_features, self.features, 1, algorithm="kdtree", trees=16, checks=64)
        retrieved_index = result[0]
        # print("Retrieved index: "+str(retrieved_index))

        # For faster testing on the 16.jpg image from SCCvSD repo
        # retrieved_index = 46736

        # Find the camera that corresponds to the feature set that has been found to match
        retrieved_camera_data = self.database_cameras[retrieved_index]

        # Determine camera variables
        u, v, fl = retrieved_camera_data[0:3]
        rod_rot = retrieved_camera_data[3:6]
        cc = retrieved_camera_data[6:9]
        retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

        # Which homography to use? One corrected for the field or not (seems to mess with visualization, for calculation the first one might be better)
        retrieved_homography = IouUtil.template_to_image_homography_uot(retrieved_camera, self.template_h, self.template_w)
        # retrieved_homography = retrieved_camera.get_homography()

        # Turn the camera to an image with a template
        self.retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, self.model_points, self.model_line_index, im_h=self.frame_height, im_w=self.frame_width, line_width=4)
        self.pix_lines = cv2.resize(self.pix_lines, (1280, 720), interpolation=cv2.INTER_CUBIC)[:, :, None]
        
        # Refine using lucas kanade algorithm
        dist_threshold = 50
        query_dist = SyntheticUtil.distance_transform(self.pix_lines)
        retrieved_dist = SyntheticUtil.distance_transform(self.retrieved_image)
        query_dist[query_dist > dist_threshold] = dist_threshold
        retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

        warp = SyntheticUtil.find_transform(retrieved_dist, query_dist)
        
        # Use the homography to refine the found image to validate correctness.
        self.refined_retrieved_image = cv2.warpPerspective(self.retrieved_image, warp, (1280, 720))

        # This is the homography mapping the field to a top down view
        self.homography = warp@retrieved_homography

        # For validation: warp an image to top view:
        # self.normalized_image = cv2.warpPerspective(self.original, np.linalg.inv(self.homography),(1280,720))

    def visualize(self):
        cv2.imshow('adapted', self.pix_lines)
        cv2.imshow('Edge image of retrieved camera', self.retrieved_image)
        # cv2.imshow('original', self.original)
        # cv2.imshow('resized', self.temp_frame)
        cv2.imshow('refined', self.refined_retrieved_image)
        cv2.waitKey(40000)

    def save(self):
        cv2.imwrite('./output_images/{}_original.jpg'.format(self.i), self.original)
        cv2.imwrite('./output_images/{}_pix2pix.jpg'.format(self.i), self.pix_lines)
        # cv2.imwrite('./output_images/{}_retrieved.jpg'.format(self.i), self.retrieved_image)
        cv2.imwrite('./output_images/{}_refined.jpg'.format(self.i), self.refined_retrieved_image)
        # cv2.imwrite('./output_images/{}_normalized.jpg'.format(self.i), self.normalized_image)
        # cv2.imwrite('./output_images/{}_unwarped_image.jpg'.format(self.i), self.unwarped_image)
        

