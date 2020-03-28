import cv2
import numpy as np
import pyflann

from PIL import Image

from SCCvSD_Utils.synthetic_util import SyntheticUtil
from SCCvSD_Utils.iou_util import IouUtil
from SCCvSD_Utils.projective_camera import ProjectiveCamera

import playerdetection_maskrcnn as pldec
import twodvisualisation as twodvis 
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

        # Player detection, find all person coordinates in the frame
        # ---------------
        pldec.config_tf()
        result = pldec.detectplayersbox(frame) # returns list of detections over multiple frames
        # pldec.save_result(result[0], frame, self.i) # only 1 frame being processed
        self.playercoos = pldec.findplayerscoos(result[0])
        # ---------------
        # Visualize player coordinates as circles below players:
        for c in self.playercoos:
            cv2.circle(self.original,(int(c[0]), int(c[1])), 5, (0,0,255), 3)
        

        # Call all functions that extract data from the frame to determine homography
        self.extract_pitch_lines()
        self.create_line_image()
        self.generate_hog_features()
        self.retrieve_a_camera()

        # Use inverse of the homography to calculate topview coordinates of players
        # This should be implemented in a separate function: create person Class? To also determine teamA/B or referee?
        self.warped_coords = [] 
        for c in self.playercoos:
            c_mat = np.array([[c[0]],[c[1]],[1]])

            hom_cords = np.linalg.inv(self.homography)@c_mat

            self.warped_coords.append([hom_cords[0][0]/hom_cords[2][0],hom_cords[1][0]/hom_cords[2][0]])

        twodvis.twodvisualisation(self.warped_coords, self.i, self.template_w, self.template_h)

        # Warped coords belonging to worldcup image 16 (placeholder for faster development)
        # self.warped_coords = [[92.01287027834677, 28.786941200521106], [94.66302995069726, 67.04521116566309], [95.50522352407108, 60.511763034888595], [97.82874971791972, 42.39573348700747], [91.24491420783876, 56.98150085508616], [113.06896849777085, 38.37172302960884], [90.4490186162751, 62.02393262415963], [80.46709610719846, 44.462272402932655], [88.0790275606092, 19.51399275883541], [97.8791538613156, 35.674698750579715], [81.00846673234102, 38.831727152810615], [78.82873504463414, 26.249622960034305], [87.45029855143562, 36.598743959497696], [99.58510108821706, 33.51209547287334], [78.72732178814238, 20.72997452258794]
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

        self.original = cv2.resize(self.original,(1280, 720), interpolation=cv2.INTER_CUBIC)
        print(self.original.shape)
        print(self.refined_retrieved_image.shape)

        self.overlayed_image = cv2.addWeighted(self.original,0.4,self.refined_retrieved_image,0.1,0)

        # This is the homography mapping the field to a top down view
        self.homography = warp@retrieved_homography

        # For validation: warp an image to top view:
        self.normalized_image = cv2.warpPerspective(self.original, np.linalg.inv(self.homography),(1280,720))

    def visualize(self):
        cv2.imshow('adapted', self.pix_lines)
        cv2.imshow('Edge image of retrieved camera', self.retrieved_image)
        # cv2.imshow('original', self.original)
        # cv2.imshow('resized', self.temp_frame)
        cv2.imshow('refined', self.refined_retrieved_image)
        cv2.waitKey(40000)

    def save(self):
        # cv2.imwrite('./output_images/{}_original.jpg'.format(self.i), self.original)
        # cv2.imwrite('./output_images/{}_pix2pix.jpg'.format(self.i), self.pix_lines)
        # cv2.imwrite('./output_images/{}_retrieved.jpg'.format(self.i), self.retrieved_image)
        # cv2.imwrite('./output_images/{}_refined.jpg'.format(self.i), self.refined_retrieved_image)
        # cv2.imwrite('./output_images/{}_normalized.jpg'.format(self.i), self.normalized_image)
        # cv2.imwrite('./output_images/{}_unwarped_image.jpg'.format(self.i), self.unwarped_image)
        cv2.imwrite('./output_images/{}_overlayed_image.jpg'.format(self.i), self.overlayed_image)
        

