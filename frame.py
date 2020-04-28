import cv2
import numpy as np
import sys
from PIL import Image

from SCCvSD_Utils.synthetic_util import SyntheticUtil
from SCCvSD_Utils.iou_util import IouUtil
from SCCvSD_Utils.projective_camera import ProjectiveCamera

import playerdetection_maskrcnn as pldec
import twodvisualisation as twodvis 
from utilities.ANN import NNSearcher

import torchvision.transforms as transforms

class Frame:
    def __init__(self, frame, database_features, database_cameras, model_points, model_line_index, pix2pix_model, nnsearcher:NNSearcher, identifier):
        # Initialize variables
        self.frame = frame
        self.original = frame
        self.features = None
        self.i = identifier
        self.resolution = (frame.shape[1],frame.shape[0])
        self.database_features = database_features
        self.database_cameras = database_cameras
        self.model_points = model_points
        self.model_line_index = model_line_index

        self.pix2pix_model = pix2pix_model

        self.template_h = 74
        self.template_w = 115

        self.nnsearcher = nnsearcher
        self.picstosave = []
        

    def process(self): 
        # Player detection, find all person coordinates in the frame
        # ---------------
        result = pldec.detectplayerskeypoints(self.frame)[0] # returns list of detections over multiple frames
        # pldec.save_result(result[0], frame, self.i) # only 1 frame being processed
        #self.detectionfeatures = result['features'] # general detection features 
        self.playersfeetcoos = pldec.findplayersfeetcoos(result)
        self.playerstorsocoos = pldec.findplayerstorsocoos(result)
        self.playerkeypoints = pldec.findplayerkeypointsall(result)
        # ---------------
        # Visualize player coordinates as circles below players:
        for c in self.playersfeetcoos:
            cv2.circle(self.original,(int(c[0]), int(c[1])), 5, (0,0,255), 3)
        self.picstosave.append(('original',self.original))
        

        # Call all functions that extract data from the frame to determine homography
        pix_lines = self.extract_pitch_lines(self.frame, self.pix2pix_model, load_size = 256)
        pix_lines = self.plot_lines_on_image(pix_lines)
        features = self.generate_hog_features(pix_lines)
        pix_lines, retrieved_image, retrieved_homography = self.retrieve_a_camera(pix_lines, features)
        self.final_homography = self.refine_camera(pix_lines, retrieved_image, retrieved_homography)
        self.twod_coords = self.calculate_2d_coordinates(self.final_homography, self.playersfeetcoos)
        self.create_normalizedview(self.original, self.final_homography)
        self.create_overlayedview()
        # twodvis.twodvisualisation(self.twod_coords, self.i)  #---> probably does not belong here
        # Warped coords belonging to worldcup image 16 (placeholder for faster development)
        # self.warped_coords = [[92.01287027834677, 28.786941200521106], [94.66302995069726, 67.04521116566309], [95.50522352407108, 60.511763034888595], [97.82874971791972, 42.39573348700747], [91.24491420783876, 56.98150085508616], [113.06896849777085, 38.37172302960884], [90.4490186162751, 62.02393262415963], [80.46709610719846, 44.462272402932655], [88.0790275606092, 19.51399275883541], [97.8791538613156, 35.674698750579715], [81.00846673234102, 38.831727152810615], [78.82873504463414, 26.249622960034305], [87.45029855143562, 36.598743959497696], [99.58510108821706, 33.51209547287334], [78.72732178814238, 20.72997452258794]
        # Visualize or save the frame:
        # self.visualize()
        self.save(self.picstosave)


    def calculate_2d_coordinates(self, homography, coos): 
        # Use inverse of the homography to calculate topview coordinates of players
        warped_coords = [] 
        for c in coos:
            c_mat = np.array([[c[0]],[c[1]],[1]])
            hom_cords = np.linalg.inv(homography)@c_mat
            warped_coords.append([hom_cords[0][0]/hom_cords[2][0],hom_cords[1][0]/hom_cords[2][0]])
        return warped_coords


    # Pix2Pix model to extract lines and field area 
    def extract_pitch_lines(self, frame, pix2pix_model, load_size=256):
        
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

        pix2pix_model.set_input(input_image)
        pix2pix_model.test()
        
        vis = pix2pix_model.get_current_visuals()
        
        # self.pix_original = vis['real_A']
        # self.pix_no_crowd = vis['fake_C']
        pix_lines = vis['fake_D']
        return pix_lines

    def plot_lines_on_image(self, pix_lines):
        pix_lines = cv2.cvtColor(pix_lines, cv2.COLOR_BGR2GRAY)
        pix_lines = cv2.threshold(pix_lines, 127, 255, cv2.THRESH_BINARY)[1]
        #self.save('pix2pix',pix_lines)
        self.picstosave.append(('pix2pix',pix_lines))
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
        # For faster testing on the 16.jpg image from SCCvSD repo
        # retrieved_index = 46736

        # Find the camera that corresponds to the feature set that has been found to match

        retrieved_index = self.nnsearcher.seek_nn(features)
        print("Retrieved index: "+str(retrieved_index)) 
        retrieved_camera_data = self.database_cameras[retrieved_index]

        u, v, fl = retrieved_camera_data[0:3]
        rod_rot = retrieved_camera_data[3:6]
        cc = retrieved_camera_data[6:9]
        retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)

        # Which homography to use? One corrected for the field or not (seems to mess with visualization, for calculation the first one might be better)
        retrieved_homography = IouUtil.template_to_image_homography_uot(retrieved_camera, self.template_h, self.template_w)
        # retrieved_homography = retrieved_camera.get_homography()

        # Turn the camera to an image with a template
        retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, self.model_points, self.model_line_index, im_h=self.resolution[1], im_w=self.resolution[0], line_width=4)
        #self.save('retrieved',retrieved_image)
        self.picstosave.append(('retrieved',retrieved_image))
        
        pix_lines = cv2.resize(pix_lines, self.resolution, interpolation=cv2.INTER_CUBIC)[:, :, None]

        return pix_lines, retrieved_image, retrieved_homography

    def refine_camera(self, pix_lines, retrieved_image, retrieved_homography):         
        # Refine using lucas kanade algorithm
        dist_threshold = 50
        query_dist = SyntheticUtil.distance_transform(pix_lines)
        retrieved_dist = SyntheticUtil.distance_transform(retrieved_image)
        query_dist[query_dist > dist_threshold] = dist_threshold
        retrieved_dist[retrieved_dist > dist_threshold] = dist_threshold

        warp = SyntheticUtil.find_transform(retrieved_dist, query_dist)
        # This is the homography mapping the field to a top down view
        homography = warp@retrieved_homography

        self.refined_retrieved_image = cv2.warpPerspective(retrieved_image, warp, self.resolution)
        self.picstosave.append(('refined',self.refined_retrieved_image))
        return homography
        
    def create_overlayedview(self): 
        self.original = cv2.resize(self.original,self.resolution, interpolation=cv2.INTER_CUBIC)
        self.overlayed_image = cv2.addWeighted(self.original,0.4,self.refined_retrieved_image,0.1,0)
        self.picstosave.append(('overlayed_image',self.overlayed_image))

    def create_normalizedview(self, original, homography): 
        # Use the homography to refine the found image to validate correctness.
        # For validation: warp an image to top view:
        normalized_image = cv2.warpPerspective(original, np.linalg.inv(homography),self.resolution)
        self.picstosave.append(('normalized',normalized_image))
        return 

    def visualize(self, label, pic):
        cv2.imshow(label, pic)

    def save(self, tosave):
        for e in tosave: 
            cv2.imwrite('./output_images/{}_{}.jpg'.format(self.i, e[0]), e[1])
        

