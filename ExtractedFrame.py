import numpy as np


class ExtractedFrame():
    '''
    This class contains all data that is extracted from a frame,
    '''
    def __init__(self, frame_number, extracted_homography, feet, detections):
            self.frame_number = frame_number
            self.homography = extracted_homography
            self.feet_coordinates = feet
            self.detections = detections
            self.smoothed_homography = None
    

    def calculate_normalized_player_coordinates(self):
        # Use inverse of the homography to calculate topview coordinates of players
        self.warped_feet_coordinates = [] 
        
        for c in self.feet_coordinates:
            c_mat = np.array([[c[0]],[c[1]],[1]])
            hom_cords = np.linalg.inv(self.homography)@c_mat
            self.warped_feet_coordinates.append([hom_cords[0][0]/hom_cords[2][0],hom_cords[1][0]/hom_cords[2][0]])

    