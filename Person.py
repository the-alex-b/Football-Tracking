import cv2
import numpy as np
from random import randrange

class Person:
    _COUNTER = 0 
    def __init__(self, i, keypoints, overlay, homography):
        self.id = Person._COUNTER
        Person._COUNTER += 1

        self.kp = keypoints
        self.color = (randrange(0, 255),randrange(0, 255),randrange(0, 255))
        self.line_thickness = 2
        self.i = i
        self.homography = homography
        self.center_of_gravity = (0,0) 

        self.old_homographies = {}
        self.old_keypoints = {}
        self.old_coordinates = {}
        
        # Tracking variables
        self.cutoff_point = 8000
        self.allowed_missed_matches = 10
        self.missed_matches = 0
        self.tracking_lost = False
        
        # TODO:
        self.team = "A" #A, B or REF
        self.normalized_coordinates = None


        # Detect color of shirt
        self.detect_shirt_color(overlay)

    def find_best_next_keypoints(self, i, options, overlay):
        self.i = i

        if self.tracking_lost == False:
            differences = []

            # Kinda hacky.. should be made more efficient probably
            for option in options:
                res = np.subtract(self.kp, option)
                differences.append(np.sum(res**2))
            

            differences = np.array(differences)
            # print(differences)
            best_match_index = np.argmin(differences)

            # Calculate the remaining options to pass through
            remaining_options = np.delete(options, best_match_index,0)

            # print(len(remaining_options))
            # print(differences)
            
            # print(differences[best_match_index])

            if differences[best_match_index] < self.cutoff_point:
                self.missed_matches = 0
                self.update_keypoints(options[best_match_index])
                self.detect_shirt_color(overlay)
                self.draw_on_image(overlay)

            else:
                self.missed_matches = self.missed_matches + 1
                if self.missed_matches > self.allowed_missed_matches:
                    self.tracking_lost = True

            # Return the remaining options
            return remaining_options
        else:
            # print("Tracking lost")

            # Return options without taking one
            return options

    def update_keypoints(self, new_keypoints):
        # Archive keypoints and center of gravity
        self.old_homographies[self.i] = self.homography
        self.old_keypoints[self.i] = self.kp
        self.old_coordinates[self.i] = [self.center_of_gravity[0], self.center_of_gravity[1],1]

        # Update keypoints
        self.kp = new_keypoints

    def tracking_is_lost(self):
        self.tracking_lost = True    

    def detect_shirt_color(self, overlay):
        # Create a box centered on the torso
        xs = [self.kp[11][0], self.kp[12][0], self.kp[5][0], self.kp[6][0]]
        ys = [self.kp[11][1], self.kp[12][1], self.kp[5][1], self.kp[6][1]]
        
        square = [min(xs) , max(xs), min(ys), max(ys)]
        torso = overlay[square[2]:square[3], square[0]:square[1]]

        # cv2.imshow('cutout', torso)
        # print(torso)
        # print(torso.mean(1).mean(0))
        average_color = torso.mean(1).mean(0)

        # Set color to the average color of the torso
        self.color = average_color

    def update_homography(self,homography):
        self.homography = homography


    def draw_on_image(self, image):
        # Only draw of tracking is not lost

        if self.tracking_lost == False:
            # draw center of gravity (middle between feet)
            cog_x = int((self.kp[15][0] + self.kp[16][0])/2)
            cog_y = int((self.kp[15][1] + self.kp[16][1])/2)

            self.center_of_gravity = (cog_x,cog_y)

            image = cv2.circle(image, (cog_x,cog_y), 4, self.color,2)


            # draw joints
            for point in self.kp:
                coords = (int(point[0]), int(point[1]))
                image = cv2.circle(image, coords, 2, self.color, 2) 

            # draw limbs
            # leg1
            image = cv2.line(image, (self.kp[15][0], self.kp[15][1]), (self.kp[13][0], self.kp[13][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[13][0], self.kp[13][1]), (self.kp[11][0], self.kp[11][1]), self.color, self.line_thickness)

            # # leg2
            image = cv2.line(image, (self.kp[16][0], self.kp[16][1]), (self.kp[14][0], self.kp[14][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[14][0], self.kp[14][1]), (self.kp[12][0], self.kp[12][1]), self.color, self.line_thickness)

            # # waist
            image = cv2.line(image, (self.kp[11][0], self.kp[11][1]), (self.kp[12][0], self.kp[12][1]), self.color, self.line_thickness)

            # # shoulder line
            image = cv2.line(image, (self.kp[5][0], self.kp[5][1]), (self.kp[6][0], self.kp[6][1]), self.color, self.line_thickness)

            # # arm 1
            image = cv2.line(image, (self.kp[9][0], self.kp[9][1]), (self.kp[7][0], self.kp[7][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[7][0], self.kp[7][1]), (self.kp[5][0], self.kp[5][1]), self.color, self.line_thickness)

            # # arm 2
            image = cv2.line(image, (self.kp[10][0], self.kp[10][1]), (self.kp[8][0], self.kp[8][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[8][0], self.kp[8][1]), (self.kp[6][0], self.kp[6][1]), self.color, self.line_thickness)

            # Sides
            image = cv2.line(image, (self.kp[11][0], self.kp[11][1]), (self.kp[5][0], self.kp[5][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[12][0], self.kp[12][1]), (self.kp[6][0], self.kp[6][1]), self.color, self.line_thickness)

            # Head?
            image = cv2.line(image, (self.kp[1][0], self.kp[1][1]), (self.kp[2][0], self.kp[2][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[2][0], self.kp[2][1]), (self.kp[3][0], self.kp[3][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[3][0], self.kp[3][1]), (self.kp[4][0], self.kp[4][1]), self.color, self.line_thickness)
            image = cv2.line(image, (self.kp[4][0], self.kp[4][1]), (self.kp[1][0], self.kp[1][1]), self.color, self.line_thickness)

            cv2.putText(    
                image,
                str(self.id),
                tuple(e+5 for e in self.center_of_gravity),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255,0,0))




        return image

    
