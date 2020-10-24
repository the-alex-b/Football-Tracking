import cv2
import numpy as np
from random import randrange

class Person:
    _COUNTER = 0 
    def __init__(self, i, keypoints, overlay, homography):
        self.id = Person._COUNTER
        # Person._COUNTER += 1

        self.kp = keypoints
        self.color = (255,255,255)
        self.colors = (255,255,255)
        self.line_thickness = 2
        self.i = i

        self.homography = homography
        self.inverse_homography =  np.linalg.inv(homography)
        
        self.center_of_gravity = (0,0) 
        self.normalized_center_of_gravity = (0,0)

        self.old_homographies = {}
        self.old_keypoints = {}
        self.old_coordinates = {}
        
        # Tracking variables
        self.cutoff_point = 8000
        self.allowed_missed_matches = 1
        self.missed_matches = 0
        self.tracking_lost = False

        self.team = None

        self.on_field = None
        
        # TODO:
        self.normalized_coordinates = None


        # Detect color of shirt
        self.determine_normalized_center_of_gravity()
        self.check_if_on_field()
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
        self.determine_normalized_center_of_gravity()

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
        # Instead of updating at every frame, create a rolling average of the color
        self.colors = np.vstack([self.colors, average_color])
        self.color = np.mean(self.colors, axis=0)

    def update_homography(self,homography):
        self.homography = homography

    def set_team(self, team): 
        self.team = team


    def determine_normalized_center_of_gravity(self):
        self.center_of_gravity = (int((self.kp[15][0] + self.kp[16][0])/2), int((self.kp[15][1] + self.kp[16][1])/2))
        
        homogenous_coordinates = [self.center_of_gravity[0], self.center_of_gravity[1], 1]
        transformed = self.inverse_homography@homogenous_coordinates        
        normalized_coordinates = transformed[:2]/transformed[2]

        self.normalized_center_of_gravity = (normalized_coordinates[0], normalized_coordinates[1])
        # print(self.normalized_center_of_gravity)



    def check_if_on_field(self):
        if 0-5 < self.normalized_center_of_gravity[0] < 115+5 and 0-5 < self.normalized_center_of_gravity[1] < 74+5:
            # print('true detection')
            self.on_field = True
            
            # If player on field, increase counter
            Person._COUNTER += 1
        else:
            # print("false detection")
            self.on_field = False



    def draw_on_image(self, image):
        # Only draw of tracking is not los  t
        # print(self.on_field)
        if self.tracking_lost == False and self.on_field == True :    
            # draw center of gravity (middle between feet)
            # cog_x = int((self.kp[15][0] + self.kp[16][0])/2)
            # cog_y = int((self.kp[15][1] + self.kp[16][1])/2)

            # self.center_of_gravity = (cog_x,cog_y)

            # # Detect the 'top down' coordinates. If these are off-field set self.false_detection = True
            # temp_c = [cog_x, cog_y, 1]
            # Hinv = np.linalg.inv(self.homography)
            # # print(Hinv)

            # normalized_cog = Hinv@temp_c
            # coords = normalized_cog[:2]/normalized_cog[2]

            # if 0 < coords[0] < 115 or 0 < coords[1] < 74:
            #     print('true detection')
            #     print(coords)
            #     return image
            # self.normalized_center_of_gravity = 

            image = cv2.circle(image, self.center_of_gravity, 4, self.color,2)


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

            #cv2.putText(    
             #   image,
             #   str(self.team.getname()),
             #   tuple(e+10 for e in self.center_of_gravity),
            #  cv2.FONT_HERSHEY_SIMPLEX,
            #    0.3,
             #    (255,0,0))



        return image

    
