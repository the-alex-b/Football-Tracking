import cv2
import torch
import numpy as np

# Import classes
from Logger import Logger
from Person import Person
from Team import Team
from TeamDetector import TeamDetector

# Various helper functions
from utilities import write_extracted_frames_to_disk, load_extracted_frames_from_disk, smooth_homographies, smooth_traj_kalman 

# Initialize the logger
logger = Logger("Main runtime")
logger.log("Starting Analysis")

''' --- Run extraction? ---
Determine wheter the extraction should be ran or data should be loaded from disk. 
This should probably be turned into an argument that can be supplied to the main function
'''
run_extraction = False

if run_extraction == True:
    # Importgitdetectors
    from HomographyDetector.HomographyDetector import HomographyDetector
    from PlayerDetector.PlayerDetector import PlayerDetector
    from ExtractedFrame import ExtractedFrame


    logger.log("Running extraction step")

    '''--- Determining GPU availabity ---

    This availability is passed on the detectors that are initialized in the next step. Runnign with gpu will be much more efficient and fast ofcourse. Future detectors should take this variable in account so code can be run both with and without GPU.

    '''
    GPU_AVAILABILITY = torch.cuda.is_available()
    print("GPU availabiltiy is: {}".format(GPU_AVAILABILITY))
    logger.log("Determined GPU availability")


    ''' --- Initialization of detectors ---

    We initialize various detectors required to extract data from the frames. Now we create a homography detector and a player detector. Later on this should be extended with:

    *Ball detector
    *Score detector
    *Time detector
    *Is_this_a_football_scene? detector
    *And probably a few more :)

    All computation should happen on the detector instances so we can easily extract data on a per frame basis.

    '''
    logger.log("Initializing detectors")
    homography_detector = HomographyDetector(useGpu=GPU_AVAILABILITY)
    player_detector = PlayerDetector(useGpu=GPU_AVAILABILITY)
    # Ball detector
    # Score detector
    # Time detector
    # etc...

    logger.log("Detectors initalized")


    ''' --- Reading the video stream and setting runtime parameters ---
    Below we open the video and set parameters for analysis. 
    '''
    # Select a video to analyze
    stream = cv2.VideoCapture('./input_footage/video/1080_HQ.mp4')

    # Set start & end boundaries and snip stream
    start_time_in_ms = 0
    end_time_in_ms = 10000
    stream.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)

    # Set frame limits
    skip_frames_modulo = 1
    max_number_of_frames = 600

    # Create empty array that will hold properties of analyzed frames

    # Create frame counter i
    i = 0


    ''' --- Start the video loop ---
    The code below will loop through all frames that are returned from the video stream. Once the stream finished the loop breaks and analysis will be finished. An array to hold the data from frame extraction is initialized below.

    Each frame is going through the following steps:

    1. Adaption: Frame is manipulated to match required criteria for further analysis.

    2. Extraction: Data is extracted from the frame -> All relevant information for further analysis should be turned into 'data' and stored on the ExtractedFrame object. These objects are collected in the extractedFrames array.

    3. Storage of frame


    '''
    extractedFrames = []

    while (True):
        frameLogger = Logger("Frame Loop {}".format(i))
        ret, frame = stream.read()
        
        # Break if end of stream or max number of frames reached
        if not ret or i>=max_number_of_frames:
            break

        # Break if stream manually interrupted
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Analyze the frame
        else:
            ''' --- 1. Adaption Step ----
            Resize the frame for homography algorithm

            '''
            frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_CUBIC)

            ''' --- 2. Extraction of Data ---
            In the following steps we invoke the detector objects we have initalized earlier on extract data from the frames in the video stream. The extracted data is then added to the extractedFrames array. This array is used for analysis, smoothing and further calculations. 
            
            Furthermore this array can be stored on disk so we can skip the extraction step during future development.
            '''
            feet_coordinates, detections = player_detector.detect_players(frame)


            homography = homography_detector.detect_homography(frame)

        
            ''' --- 3. Storage ---
            Below we will create an extractedFrame instance with the data that has been extracted and add it to the extractedFrames array for storage later on.
            '''
            extractedFrame = ExtractedFrame(i,homography,feet_coordinates,detections)
            extractedFrames.append(extractedFrame)
            
            
            
            
            # Increase frame counter with 1
            i = i+ 1
            frameLogger.log("Analysis done")

    # Close the stream and windows if opened
    stream.release()
    cv2.destroyAllWindows()

    ''' --- Write/load extractedFrames to/from disk ---
    Below we will write the extracted frames to disk.

    '''
    write_extracted_frames_to_disk(extractedFrames)

else:
    logger.log("Skipping extraction step and loading extractedFrames from disk")
    extractedFrames = load_extracted_frames_from_disk('_fullrun_fullkeypoints')

    # Set i so full run calculations can be made.
    i = len(extractedFrames)
    print(i)


''' --- Smoothing and overall analysis ---

Below we will analyse the data from the extractedFrames. 
Here we will perform steps like coordinate normalization, smoothing, tracking players over multiple frames etc.

'''

# TODO : Calculate normalized coordinates of detected persons
# for frame in extractedFrames:
#     frame.calculate_normalized_coordinates()

smoothedExtractedFrames = smooth_homographies(extractedFrames, 51, 3)


# Experimental visualization stuff -- Need to clean, apply, improve etc.

# Get the pitch template
path = './input_footage/picture/PitchTemplate.png'
img = cv2.imread(path, 1)

# We open a new stream that will allow us to visualize all tracked players on the original video for reference
stream2 = cv2.VideoCapture('./input_footage/video/1080_HQ.mp4')

# Initialize an empty array that will collect all of our identified players
tracked_persons = []

# Initalize an empty array to hold our images for saving
img_array = []
frames = []

# initialize frame counter
j = 0

# Loop through the video stream same way we did before
while (True):
    ret, frame = stream2.read()
    
    # Break if end of stream or max number of frames reached
    if not ret:
        break

    # Break if stream manually interrupted
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Analyze the frame
    else:
        ''' --- 1. Adaption Step ----
        Resize the frame for homography algorithm

        '''
        frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
        template_h = 74
        template_w = 115
        scale = 1
        img_resized = cv2.resize(img,dsize=(scale*template_w, scale*template_h), interpolation=cv2.INTER_AREA)
        
        warpedImg  = cv2.warpPerspective(img_resized, smoothedExtractedFrames[j].smoothed_homography, (1280,720))

        #  add the original footage to the overlay
        overlay = cv2.addWeighted(frame,0.5,warpedImg,0.3,0)

        # Detect all
        # for kp in extractedFramesSmoothed[i].detections:
        #     person = Person(kp)
        #     person.update_keypoints(kp)
        
        options = smoothedExtractedFrames[j].detections

        for person in tracked_persons:
            person.update_homography(smoothedExtractedFrames[j].smoothed_homography)

            if len(options) > 0:
                options = person.find_best_next_keypoints(j, options, overlay)
                overlay = person.draw_on_image(overlay)
                # No more options available, tracking seems to be lost..
            else:
                person.tracking_is_lost()

        # Create new Person for all detections that havent been matched
        # Only look at the detections that have at least 1 3rd coordinate non-zero
        for o in  [o for o in options if np.sum(o[:,2]) > 0 ]:
        # for o in options:
            tp = Person(i, o, overlay, smoothedExtractedFrames[j].smoothed_homography)
            overlay = tp.draw_on_image(overlay)


            # Only append if the Person is on field (and not random detection in the stands)
            if tp.on_field == True:
                tracked_persons.append(tp)

        

        cv2.imshow('overlay', overlay)
        # cv2.imshow('normalized', normalized)
        
        print(len(tracked_persons))

        j = j + 1

        img_array.append(overlay)

# Uncomment if you want to save a video
# print("Saving video")
# size = (img_array[0].shape[1], img_array[0].shape[0])
# print(size)
# out = cv2.VideoWriter('./output_videos/tracked_players_color.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)    

# for img in img_array:
#     out.write(img)
# out.release()


## Now that the tracking has been done based on the frames, additional steps can be performed, e.g. the smoothing of the 2d trajectories

# Team detection 
teams = [
    Team(0, 'Ajax', np.array([255, 102, 0])) # red 
    ,Team(1, 'Getafe', np.array([0, 102, 255])) # blue 
    ,Team(2, 'Referees/Keepers', np.array([255, 255, 0])) # yellow
]
td = TeamDetector(teams)
detected_teams = td.get_teams(tracked_persons)


###

# smoothing of the 2d trajectories
smoothed_trajs = []
for player_idx in range(len(tracked_persons)):
    # convert to 2d coordinates
    twod_trajs = {}
    for i in tracked_persons[player_idx].old_homographies.keys(): 
        Hinv = np.linalg.inv(tracked_persons[player_idx].old_homographies[i])
        twod_traj = tracked_persons[player_idx].old_coordinates[i]
        twod_trajs[i] = np.matmul(twod_traj,Hinv.T)[:2]/np.matmul(twod_traj,Hinv.T)[2]
    # reformat
    xs = []
    ys = []
    for e in twod_trajs.keys(): 
        xs.append(twod_trajs[e][0])
        ys.append(twod_trajs[e][1])

    measurements = list(zip(xs,ys))
    initial_state_x = list(zip(xs,ys))[0][0]
    initial_state_y = list(zip(xs,ys))[0][1]
    # smooth 
    smoothed_trajs.append(smooth_traj_kalman(measurements, initial_state_x, initial_state_y, observation_uncertainty=100)[:,[0,2]]) # 0, 2 Depends on the shape of the observation_matrix



'''--- Finalize analysis ---
Log full run statistics
'''
logger.log("Analysis done")
logger.print_average(i)






