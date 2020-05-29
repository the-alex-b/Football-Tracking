import cv2
import torch
from HomographyDetector.HomographyDetector import HomographyDetector
from PlayerDetector.PlayerDetector import PlayerDetector
from Logger import Logger

# Initialize the logger
logger = Logger("Main runtime")

logger.log("Starting Analysis")

GPU_AVAILABILITY = torch.cuda.is_available()
print("GPU availabiltiy is: {}".format(GPU_AVAILABILITY))
logger.log("Determined GPU availability")

# Initialize detectors
logger.log("Initializing detectors")

homography_detector = HomographyDetector(useGpu=GPU_AVAILABILITY)
player_detector = PlayerDetector(useGpu=GPU_AVAILABILITY)

# ball detector
# score and time detector?
# is this a football scene detector?
# More...
logger.log("Detectors initalized")


# Select a video to analyze
stream = cv2.VideoCapture('./input_footage/video/1080_HQ.mp4')

# Set start & end boundaries and snip stream
start_time_in_ms = 0
end_time_in_ms = 10000
stream.set(cv2.CAP_PROP_POS_MSEC, start_time_in_ms)

# Set frame limits
skip_frames_modulo = 1
max_number_of_frames = 100

# Create empty array that will hold properties of analyzed frames
analyzed_frames = []

# Create frame counter i
i = 0


# Video loop
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
        # Resize the frame for homography algorithm
        frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_CUBIC)

        # Detect players
        feet_coordinates = player_detector.detect_players(frame)

        # Draw circles for sanity check
        # for c in feet_coordinates:
            # cv2.circle(frame,(int(c[0]), int(c[1])), 5, (0,0,255), 3)


        # Detect homography
        homography = homography_detector.detect_homography(frame)
        print(homography)
        print(feet_coordinates)
        # Append results to analyzed_frames



        # Visualize
        # cv2.imshow("frame", frame)
        
        # Increase frame counter with 1
        i = i+ 1
    
        frameLogger.log("Analysis done")








