import cv2
import torch

# Import classes
from Logger import Logger

# Various helper functions
from utilities import write_extracted_frames_to_disk, load_extracted_frames_from_disk, smooth_homographies 

# Initialize the logger
logger = Logger("Main runtime")
logger.log("Starting Analysis")

''' --- Run extraction? ---
Determine wheter the extraction should be ran or data should be loaded from disk. This should probably be turned into an argument that can be supplied to the main function
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
    max_number_of_frames = 5

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
            feet_coordinates = player_detector.detect_players(frame)
            homography = homography_detector.detect_homography(frame)

        
            ''' --- 3. Storage ---
            Below we will create an extractedFrame instance with the data that has been extracted and add it to the extractedFrames array for storage later on.
            '''
            extractedFrame = ExtractedFrame(i,homography,feet_coordinates)
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
    extractedFrames = load_extracted_frames_from_disk('_50')

    # Set i so full run calculations can be made.
    i = len(extractedFrames)


''' --- Smoothing and overall analysis ---

Below we will analyse the data from the extractedFrames. Here we will perform steps like coordinate normalization, smoothing, tracking players over multiple frames etc.

'''

# TODO : Calculate normalized coordinates of detected persons
# for frame in extractedFrames:
    # frame.calculate_normalized_coordinates()

smoothedExtractedFrames = smooth_homographies(extractedFrames)


for frame in smoothedExtractedFrames:
    print(frame.homography)
    print(frame.smoothed_homography)


'''--- Finalize analysis ---
Log full run statistics
'''
logger.log("Analysis done")
logger.print_average(i)






