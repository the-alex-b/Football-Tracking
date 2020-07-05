import pickle
from pprint import pprint
import scipy.signal
import numpy as np


'''
For now we use pickle to store the extracted frames on disk. Is this optimal? Or should we change it
'''
def write_extracted_frames_to_disk(extractedFrames):
    pickle.dump(extractedFrames, open("./storage/extracted_frames/latest_stored_extracted_frames.p", "wb"))



def load_extracted_frames_from_disk(added=''):
    storedExtractedFrames= pickle.load(open("./storage/extracted_frames/latest_stored_extracted_frames{}.p".format(added), "rb"))
    
    return storedExtractedFrames

''' Below we unpack all the extracted frames from a scene and smooth the homographies that were found to create a less jittery result.
To smooth we use an savgol_filter with parameters window and poly. It is still unclear wether this smoothing is optimal and what parameters should be used. --> Move to extracted frame object!!!
'''
def smooth_homographies(extractedFrames, window, poly):
    
    unpackedMatrix = {}

    # Initialize a dict with dict that will contain the unpacked arrays
    for i in range(3):
            unpackedMatrix[i] = {}
            for j in range(3):
                unpackedMatrix[i][j] = []
    

    # Fill the arrays with unpacked arrays from the original matrix
    for frame in extractedFrames:
        h = frame.homography
        for i in range(3):
            for j in range(3):
                # print(h[i][j])
                unpackedMatrix[i][j].append(h[i][j])
    
    # Smoothen the different arrays
    for i in range(3):
        for j in range(3):
            unpackedMatrix[i][j] = scipy.signal.savgol_filter(unpackedMatrix[i][j], window, poly)

    # Put the smoothed homographies back in the extractedFrames object
    for i in range(len(extractedFrames)):
        extractedFrames[i].smoothed_homography = np.array([
            [unpackedMatrix[0][0][i], unpackedMatrix[0][1][i], unpackedMatrix[0][2][i]],
            [unpackedMatrix[1][0][i], unpackedMatrix[1][1][i], unpackedMatrix[1][2][i]],
            [unpackedMatrix[2][0][i], unpackedMatrix[2][1][i], unpackedMatrix[2][2][i]]
        ])
    
    return extractedFrames


import numpy as np 
from pykalman import KalmanFilter


def smooth_traj_kalman(measurements, initial_state_x, initial_state_y, observation_uncertainty=10):

  initial_state_mean = [initial_state_x,
                        0,  # x velocity
                        initial_state_y,
                        0   # y velocity
                        ]

  transition_matrix = [[1, 1, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1]]

  observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]

  kf1 = KalmanFilter(transition_matrices = transition_matrix,
                     observation_matrices = observation_matrix,
                     initial_state_mean = initial_state_mean)

  kf1 = kf1.em(measurements, n_iter=5)

  kf2 = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    observation_covariance = observation_uncertainty*kf1.observation_covariance,
                    em_vars=['transition_covariance', 'initial_state_covariance'])

  kf2 = kf2.em(measurements, n_iter=5)
  (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)
  return smoothed_state_means                   