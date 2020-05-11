import numpy as np 

from pykalman import KalmanFilter


def smooth_traj_kalman(k, fpp, was_detected):
  measurements = np.ma.array(fpp[k])
  measurements[~was_detected[k]] = np.ma.masked

  initial_state_mean = [measurements[was_detected[k], 0][0],
                        0,
                        measurements[was_detected[k], 1][0],
                        0]

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
                    observation_covariance = 10*kf1.observation_covariance,
                    em_vars=['transition_covariance', 'initial_state_covariance'])

  kf2 = kf2.em(measurements, n_iter=5)
  (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)
  return smoothed_state_means                          
