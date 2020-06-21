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