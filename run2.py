import torch
from HomographyDetector.HomographyDetector import HomographyDetector
from PlayerDetector.PlayerDetector import PlayerDetector

GPU_AVAILABILITY = torch.cuda.is_available()
print("GPU availabiltiy is: {}".format(GPU_AVAILABILITY))

# Create detectors
# homography_detector = HomographyDetector(useGpu=GPU_AVAILABILITY)
player_detector = PlayerDetector(useGpu=GPU_AVAILABILITY)