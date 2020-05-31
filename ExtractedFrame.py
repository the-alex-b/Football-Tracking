


class ExtractedFrame():
    '''
    This class contains all data that is extracted from a frame,
    '''
    def __init__(self, frame_number, extracted_homography, detected_persons):
            self.frame_number = frame_number
            self.homography = extracted_homography
            self.persons = detected_persons
    